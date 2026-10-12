r"""Model evaluation: probabilities, threshold sweeps, ranking metrics, per-group views.

``MatchingModel.run_eval`` returns one number, the F1 at the 0.5 decision boundary.
For entity matching that is rarely the whole story: the operating threshold is
usually chosen for a target precision or recall, and accuracy often differs a lot
between easy and hard regions of the data. :func:`evaluate` computes everything from
one forward pass::

    from deepmatcher.evaluation import evaluate
    result = evaluate(model, valid_dataset, device='auto')
    result.f1, result.best_f1_threshold, result.roc_auc, result.pr_auc
    result.threshold_for_precision(0.99)      # lowest threshold reaching 99 % precision
    result.sweep                              # DataFrame, thresholds 0.01 .. 0.99
    result = evaluate(model, valid_dataset, group_by='jaro_winkler_similarity',
                      group_bin=0.1)
    result.groups                             # DataFrame, one row per 0.1 JW bin
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .data import MatchingIterator
from .runtime import select_device

THRESHOLDS = np.round(np.arange(1, 100) / 100, 2)


def _counts(labels, predicted):
    tp = int(((predicted == 1) & (labels == 1)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())
    tn = int(((predicted == 0) & (labels == 0)).sum())
    return tp, fp, fn, tn


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _sweep(probs, labels):
    rows = []
    for t in THRESHOLDS:
        predicted = (probs >= t).astype(np.int64)
        tp, fp, fn, _ = _counts(labels, predicted)
        p, r, f = _prf(tp, fp, fn)
        rows.append((float(t), p, r, f, int(predicted.sum())))
    return pd.DataFrame(rows, columns=['threshold', 'precision', 'recall', 'f1', 'matches'])


def _auc(fn, labels, probs):
    # AUCs are undefined with a single class; NaN is honest, an exception is not useful.
    if len(np.unique(labels)) < 2:
        return float('nan')
    return float(fn(labels, probs))


def _summary(probs, labels):
    from sklearn.metrics import average_precision_score, roc_auc_score
    predicted = (probs >= 0.5).astype(np.int64)
    tp, fp, fn, tn = _counts(labels, predicted)
    precision, recall, f1 = _prf(tp, fp, fn)
    sweep = _sweep(probs, labels)
    best = sweep.loc[sweep['f1'].idxmax()]
    return {
        'n': int(len(labels)), 'positives': int(labels.sum()),
        'precision': precision, 'recall': recall, 'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'best_f1_threshold': float(best['threshold']), 'best_f1': float(best['f1']),
        'roc_auc': _auc(roc_auc_score, labels, probs),
        'pr_auc': _auc(average_precision_score, labels, probs),
    }


@dataclass
class EvalResult:
    """Evaluation of one model on one labeled dataset.

    Attributes:
        probabilities, labels, ids: per-pair match probability, 0/1 label, pair id.
        precision, recall, f1: at the 0.5 threshold.
        tp, fp, fn, tn: confusion counts at 0.5.
        roc_auc, pr_auc: ranking metrics (average precision for ``pr_auc``); NaN when
            the dataset has a single class.
        sweep: DataFrame, one row per threshold 0.01 .. 0.99
            (``threshold, precision, recall, f1, matches``).
        best_f1_threshold, best_f1: the sweep threshold with maximal F1 (first one on
            ties) and that F1. Tuned on this data: report it on held-out data only.
        groups: per-group DataFrame (same metrics as the totals) or None.
        seconds: wall-clock of the forward passes.
    """
    probabilities: np.ndarray
    labels: np.ndarray
    ids: list
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int
    roc_auc: float
    pr_auc: float
    sweep: pd.DataFrame
    best_f1_threshold: float
    best_f1: float
    groups: Optional[pd.DataFrame] = None
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    def threshold_for_precision(self, target: float) -> Optional[float]:
        """Lowest swept threshold whose precision is >= ``target`` (so, the highest
        recall that still meets it), or None when no threshold reaches it."""
        ok = self.sweep[(self.sweep['precision'] >= target) & (self.sweep['matches'] > 0)]
        return float(ok['threshold'].iloc[0]) if len(ok) else None

    def threshold_for_recall(self, target: float) -> Optional[float]:
        """Highest swept threshold whose recall is >= ``target`` (so, the highest
        precision that still meets it), or None when no threshold reaches it."""
        ok = self.sweep[self.sweep['recall'] >= target]
        return float(ok['threshold'].iloc[-1]) if len(ok) else None

    def metrics(self) -> dict:
        """Scalar metrics as a JSON-able dict."""
        return {k: getattr(self, k) for k in (
            'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'tn', 'roc_auc', 'pr_auc',
            'best_f1_threshold', 'best_f1')}


def evaluate(model,
             dataset,
             device='auto',
             batch_size: int = 512,
             group_by: Optional[str] = None,
             group_bin: Optional[float] = None,
             groups: Optional[Sequence] = None,
             autocast_dtype: Optional[torch.dtype] = None) -> EvalResult:
    """Score ``dataset`` with ``model`` and compute :class:`EvalResult`.

    Args:
        model: an initialized :class:`~deepmatcher.MatchingModel`.
        dataset: a labeled dataset from :func:`deepmatcher.data.process`.
        device: ``auto|cuda|mps|cpu`` (see :func:`deepmatcher.runtime.select_device`).
        batch_size: evaluation batch size (no gradients, so larger than training).
        group_by: name of a column of the dataset's CSV (also an ``ignore_columns``
            one; the file is read with pandas and joined on the id column) to break
            the metrics down by, e.g. ``'country'``.
        group_bin: when set, ``group_by`` is numeric and is cut into bins of this
            width (``0.1`` for a similarity in [0, 1]).
        groups: alternatively, one group value per example in dataset order.
        autocast_dtype: evaluate under :func:`torch.autocast` with this dtype.
    """
    device = select_device(device) if not isinstance(device, torch.device) else device
    label_attr = model.meta.label_field
    id_attr = model.meta.id_field
    was_training = model.training
    model.to(device).eval()
    iterator = MatchingIterator(dataset, model.meta, False, batch_size=batch_size,
                                device=device.type, sort_in_buckets=False)
    probs, labels, ids = [], [], []
    start = time.time()
    with torch.no_grad():
        for batch in iterator:
            if autocast_dtype is not None:
                with torch.autocast(device.type, dtype=autocast_dtype):
                    out = model(batch)
            else:
                out = model(batch)
            probs.append(out[:, 1].float().exp().cpu())
            labels.append(getattr(batch, label_attr).cpu())
            ids.extend(getattr(batch, id_attr, []))
    seconds = time.time() - start
    model.train(was_training)

    probs = torch.cat(probs).numpy().astype(np.float64)
    labels = torch.cat(labels).numpy().astype(np.int64)
    summary = _summary(probs, labels)
    sweep = _sweep(probs, labels)

    group_values = None
    if groups is not None:
        group_values = np.asarray(list(groups))
        if len(group_values) != len(labels):
            raise ValueError('`groups` must have one value per example in dataset order')
    elif group_by is not None:
        table = pd.read_csv(dataset.path)
        if group_by not in table.columns:
            raise ValueError(f'column {group_by!r} not in {dataset.path}')
        id_col = dataset.id_field
        if id_col in table.columns and ids:
            lookup = table.drop_duplicates(id_col).set_index(table[id_col].astype(str))
            group_values = lookup.reindex([str(i) for i in ids])[group_by].to_numpy()
        else:  # no ids to join on: the iterator keeps CSV order
            group_values = table[group_by].to_numpy()[:len(labels)]

    group_frame = None
    if group_values is not None:
        if group_bin:
            numeric = pd.to_numeric(pd.Series(group_values), errors='coerce')
            lo = np.floor(numeric / group_bin + 1e-9) * group_bin
            keys = lo.map(lambda v: 'n/a' if pd.isna(v)
                          else f'[{v:.2f},{v + group_bin:.2f})').to_numpy()
            order = {k: (float(lo[i]) if not pd.isna(lo[i]) else float('inf'))
                     for i, k in enumerate(keys)}
        else:
            keys = np.asarray([str(v) for v in group_values])
            order = {k: k for k in keys}
        rows = []
        for key in sorted(set(keys), key=lambda k: order[k]):
            mask = keys == key
            row = _summary(probs[mask], labels[mask])
            row['group'] = key
            rows.append(row)
        group_frame = pd.DataFrame(rows).set_index('group')

    return EvalResult(
        probabilities=probs, labels=labels, ids=ids, precision=summary['precision'],
        recall=summary['recall'], f1=summary['f1'], tp=summary['tp'], fp=summary['fp'],
        fn=summary['fn'], tn=summary['tn'], roc_auc=summary['roc_auc'],
        pr_auc=summary['pr_auc'], sweep=sweep,
        best_f1_threshold=summary['best_f1_threshold'], best_f1=summary['best_f1'],
        groups=group_frame, seconds=seconds)


def agreement(probs_a, probs_b, threshold: float = 0.5) -> dict:
    """Compare two score vectors for the same pairs (e.g. a reproduced model against a
    reference): how often they make the same match decision and how far apart the
    scores are.

    Returns ``n``, ``agreement_rate`` (fraction with the same decision), ``flips``
    (decisions that differ, split into ``flips_to_match`` / ``flips_to_non_match`` from
    A's point of view), ``max_abs_diff`` and ``mean_abs_diff``.
    """
    a = np.asarray(probs_a, dtype=np.float64)
    b = np.asarray(probs_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f'shape mismatch: {a.shape} vs {b.shape}')
    decision_a, decision_b = a >= threshold, b >= threshold
    flips = decision_a != decision_b
    diff = np.abs(a - b)
    n = int(a.size)
    return {
        'n': n,
        'threshold': threshold,
        'agreement_rate': float(1 - flips.sum() / n) if n else 1.0,
        'flips': int(flips.sum()),
        'flips_to_match': int((~decision_a & decision_b).sum()),
        'flips_to_non_match': int((decision_a & ~decision_b).sum()),
        'max_abs_diff': float(diff.max()) if n else 0.0,
        'mean_abs_diff': float(diff.mean()) if n else 0.0,
    }
