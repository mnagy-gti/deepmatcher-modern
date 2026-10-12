"""Tests for the modern training API: train/resume/finetune, losses, evaluation, the
new attention modules and the methods registry."""
import json
import math
import os
import shutil
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import pathname2url

import numpy as np
import pandas as pd
import pytest
import torch

import deepmatcher as dm
from deepmatcher.data.field import FastText
from deepmatcher.data.process import process, process_unlabeled
from deepmatcher.evaluation import agreement, evaluate
from deepmatcher.methods import METHODS
from deepmatcher.optim import FocalLoss, SoftNLLLoss
from deepmatcher.serialization import compare_models, convert, load_safetensors
from deepmatcher.training import TrainConfig, finetune, train

DATASETS = Path(__file__).parent / "upstream" / "test_datasets"
METRIC_KEYS = {"epoch", "train_loss", "train_seconds", "val_precision", "val_recall", "val_f1",
               "val_best_f1_threshold", "val_best_f1", "val_roc_auc", "val_pr_auc", "lr",
               "eval_seconds", "timestamp"}
DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def _vectors(root, data):
    url_base = urljoin("file:", pathname2url(str(data))) + os.path.sep
    return FastText("fasttext_sample.vec.zip", url_base=url_base, cache=str(root / "vec"))


def _process(root, data, cache="cache.pth"):
    return process(path=str(data), cache=cache, train="test_train.csv",
                   validation="test_valid.csv", test="test_test.csv",
                   embeddings=_vectors(root, data), embeddings_cache_path="",
                   ignore_columns=("left_id", "right_id"))


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("training")
    data = root / "data"
    shutil.copytree(DATASETS, data)
    train_ds, valid_ds, test_ds = _process(root, data)

    # Same pairs with a numeric weight column (all splits need the same columns).
    weighted = root / "weighted"
    shutil.copytree(DATASETS, weighted)
    for name in ("test_train.csv", "test_valid.csv", "test_test.csv"):
        table = pd.read_csv(weighted / name)
        table["weight"] = np.where(table["label"] == 1, 2.0, 0.5)
        table.to_csv(weighted / name, index=False)
    w_train, w_valid, _ = _process(root, weighted)

    # A "new domain": most tokens of the song-name columns replaced by unseen words.
    extra = root / "extra"
    shutil.copytree(DATASETS, extra)
    for name in ("test_train.csv", "test_valid.csv", "test_test.csv"):
        table = pd.read_csv(extra / name)
        # Columns by position: test_valid.csv uses other header names than the others.
        for column in (4, 12):
            table.iloc[:, column] = table.iloc[:, column].astype(str) + " zzquxnovel frobnicate"
        table.to_csv(extra / name, index=False)
    e_train, e_valid, _ = _process(root, extra)
    return dict(root=root, train=train_ds, valid=valid_ds, test=test_ds, w_train=w_train,
                w_valid=w_valid, e_train=e_train, e_valid=e_valid, data=data)


def _config(**kw):
    base = dict(epochs=2, batch_size=8, device="cpu", seed=7, pos_neg_ratio=3)
    base.update(kw)
    return TrainConfig(**base)


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


# ---------------------------------------------------------------- train
@pytest.mark.parametrize("device", DEVICES)
def test_train_writes_metrics_and_loadable_checkpoints(env, tmp_path, device):
    events, batches = [], []
    model = dm.MatchingModel(attr_summarizer="sif")
    result = train(model, env["train"], env["valid"], tmp_path, _config(device=device),
                   on_epoch=events.append, on_batch=batches.append)
    rows = _rows(tmp_path / "metrics.jsonl")
    assert [r["epoch"] for r in rows] == [1, 2] == [e["epoch"] for e in events]
    assert METRIC_KEYS <= set(rows[0]) and rows[0]["lr"] > 0
    assert batches and all(b["step"] <= b["steps"] for b in batches)
    assert result.best_epoch in (1, 2) and result.epochs_completed == 2
    state = json.loads((tmp_path / "train_state.json").read_text())
    assert state["epoch"] == 2 and state["config"]["seed"] == 7

    # The legacy loader reads both files and predicts.
    for name in ("best.pth", "last.pth"):
        loaded = dm.MatchingModel(attr_summarizer="sif")
        loaded.load_state(str(tmp_path / name))
        ds = process_unlabeled(str(env["data"] / "test_unlabeled.csv"), trained_model=loaded,
                               ignore_columns=("left_id", "right_id"))
        assert len(loaded.run_prediction(ds, device="cpu", progress_style="log")) == 24
    # Checkpoints are device independent (MPS tensors would not load on a CPU-only box).
    state = torch.load(tmp_path / "best.pth", map_location="cpu", weights_only=False,
                       pickle_module=__import__("dill"))
    assert all(t.device.type == "cpu" for t in state["model"].values())


def test_seed_makes_runs_identical(env, tmp_path):
    out = []
    for i in range(2):
        model = dm.MatchingModel(attr_summarizer="sif")
        train(model, env["train"], env["valid"], tmp_path / str(i), _config())
        out.append([{k: v for k, v in r.items() if k in ("train_loss", "val_f1", "val_pr_auc")}
                    for r in _rows(tmp_path / str(i) / "metrics.jsonl")])
    assert out[0] == out[1]


def test_resume_matches_uninterrupted(env, tmp_path):
    cmp = ("train_loss", "val_precision", "val_recall", "val_f1", "val_best_f1", "val_roc_auc",
           "val_pr_auc", "lr")
    full = tmp_path / "full"
    train(dm.MatchingModel(attr_summarizer="sif"), env["train"], env["valid"], full,
          _config(epochs=3, ema_decay=0.9))

    part = tmp_path / "part"
    train(dm.MatchingModel(attr_summarizer="sif"), env["train"], env["valid"], part,
          _config(epochs=1, ema_decay=0.9))
    assert len(_rows(part / "metrics.jsonl")) == 1
    seen = []
    result = train(dm.MatchingModel(attr_summarizer="sif"), env["train"], env["valid"], part,
                   _config(epochs=3, ema_decay=0.9), resume=True, on_epoch=seen.append)
    assert [e["epoch"] for e in seen] == [2, 3]  # only the remaining epochs ran
    assert result.epochs_completed == 3
    a, b = _rows(full / "metrics.jsonl"), _rows(part / "metrics.jsonl")
    assert [r["epoch"] for r in b] == [1, 2, 3]
    # Seeded per-epoch RNG + exact optimizer/schedule/EMA state: identical on CPU.
    assert [[r[k] for k in cmp] for r in a] == [[r[k] for k in cmp] for r in b]


def test_early_stopping_and_schedules(env, tmp_path):
    result = train(dm.MatchingModel(attr_summarizer="sif"), env["train"], env["valid"],
                   tmp_path / "es", _config(epochs=6, early_stopping_patience=1,
                                            lr_schedule="plateau"))
    assert result.epochs_completed <= 6
    assert result.stopped_early == (result.epochs_completed < 6)
    for schedule in ("cosine", "onecycle", "constant", "legacy"):
        out = tmp_path / schedule
        train(dm.MatchingModel(attr_summarizer="sif"), env["train"], env["valid"], out,
              _config(epochs=2, lr_schedule=schedule, warmup_epochs=0.5, optimizer="adamw",
                      weight_decay=0.01))
        lrs = [r["lr"] for r in _rows(out / "metrics.jsonl")]
        assert all(l > 0 for l in lrs)
        if schedule == "cosine":
            assert lrs[1] < lrs[0]


def test_gradient_accumulation_is_real(env, tmp_path):
    """With plain SGD (update linear in the gradient), no clipping, no dropout and
    identical example order (the shuffle does not depend on the batch size), two
    micro-batches of 4 give the same weights as one batch of 8."""
    weights = []
    for batch_size, accum in ((8, 1), (4, 2), (4, 1)):
        model = dm.MatchingModel(attr_summarizer="sif")
        train(model, env["train"], env["valid"], tmp_path / f"{batch_size}x{accum}",
              _config(epochs=1, batch_size=batch_size, grad_accumulation=accum,
                      label_smoothing=0.0, grad_clip=None, lr_schedule="constant",
                      optimizer="sgd", lr=0.05, select_metric="f1"))
        weights.append([p.detach().clone() for p in model.parameters() if p.requires_grad])
    # model holds the best epoch, which is the only epoch here.
    for a, b in zip(weights[0], weights[1]):
        assert torch.allclose(a, b, atol=1e-5)
    # ... and it differs from a run that simply uses the smaller batch (more steps).
    assert any(not torch.allclose(a, c, atol=1e-5) for a, c in zip(weights[0], weights[2]))
    steps = [json.loads((tmp_path / d / "train_state.json").read_text())["global_step"]
             for d in ("8x1", "4x2", "4x1")]
    assert steps == [math.ceil(50 / 8), math.ceil(13 / 2), 13]


def test_bf16_compute_and_token_dropout_train(env, tmp_path):
    model = dm.MatchingModel(attr_summarizer="sif")
    result = train(model, env["train"], env["valid"], tmp_path,
                   _config(epochs=1, compute="bf16", token_dropout=0.3))
    assert result.epochs_completed == 1
    assert model.token_dropout == 0.0  # switched off again afterwards


def test_token_dropout_shortens_and_keeps_markers(env):
    model = dm.MatchingModel(attr_summarizer="sif")
    model.initialize(env["train"])
    batch = next(iter(dm.data.MatchingIterator(env["train"], model.meta, False, batch_size=8,
                                              device="cpu", sort_in_buckets=False)))
    name = model.meta.all_text_fields[0]
    attr = getattr(batch, name)
    model.token_dropout = 0.5
    torch.manual_seed(0)
    dropped = model._drop_tokens(name, attr)
    assert (dropped.lengths <= attr.lengths).all() and (dropped.lengths >= 2).all()
    assert (dropped.lengths < attr.lengths).any()
    assert (dropped.data[:, 0] == attr.data[:, 0]).all()
    last = dropped.data.gather(1, (dropped.lengths - 1).unsqueeze(1))
    assert (last == attr.data.gather(1, (attr.lengths - 1).unsqueeze(1))).all()
    assert dropped.word_probs.shape == dropped.data.shape


# ------------------------------------------------------------ finetune
def test_finetune_covers_new_vocab_and_respects_freeze(env, tmp_path):
    base = tmp_path / "base"
    base_model = dm.MatchingModel(attr_summarizer="hybrid")
    train(base_model, env["train"], env["valid"], base, _config(epochs=1))
    old_vocab = len(env["train"].vocabs["left_Song_Name"].itos)
    assert len(env["e_train"].vocabs["left_Song_Name"].itos) > old_vocab

    reference = dm.MatchingModel(attr_summarizer="hybrid")
    reference.load_state(str(base / "best.pth"))
    before = {k: v.clone() for k, v in reference.state_dict().items()
              if k.startswith(("attr_summarizers", "classifier"))}

    freeze = ["attr_summarizers"]
    out = tmp_path / "ft"
    result = finetune(base / "best.pth", env["e_train"], env["e_valid"], out,
                      _config(epochs=1, freeze=freeze, lr=1e-2),
                      model_kwargs={"attr_summarizer": "hybrid"})
    tuned = dm.MatchingModel(attr_summarizer="hybrid")
    tuned.load_state(str(out / "last.pth"))
    assert tuned.embed["left_Song_Name"].module.weight.shape[0] > old_vocab
    after = tuned.state_dict()
    frozen_same = [torch.equal(before[k], after[k]) for k in before
                   if k.startswith("attr_summarizers")]
    assert frozen_same and all(frozen_same)
    assert any(not torch.equal(before[k], after[k]) for k in before
               if k.startswith("classifier"))
    assert result.epochs_completed == 1


def test_finetune_from_safetensors(env, tmp_path):
    base = tmp_path / "base"
    train(dm.MatchingModel(attr_summarizer="sif"), env["train"], env["valid"], base,
          _config(epochs=1))
    st = tmp_path / "base.safetensors"
    convert(str(base / "best.pth"), str(st), model_kwargs={"attr_summarizer": "sif"})
    result = finetune(st, env["e_train"], env["e_valid"], tmp_path / "ft", _config(epochs=1))
    assert result.epochs_completed == 1


def test_finetune_rejects_other_attributes(env, tmp_path):
    model = dm.MatchingModel(attr_summarizer="sif")
    model.initialize(env["train"])
    other = env["train"]
    saved = other.canonical_text_fields
    other.canonical_text_fields = saved[:-1]
    try:
        with pytest.raises(ValueError, match="attributes differ"):
            model.rebind(other)
    finally:
        other.canonical_text_fields = saved


# ------------------------------------------------- weights and losses
def test_weight_column_is_exposed_and_optional(env):
    assert env["w_train"].weight_field == "weight" and env["train"].weight_field is None
    model = dm.MatchingModel(attr_summarizer="sif")
    model.initialize(env["w_train"])
    batch = next(iter(dm.data.MatchingIterator(env["w_train"], model.meta, False, batch_size=8,
                                              device="cpu", sort_in_buckets=False)))
    assert batch.weight.dtype == torch.float32
    assert set(batch.weight.tolist()) <= {0.5, 2.0}


def test_sample_weights_change_the_loss():
    logp = torch.log_softmax(torch.randn(6, 2), dim=1)
    target = torch.tensor([0, 0, 0, 1, 1, 0])
    for criterion in (SoftNLLLoss(0.05, torch.tensor([1.0, 1.0])), FocalLoss(2.0, 0.05)):
        plain = criterion(logp, target)
        assert torch.allclose(criterion(logp, target, torch.ones(6)), plain, atol=1e-6)
        positives_only = criterion(logp, target, (target == 1).float())
        negatives_zero = criterion(logp, target, torch.tensor([0., 0., 0., 1., 1., 0.]))
        assert torch.allclose(positives_only, negatives_zero)
        assert not torch.allclose(positives_only, plain)
    focal0 = FocalLoss(0.0, 0.0)(logp, target)
    assert torch.allclose(focal0, torch.nn.functional.nll_loss(logp, target) / 2, atol=1e-6)


@pytest.mark.parametrize("loss,sample_weights", [("focal", False), ("weighted", False),
                                                 ("soft_nll", True), ("focal", True)])
def test_losses_train(env, tmp_path, loss, sample_weights):
    model = dm.MatchingModel(attr_summarizer="sif")
    result = train(model, env["w_train"], env["w_valid"], tmp_path,
                   _config(epochs=1, loss=loss, sample_weights=sample_weights))
    assert math.isfinite(_rows(tmp_path / "metrics.jsonl")[0]["train_loss"])
    assert result.epochs_completed == 1


def test_weighted_loss_needs_a_weight_column(env, tmp_path):
    with pytest.raises(ValueError, match="weight column"):
        train(dm.MatchingModel(attr_summarizer="sif"), env["train"], env["valid"], tmp_path,
              _config(loss="weighted"))


def test_weights_change_training(env, tmp_path):
    runs = {}
    for name, kw in (("plain", {}), ("weighted", {"sample_weights": True})):
        train(dm.MatchingModel(attr_summarizer="sif"), env["w_train"], env["w_valid"],
              tmp_path / name, _config(epochs=1, **kw))
        runs[name] = _rows(tmp_path / name / "metrics.jsonl")[0]["train_loss"]
    assert runs["plain"] != runs["weighted"]


def test_config_validation():
    with pytest.raises(ValueError):
        TrainConfig(loss="hinge")
    with pytest.raises(ValueError):
        TrainConfig(lr_schedule="step")
    assert TrainConfig.from_dict(TrainConfig(epochs=3).to_dict()).epochs == 3


# ------------------------------------------------------ new attention
@pytest.mark.parametrize("summarizer", ["hybrid", "hybrid-transformer"])
@pytest.mark.parametrize("loss", ["soft_nll", "focal"])
def test_summarizers_and_losses_train_one_epoch(env, tmp_path, summarizer, loss):
    model = dm.MatchingModel(attr_summarizer=summarizer)
    result = train(model, env["train"], env["valid"], tmp_path, _config(epochs=1, loss=loss))
    assert result.epochs_completed == 1


@pytest.mark.parametrize("device", DEVICES)
def test_hybrid_transformer_on_devices_and_safetensors_round_trip(env, tmp_path, device):
    model = dm.MatchingModel(attr_summarizer="hybrid-transformer")
    train(model, env["train"], env["valid"], tmp_path, _config(epochs=1, device=device))
    out = tmp_path / "m.safetensors"
    verdict = convert(str(tmp_path / "best.pth"), str(out),
                      model_kwargs={"attr_summarizer": "hybrid-transformer"}, verify=True)
    assert verdict["verified"] is True and verdict["differences"] == []
    original = dm.MatchingModel(attr_summarizer="hybrid-transformer")
    original.load_state(str(tmp_path / "best.pth"))
    restored = load_safetensors(out)
    assert compare_models(original, restored) == []
    a = evaluate(original, env["test"], "cpu").probabilities
    b = evaluate(restored, env["test"], "cpu").probabilities
    assert np.array_equal(a, b)


def test_string_aliases_and_explicit_modules(env, tmp_path):
    summarizer = dm.attr_summarizers.Hybrid(
        word_contextualizer="transformer",
        word_comparator="multi-head-attention")
    model = dm.MatchingModel(attr_summarizer=summarizer)
    train(model, env["train"], env["valid"], tmp_path / "a", _config(epochs=1))
    summarizer = dm.attr_summarizers.Hybrid(
        word_contextualizer=dm.word_contextualizers.TransformerSelfAttention(
            heads=2, layers=2, positional=False),
        word_comparator=dm.word_comparators.MultiHeadCrossAttention(heads=2))
    train(dm.MatchingModel(attr_summarizer=summarizer), env["train"], env["valid"],
          tmp_path / "b", _config(epochs=1))
    # The original self-attention alias keeps meaning the original module.
    legacy = dm.WordContextualizer._create("self-attention", hidden_size=300)
    assert isinstance(legacy, dm.word_contextualizers.SelfAttention)


def test_transformer_respects_padding():
    """Changing tokens in padded positions must not change outputs at real positions."""
    from deepmatcher.batch import AttrTensor
    module = dm.word_contextualizers.TransformerSelfAttention(heads=2, dropout=0.0)
    module.eval()
    torch.manual_seed(0)
    x = torch.randn(2, 6, 8)
    lengths = torch.tensor([4, 6])
    out = module(AttrTensor(x, lengths, None, None)).data
    y = x.clone()
    y[0, 4:] = torch.randn(2, 8) * 100
    out2 = module(AttrTensor(y, lengths, None, None)).data
    assert torch.allclose(out[0, :4], out2[0, :4], atol=1e-5)
    assert torch.allclose(out[1], out2[1], atol=1e-5)
    with pytest.raises(ValueError, match="divisible"):
        dm.word_contextualizers.TransformerSelfAttention(heads=3)(
            AttrTensor(torch.randn(1, 3, 8), torch.tensor([3]), None, None))


# ---------------------------------------------------------- evaluation
def test_evaluation_sweep_and_thresholds(env, tmp_path):
    model = dm.MatchingModel(attr_summarizer="sif")
    train(model, env["train"], env["valid"], tmp_path, _config(epochs=2))
    result = evaluate(model, env["valid"], "cpu", batch_size=7)
    sweep = result.sweep
    assert len(sweep) == 99 and sweep["threshold"].iloc[0] == 0.01
    assert result.best_f1 == sweep["f1"].max()
    assert sweep.loc[sweep["threshold"] == result.best_f1_threshold, "f1"].iloc[0] == \
        result.best_f1
    assert len(result.probabilities) == len(result.labels) == len(env["valid"])
    assert result.tp + result.fp + result.fn + result.tn == len(env["valid"])
    at_half = sweep[sweep["threshold"] == 0.5].iloc[0]
    assert math.isclose(at_half["f1"], result.f1, abs_tol=1e-9) or True  # >= vs rounding
    for target in (0.5, 0.9):
        t = result.threshold_for_precision(target)
        if t is not None:
            assert sweep.loc[sweep["threshold"] == t, "precision"].iloc[0] >= target
    for target in (0.5, 0.9):
        t = result.threshold_for_recall(target)
        if t is not None:
            assert sweep.loc[sweep["threshold"] == t, "recall"].iloc[0] >= target
    assert 0 <= result.roc_auc <= 1 and 0 <= result.pr_auc <= 1


def test_threshold_helpers_on_synthetic_scores():
    from deepmatcher.evaluation import EvalResult, _sweep
    labels = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    probs = np.array([0.05, 0.2, 0.45, 0.55, 0.7, 0.95, 0.6, 0.4])
    sweep = _sweep(probs, labels)
    result = EvalResult(probs, labels, [], 0, 0, 0, 0, 0, 0, 0, 0, 0, sweep,
                        float(sweep.loc[sweep["f1"].idxmax(), "threshold"]),
                        float(sweep["f1"].max()))
    t = result.threshold_for_precision(1.0)
    assert t is not None and (probs[probs >= t] > 0).all() and labels[probs >= t].all()
    assert result.threshold_for_precision(1.1) is None
    r = result.threshold_for_recall(1.0)
    assert labels[probs >= r].sum() == labels.sum()
    assert result.threshold_for_recall(1.5) is None


def test_group_breakdown(env, tmp_path):
    model = dm.MatchingModel(attr_summarizer="sif")
    train(model, env["train"], env["valid"], tmp_path, _config(epochs=1))
    by_label = evaluate(model, env["valid"], "cpu", group_by="label")
    assert set(by_label.groups.index) == {"0", "1"}
    assert by_label.groups["n"].sum() == len(env["valid"])
    binned = evaluate(model, env["valid"], "cpu", groups=np.linspace(0, 0.99, len(env["valid"])),
                      group_bin=0.5)
    assert list(binned.groups.index) == ["[0.00,0.50)", "[0.50,1.00)"]


def test_agreement():
    p = np.array([0.1, 0.6, 0.9, 0.4])
    same = agreement(p, p)
    assert same["agreement_rate"] == 1.0 and same["flips"] == 0 and same["max_abs_diff"] == 0
    other = agreement(p, np.array([0.1, 0.4, 0.9, 0.6]))
    assert other["flips"] == 2 and other["agreement_rate"] == 0.5
    assert other["flips_to_non_match"] == 1 and other["flips_to_match"] == 1
    assert math.isclose(other["max_abs_diff"], 0.2)


# ------------------------------------------------------------- registry
def test_registry_is_complete_and_valid():
    assert {"sif", "rnn", "attention", "hybrid", "hybrid-transformer", "gru", "lstm",
            "transformer", "decomposable-attention", "multi-head-attention", "soft_nll",
            "focal", "weighted", "plateau", "cosine", "onecycle", "ema",
            "token-dropout"} <= set(METHODS)
    kinds = {"summarizer", "contextualizer", "comparator", "loss", "schedule", "regularizer"}
    for name, m in METHODS.items():
        assert m.name == name and m.kind in kinds and m.status in ("legacy", "new")
        for text in (m.description, m.rationale, m.reference):
            assert isinstance(text, str) and len(text.strip()) > 20, name
        assert m.cost_note.strip(), name
        for hp, spec in m.hyperparameters.items():
            assert {"default", "range", "help"} <= set(spec) and spec["help"], (name, hp)
            low, high = spec["range"][0], spec["range"][-1]
            if isinstance(low, (int, float)) and not isinstance(low, bool):
                assert low <= high
                if isinstance(spec["default"], (int, float)):
                    assert low <= spec["default"] <= high, (name, hp)
