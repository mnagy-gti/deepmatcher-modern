r"""Modern training for DeepMatcher models: :func:`train`, :func:`finetune`.

``MatchingModel.run_train`` (the original API, kept unchanged) trains in a progress-bar
loop and keeps only the best checkpoint. This module adds what an orchestration layer
needs: a typed config, reproducible runs, resume, per-epoch metrics on disk, UI
callbacks, learning-rate schedules, real gradient accumulation, mixed precision, EMA
weights, per-example loss weights and fine-tuning. Checkpoints are written in the
original ``save_state`` format, so every existing loader keeps working.

Example::

    import deepmatcher as dm
    from deepmatcher.training import TrainConfig, train

    train_ds, valid_ds, _ = dm.data.process(path, train='train.csv', validation='valid.csv',
                                            test='test.csv', ...)
    model = dm.MatchingModel(attr_summarizer='hybrid')
    result = train(model, train_ds, valid_ds, 'runs/exp1',
                   TrainConfig(epochs=10, device='auto', seed=42),
                   on_epoch=lambda m: print(m['epoch'], m['val_f1']))
    # runs/exp1/{best.pth,last.pth,metrics.jsonl,train_state.json}

Output files (``out_dir``):

``best.pth``
    Weights of the epoch with the best validation score (``config.select_metric``).
    With EMA enabled these are the EMA weights.
``last.pth``
    State after the most recent epoch, including the optimizer: what ``resume=True``
    continues from. Always the raw (non-EMA) weights.
``ema.pt``
    The EMA shadow weights (only with ``ema_decay``), needed to resume.
``metrics.jsonl``
    One JSON object per epoch: ``epoch`` (1-based), ``train_loss``, ``train_seconds``,
    ``val_precision``, ``val_recall``, ``val_f1`` (at threshold 0.5),
    ``val_best_f1_threshold``, ``val_best_f1``, ``val_roc_auc``, ``val_pr_auc``, ``lr``
    (at the end of the epoch), ``eval_seconds``, ``timestamp`` (ISO 8601, UTC) and
    ``is_best``.
``train_state.json``
    Config, completed epochs, best score, schedule state and seeds.

Reproducibility: with ``seed`` set, the model initialization is seeded, and each epoch
re-seeds torch / numpy / random and the example shuffling from ``seed`` and the epoch
number. That makes an epoch a pure function of (weights, optimizer, schedule, seed,
epoch), which is exactly what :func:`train` ``resume`` restores, so an interrupted run
continued with ``resume=True`` gives the same metrics as an uninterrupted one (bit for
bit on CPU; on MPS/CUDA up to nondeterministic kernels).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import dill
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .data import MatchingIterator
from .evaluation import evaluate
from .optim import FocalLoss, SoftNLLLoss
from .runtime import select_device

logger = logging.getLogger('deepmatcher.training')

OPTIMIZERS = ('adam', 'adamw', 'sgd')
LR_SCHEDULES = ('plateau', 'cosine', 'onecycle', 'constant', 'legacy')
LOSSES = ('soft_nll', 'focal', 'weighted')
COMPUTE = ('fp32', 'bf16', 'fp16')
SELECT_METRICS = ('f1', 'best_f1', 'pr_auc', 'roc_auc')
_AUTOCAST = {'bf16': torch.bfloat16, 'fp16': torch.float16}


@dataclass
class TrainConfig:
    """Everything that defines a training run.

    Args:
        epochs: total epochs (a resumed run continues up to this number).
        batch_size: examples per forward pass.
        grad_accumulation: micro-batches per optimizer step. The effective batch is
            ``batch_size * grad_accumulation``; the loss of each micro-batch is
            scaled so the update equals that of one large batch. (The original
            ``run_train`` accepted this argument through ``**kwargs`` and silently
            ignored it; here it is real.)
        device: ``auto|cuda|mps|cpu`` (see :func:`deepmatcher.runtime.select_device`).
        compute: ``fp32`` | ``bf16`` | ``fp16`` autocast for forward/backward. fp16 on
            CUDA uses a gradient scaler. The loss and optimizer stay in fp32.
        seed: seeds everything (see the module docstring); None = unseeded.
        lr, optimizer, weight_decay: ``adam`` (the original), ``adamw`` (decoupled
            decay) or ``sgd`` (momentum 0.9).
        lr_schedule: ``plateau`` multiplies the rate by ``plateau_decay`` (0.8) after
            every epoch whose validation score did not beat the best so far; ``cosine``
            anneals to 0 over the whole run; ``onecycle`` warms up from lr/25 over the
            first 30 % of steps, then anneals to lr/1e4 (``warmup_epochs`` is
            ignored); ``constant``; ``legacy`` reproduces what ``run_train`` really
            does, a 0.8 decay after *every* epoch.
        warmup_epochs: linear warm-up from 0 (may be fractional; not used by
            ``onecycle``).
        label_smoothing, pos_neg_ratio: as in ``run_train`` (the match class is
            weighted ``pos_neg_ratio`` times the non-match class).
        loss: ``soft_nll`` (the original), ``focal`` (``focal_gamma``) or ``weighted``
            (``soft_nll`` with the dataset's per-example weights; needs a weight
            column).
        sample_weights: also use the per-example weights with ``soft_nll`` / ``focal``.
        grad_clip: maximum gradient norm; 0 or None disables clipping.
        early_stopping_patience: stop after this many epochs without a new best
            validation score; None runs all epochs.
        ema_decay: keep an exponential moving average of the trainable weights and use
            it for validation and ``best.pth``.
        checkpoint_every: write ``last.pth`` every N epochs (always at the end).
        eval_batch_size: validation batch size.
        freeze: parameter-name prefixes that are not trained (``'attr_summarizers'``,
            ``'classifier'``, ...); used when fine-tuning. Pretrained embeddings are
            always frozen.
        token_dropout: probability of deleting each input token during training.
        select_metric: validation score that picks ``best.pth`` and drives plateau
            decay and early stopping: ``f1`` (at 0.5, like ``run_train``), ``best_f1``
            (at the best threshold), ``pr_auc`` or ``roc_auc``.
        plateau_decay: decay factor of the ``plateau`` schedule.
    """
    epochs: int = 10
    batch_size: int = 32
    grad_accumulation: int = 1
    device: str = 'auto'
    compute: str = 'fp32'
    seed: Optional[int] = 42
    lr: float = 1e-3
    optimizer: str = 'adam'
    weight_decay: float = 0.0
    lr_schedule: str = 'plateau'
    warmup_epochs: float = 0.0
    label_smoothing: float = 0.05
    pos_neg_ratio: float = 1.0
    loss: str = 'soft_nll'
    focal_gamma: float = 2.0
    sample_weights: bool = False
    grad_clip: Optional[float] = 5.0
    early_stopping_patience: Optional[int] = None
    ema_decay: Optional[float] = None
    checkpoint_every: int = 1
    eval_batch_size: int = 512
    freeze: list = field(default_factory=list)
    token_dropout: float = 0.0
    select_metric: str = 'f1'
    plateau_decay: float = 0.8

    def __post_init__(self):
        self.freeze = list(self.freeze or [])
        for name, value, allowed in (
                ('optimizer', self.optimizer, OPTIMIZERS),
                ('lr_schedule', self.lr_schedule, LR_SCHEDULES),
                ('loss', self.loss, LOSSES), ('compute', self.compute, COMPUTE),
                ('select_metric', self.select_metric, SELECT_METRICS)):
            if value not in allowed:
                raise ValueError(f'{name}={value!r}; expected one of {allowed}')
        if self.epochs < 1 or self.batch_size < 1 or self.grad_accumulation < 1:
            raise ValueError('epochs, batch_size and grad_accumulation must be >= 1')
        if self.ema_decay is not None and not 0 < self.ema_decay < 1:
            raise ValueError('ema_decay must be in (0, 1)')
        if not 0 <= self.token_dropout < 1:
            raise ValueError('token_dropout must be in [0, 1)')
        if self.pos_neg_ratio <= 0:
            raise ValueError('pos_neg_ratio must be > 0')

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values):
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in known})


@dataclass
class TrainResult:
    """Outcome of :func:`train`. ``metrics`` has one dict per epoch of *this* run
    plus, when resumed, the epochs already in ``metrics.jsonl``."""
    best_score: float
    best_epoch: int
    epochs_completed: int
    stopped_early: bool
    metrics: list
    out_dir: str
    best_path: str
    last_path: str
    seconds: float


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def seed_everything(seed):
    """Seed python, numpy and torch (all devices)."""
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)


def _epoch_seed(seed, epoch):
    return (seed * 1_000_003 + epoch) % 2 ** 31


def _cpu_copy(obj, memo):
    """Deep-copy tensors in a nested dict/list structure to CPU, keeping tensors that
    shared storage (left/right embeddings) shared, so files do not grow."""
    if isinstance(obj, torch.Tensor):
        if obj.device.type == 'cpu':
            return obj
        key = (obj.device, obj.untyped_storage().data_ptr(), obj.storage_offset(),
               tuple(obj.shape), obj.stride(), obj.dtype)
        if key not in memo:
            memo[key] = obj.detach().cpu()
        return memo[key]
    if isinstance(obj, dict):
        return type(obj)((k, _cpu_copy(v, memo)) for k, v in obj.items())
    if isinstance(obj, list):
        return [_cpu_copy(v, memo) for v in obj]
    if isinstance(obj, tuple) and not hasattr(obj, '_fields'):
        return tuple(_cpu_copy(v, memo) for v in obj)
    return obj


def save_checkpoint(model, path):
    """``MatchingModel.save_state`` format, with tensors moved to CPU first so the
    file loads anywhere (device tags of MPS/CUDA tensors would not)."""
    memo = {}
    state = {'model': _cpu_copy(model.state_dict(), memo)}
    for k in model._train_buffers:
        state[k] = _cpu_copy(getattr(model, k), memo)
    tmp = str(path) + '.tmp'
    torch.save(state, tmp, pickle_module=dill)
    os.replace(tmp, path)


def _write_json(path, value):
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf8') as f:
        json.dump(value, f, indent=2, default=str)
    os.replace(tmp, path)


def _finite(value):
    return None if isinstance(value, float) and not math.isfinite(value) else value


class LRSchedule:
    """Learning rate as a function of (optimizer step, epoch) plus a little state
    (the plateau multiplier). Kept by hand rather than with ``torch.optim.lr_scheduler``
    so that its whole state is a few JSON numbers and resume is exact."""

    def __init__(self, config, steps_per_epoch):
        self.kind = config.lr_schedule
        self.base = config.lr
        self.total = max(1, steps_per_epoch * config.epochs)
        self.warmup = int(round(config.warmup_epochs * steps_per_epoch))
        self.decay = config.plateau_decay
        self.scale = 1.0
        self.best = -math.inf

    def lr_at(self, step, epoch):
        warm = min(1.0, (step + 1) / self.warmup) if self.warmup > 0 else 1.0
        progress = min(1.0, max(0.0, (step - self.warmup) / max(1, self.total - self.warmup)))
        if self.kind == 'cosine':
            return self.base * warm * 0.5 * (1 + math.cos(math.pi * progress))
        if self.kind == 'onecycle':
            pct = step / self.total
            start, end = self.base / 25, self.base / 1e4
            if pct < 0.3:
                return start + (self.base - start) * 0.5 * (1 - math.cos(math.pi * pct / 0.3))
            return end + (self.base - end) * 0.5 * (1 + math.cos(math.pi * (pct - 0.3) / 0.7))
        if self.kind == 'legacy':
            return self.base * warm * self.decay ** epoch
        if self.kind == 'plateau':
            return self.base * warm * self.scale
        return self.base * warm  # constant

    def epoch_end(self, score):
        if score > self.best:
            self.best = score
        elif self.kind == 'plateau':
            self.scale *= self.decay

    def state(self):
        return {'scale': self.scale, 'best': _finite(self.best)}

    def load(self, state):
        self.scale = state['scale']
        self.best = -math.inf if state['best'] is None else state['best']


class EMA:
    """Exponential moving average of the trainable parameters."""

    def __init__(self, model, decay):
        self.decay = decay
        self.updates = 0
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()
                       if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        # Warm-up of the decay so early, still bad weights do not dominate the average.
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(decay).add_(p.detach(), alpha=1 - decay)

    @contextlib.contextmanager
    def applied(self, model):
        """Temporarily load the averaged weights into ``model``."""
        backup = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow:
                    backup[n] = p.detach().clone()
                    p.copy_(self.shadow[n])
        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in backup:
                        p.copy_(backup[n])

    def state(self):
        return {'updates': self.updates,
                'shadow': {n: t.detach().cpu() for n, t in self.shadow.items()}}

    def load(self, state, device):
        self.updates = state['updates']
        self.shadow = {n: t.to(device) for n, t in state['shadow'].items()}


def _make_criterion(config, device):
    neg = 2 - 2 * config.pos_neg_ratio / (1 + config.pos_neg_ratio)
    pos = 2 * config.pos_neg_ratio / (1 + config.pos_neg_ratio)
    weight = torch.tensor([neg, pos], dtype=torch.float32)
    if config.loss == 'focal':
        criterion = FocalLoss(config.focal_gamma, config.label_smoothing, weight)
    else:
        criterion = SoftNLLLoss(config.label_smoothing, weight)
    return criterion.to(device)


def _make_optimizer(config, params):
    if config.optimizer == 'sgd':
        return torch.optim.SGD(params, lr=config.lr, momentum=0.9,
                               weight_decay=config.weight_decay)
    if config.optimizer == 'adamw':
        return torch.optim.AdamW(params, lr=config.lr, eps=1e-9,
                                 weight_decay=config.weight_decay)
    # eps=1e-9 as in the original deepmatcher optimizer.
    return torch.optim.Adam(params, lr=config.lr, eps=1e-9, weight_decay=config.weight_decay)


def _apply_freeze(model, prefixes):
    frozen = 0
    for name, p in model.named_parameters():
        if any(name.startswith(pre) for pre in prefixes):
            p.requires_grad = False
            frozen += p.numel()
    return frozen


def _score(metrics_row, select_metric):
    key = {'f1': 'val_f1', 'best_f1': 'val_best_f1', 'pr_auc': 'val_pr_auc',
           'roc_auc': 'val_roc_auc'}[select_metric]
    value = metrics_row[key]
    return -math.inf if value is None else value


# ----------------------------------------------------------------------------
# train
# ----------------------------------------------------------------------------
def train(model,
          train_ds,
          valid_ds,
          out_dir,
          config: Optional[TrainConfig] = None,
          on_epoch: Optional[Callable[[dict], None]] = None,
          on_batch: Optional[Callable[[dict], None]] = None,
          resume: bool = False,
          log: Optional[Callable[[str], None]] = None) -> TrainResult:
    """Train ``model`` and write checkpoints/metrics to ``out_dir``.

    Args:
        model: a :class:`~deepmatcher.MatchingModel`. Uninitialized models are
            initialized from ``train_ds``. A model that is already trained (or loaded)
            is trained further from its current weights, with a *fresh* optimizer and
            epoch counter, unless ``resume`` is set. To adapt a checkpoint to a new
            dataset use :func:`finetune`.
        train_ds, valid_ds: datasets from :func:`deepmatcher.data.process`. When
            ``config.loss == 'weighted'`` or ``config.sample_weights`` the training CSV
            needs a ``weight`` column (see ``process(weight_attr=...)``).
        out_dir: output directory (created).
        config: :class:`TrainConfig`; defaults if omitted. On ``resume`` the epoch
            count may be raised, other settings should not change.
        on_epoch: called with the metrics dict (the ``metrics.jsonl`` row) after each
            epoch.
        on_batch: called with ``{'epoch', 'step', 'loss', 'lr'}`` roughly every 20
            micro-batches and at the end of each epoch (``epoch`` is 1-based,
            ``loss`` the running mean of the epoch). Never called per batch.
        resume: continue from ``out_dir/last.pth`` and ``train_state.json`` (epoch,
            optimizer, schedule, EMA, best score). Without a ``last.pth`` it simply
            starts fresh.
        log: callable receiving one-line progress messages (default: the
            ``deepmatcher.training`` logger).

    Returns:
        :class:`TrainResult`. When it returns, ``model`` holds the best weights.
    """
    # The token-dropout regularizer is a model attribute; never leave it switched on
    # for whoever uses the model next.
    try:
        return _train(model, train_ds, valid_ds, out_dir, config, on_epoch, on_batch,
                      resume, log)
    finally:
        model.token_dropout = 0.0


def _train(model,
          train_ds,
          valid_ds,
          out_dir,
          config: Optional[TrainConfig] = None,
          on_epoch: Optional[Callable[[dict], None]] = None,
          on_batch: Optional[Callable[[dict], None]] = None,
          resume: bool = False,
          log: Optional[Callable[[str], None]] = None) -> TrainResult:
    config = config or TrainConfig()
    log = log or logger.info
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_path, last_path = out / 'best.pth', out / 'last.pth'
    ema_path, state_path, metrics_path = out / 'ema.pt', out / 'train_state.json', \
        out / 'metrics.jsonl'
    device = select_device(config.device)
    started = time.time()

    resuming = resume and last_path.exists() and state_path.exists()
    state = json.loads(state_path.read_text()) if resuming else {}

    # --- model ------------------------------------------------------------------
    if config.seed is not None:
        seed_everything(config.seed)  # makes the weight initialization reproducible
    if resuming:
        model.load_state(str(last_path), map_location='cpu')
    else:
        model.initialize(train_ds)
        for name in ('optimizer_state', 'best_score', 'epoch'):
            model._register_train_buffer(name, None)
    for name in ('optimizer_state', 'best_score', 'epoch'):
        model._register_train_buffer(name, getattr(model, name, None))
    model.token_dropout = config.token_dropout

    weight_attr = getattr(model.meta, 'weight_field', None)
    use_weights = config.loss == 'weighted' or config.sample_weights
    if use_weights and weight_attr is None:
        raise ValueError('per-example weights requested but the training data has no '
                         'weight column (process(..., weight_attr="weight"))')

    frozen = _apply_freeze(model, config.freeze)
    if frozen:
        log(f'froze {frozen:,} parameters ({config.freeze})')
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = _make_optimizer(config, params)
    criterion = _make_criterion(config, device)
    ema = EMA(model, config.ema_decay) if config.ema_decay else None

    iterator = MatchingIterator(train_ds, model.meta, True, batch_size=config.batch_size,
                                device=device.type, sort_in_buckets=False)
    n_batches = len(iterator)
    steps_per_epoch = math.ceil(n_batches / config.grad_accumulation)
    schedule = LRSchedule(config, steps_per_epoch)

    start_epoch, best_score, best_epoch, bad_epochs, global_step = 0, -math.inf, 0, 0, 0
    history = []
    if resuming:
        start_epoch = int(model.epoch) + 1
        best_score = state.get('best_score', -math.inf)
        best_score = -math.inf if best_score is None else best_score
        best_epoch = state.get('best_epoch', 0)
        bad_epochs = state.get('bad_epochs', 0)
        global_step = state.get('global_step', start_epoch * steps_per_epoch)
        schedule.load(state['schedule'])
        if model.optimizer_state is not None:
            optimizer.load_state_dict(model.optimizer_state)
        if ema is not None:
            if ema_path.exists():
                ema.load(torch.load(ema_path, map_location='cpu'), device)
            else:
                log('ema.pt missing; EMA restarts from the current weights')
        if metrics_path.exists():
            history = [json.loads(line) for line in metrics_path.read_text().splitlines()
                       if line.strip()]
            history = [row for row in history if row['epoch'] <= start_epoch]
        log(f'resuming after epoch {start_epoch}/{config.epochs}')
    # Rewrite metrics.jsonl: a fresh run starts it empty; a resumed one drops rows of
    # epochs that were not checkpointed.
    metrics_path.write_text(''.join(json.dumps(row) + '\n' for row in history))

    scaler = None
    if config.compute == 'fp16' and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
    autocast_dtype = _AUTOCAST.get(config.compute)

    label_attr = model.meta.label_field
    stopped_early = False
    log(f'training on {device} (compute={config.compute}), {len(train_ds)} examples, '
        f'{n_batches} batches/epoch, {sum(p.numel() for p in params):,} trainable parameters')

    for epoch in range(start_epoch, config.epochs):
        if config.seed is not None:
            seed = _epoch_seed(config.seed, epoch)
            seed_everything(seed)
            iterator.random_shuffler.random_state = random.Random(seed).getstate()
        model.train()
        epoch_start = time.time()
        loss_sum = torch.zeros((), device=device)
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        lr = schedule.lr_at(global_step, epoch)
        for index, batch in enumerate(iterator):
            group_start = index - index % config.grad_accumulation
            group_size = min(config.grad_accumulation, n_batches - group_start)
            labels = getattr(batch, label_attr)
            sample_weight = getattr(batch, weight_attr) if use_weights else None
            if autocast_dtype is not None:
                with torch.autocast(device.type, dtype=autocast_dtype):
                    output = model(batch)
            else:
                output = model(batch)
            loss = criterion(output.float(), labels, sample_weight)
            to_backprop = loss / group_size
            (scaler.scale(to_backprop) if scaler else to_backprop).backward()
            loss_sum += loss.detach() * labels.numel()
            seen += labels.numel()

            if (index + 1) % config.grad_accumulation == 0 or index + 1 == n_batches:
                lr = schedule.lr_at(global_step, epoch)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                if scaler:
                    scaler.unscale_(optimizer)
                if config.grad_clip:
                    clip_grad_norm_(params, config.grad_clip)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                global_step += 1

            if on_batch and ((index + 1) % 20 == 0 or index + 1 == n_batches):
                on_batch({'epoch': epoch + 1, 'step': index + 1, 'steps': n_batches,
                          'loss': float(loss_sum) / seen, 'lr': lr})
        train_loss = float(loss_sum) / max(seen, 1)
        train_seconds = time.time() - epoch_start

        eval_start = time.time()
        with (ema.applied(model) if ema is not None else contextlib.nullcontext()):
            result = evaluate(model, valid_ds, device, config.eval_batch_size,
                              autocast_dtype=autocast_dtype)
            row = {
                'epoch': epoch + 1, 'train_loss': _finite(train_loss),
                'train_seconds': train_seconds,
                'val_precision': result.precision, 'val_recall': result.recall,
                'val_f1': result.f1, 'val_best_f1_threshold': result.best_f1_threshold,
                'val_best_f1': result.best_f1, 'val_roc_auc': _finite(result.roc_auc),
                'val_pr_auc': _finite(result.pr_auc), 'lr': lr,
                'eval_seconds': time.time() - eval_start,
                'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            }
            score = _score(row, config.select_metric)
            row['is_best'] = bool(score > best_score)
            model.epoch = epoch
            if row['is_best']:
                best_score, best_epoch, bad_epochs = score, epoch + 1, 0
                model.best_score = best_score
                model.optimizer_state = None  # a best file is for inference
                save_checkpoint(model, best_path)
            else:
                bad_epochs += 1
        schedule.epoch_end(score)
        history.append(row)
        with open(metrics_path, 'a', encoding='utf8') as f:
            f.write(json.dumps(row) + '\n')

        done = epoch + 1 == config.epochs
        stop = (config.early_stopping_patience is not None
                and bad_epochs >= config.early_stopping_patience)
        if (epoch + 1) % config.checkpoint_every == 0 or done or stop:
            model.best_score = best_score if best_score > -math.inf else None
            model.optimizer_state = _cpu_copy(optimizer.state_dict(), {})
            save_checkpoint(model, last_path)
            if ema is not None:
                torch.save(ema.state(), ema_path)
        _write_json(state_path, {
            'config': config.to_dict(), 'epoch': epoch + 1, 'best_score': _finite(best_score),
            'best_epoch': best_epoch, 'bad_epochs': bad_epochs, 'global_step': global_step,
            'schedule': schedule.state(),
            'seeds': {'seed': config.seed,
                      'epoch_seed': None if config.seed is None
                      else _epoch_seed(config.seed, epoch)},
            'device': str(device), 'torch': torch.__version__,
        })
        log(f'epoch {epoch + 1}/{config.epochs} loss {train_loss:.4f} '
            f'val f1 {row["val_f1"]:.4f} (best-threshold f1 {row["val_best_f1"]:.4f}) '
            f'lr {lr:.2e}{" *" if row["is_best"] else ""}')
        if on_epoch:
            on_epoch(dict(row))
        if stop:
            stopped_early = True
            log(f'early stopping after {bad_epochs} epochs without improvement')
            break

    completed = history[-1]['epoch'] if history else 0
    if best_path.exists():
        model.load_state(str(best_path), map_location='cpu')
    model.to(device)
    return TrainResult(
        best_score=_finite(best_score) if best_score > -math.inf else float('nan'),
        best_epoch=best_epoch, epochs_completed=completed, stopped_early=stopped_early,
        metrics=history, out_dir=str(out), best_path=str(best_path),
        last_path=str(last_path), seconds=time.time() - started)


def load_model(path, model_kwargs=None, device='cpu'):
    """Load a ``.pth`` (needs ``model_kwargs`` if the architecture is not the default)
    or ``.safetensors`` checkpoint into a :class:`~deepmatcher.MatchingModel`."""
    from .models.core import MatchingModel
    path = str(path)
    if path.endswith('.safetensors'):
        from .serialization import load_safetensors
        return load_safetensors(path, device=device, **(model_kwargs or {}))
    model = MatchingModel(**(model_kwargs or {}))
    model.load_state(path, map_location='cpu')
    return model


def finetune(base_model_path,
             train_ds,
             valid_ds,
             out_dir,
             config: Optional[TrainConfig] = None,
             model_kwargs: Optional[dict] = None,
             **train_kwargs) -> TrainResult:
    """Continue training an existing model on a new dataset.

    The checkpoint (``.pth`` or ``.safetensors``) is loaded, then
    :meth:`MatchingModel.rebind` swaps everything that comes from the training data
    for the new set's: the embedding tables are rebuilt from the new vocabulary (the
    pretrained vectors are frozen, so tokens new to the model get their vector instead
    of ``<unk>`` and no learned weight is lost), and the SIF word probabilities and
    principal components used by the model now describe the **new** training set. For
    the ``sif`` and ``hybrid`` families that shifts the aggregation statistics under
    weights that were fitted to the old ones; with little new data prefer
    ``config.freeze`` on the contextualizer/comparator and a small ``lr``.
    The optimizer, schedule and epoch counter start fresh.

    Args:
        base_model_path: checkpoint to start from.
        train_ds, valid_ds: the new datasets; same attributes and embedding size as the
            original training data.
        out_dir, config: as in :func:`train`. ``config.freeze`` lists parameter-name
            prefixes to keep fixed, e.g. ``['attr_summarizers']``.
        model_kwargs: ``MatchingModel`` constructor arguments of the checkpoint (a
            ``.pth`` does not record them; ``.safetensors`` does).
        **train_kwargs: ``on_epoch``, ``on_batch``, ``log``.
    """
    config = config or TrainConfig()
    if config.seed is not None:
        seed_everything(config.seed)
    model = load_model(base_model_path, model_kwargs)
    model.rebind(train_ds)
    return train(model, train_ds, valid_ds, out_dir, config, resume=False, **train_kwargs)
