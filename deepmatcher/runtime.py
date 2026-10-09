r"""Inference on current PyTorch: device selection, reduced precision, torch.compile.

The defaults (``fp32`` compute and weights, no compile) reproduce the original
deepmatcher predictions; everything else is opt-in and should be checked against
an fp32 run before it is used for real decisions (scores move slightly).

Two independent precision knobs:

``compute`` (``fp32`` | ``fp16`` | ``bf16``)
    Arithmetic precision, via :func:`torch.autocast`. Faster on GPUs with
    tensor cores and on Apple silicon; little or no gain on most CPUs.

``weights`` (``fp32`` | ``fp16`` | ``bf16`` | ``fp8``)
    Storage precision of the embedding tables and linear layers (embeddings
    are ~90% of a deepmatcher model). Weights are decoded to the compute
    dtype on the fly. ``fp8`` stores float8 e4m3 with a per-row scale and
    decodes through a 256-entry lookup table, so it works on any device
    (including MPS, which has no float8 dtype). It saves memory, not time.

Example::

    from deepmatcher.runtime import InferenceOptions, Predictor
    predictor = Predictor.load('model.safetensors', InferenceOptions(device='auto'))
    dataset = predictor.process_unlabeled('pairs.csv', vector_cache='~/.vector_cache')
    scores = predictor.predict(dataset)
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DEVICES = ('auto', 'cuda', 'mps', 'cpu')
COMPUTE_DTYPES = {'fp32': None, 'fp16': torch.float16, 'bf16': torch.bfloat16}
WEIGHT_DTYPES = ('fp32', 'fp16', 'bf16', 'fp8')
FP8_MAX = 448.0  # largest finite float8_e4m3fn value


# ----------------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------------
def select_device(requested='auto'):
    r"""Resolve ``auto`` to CUDA if present, else Apple Metal (MPS), else CPU."""
    requested = (requested or 'auto').lower()
    if requested not in DEVICES and not requested.startswith('cuda:'):
        raise ValueError(f'unknown device {requested!r}; expected one of {DEVICES}')
    if requested == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    if requested.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Metal (MPS) was requested but is not available')
    return torch.device(requested)


def available_devices():
    r"""Devices usable on this machine, best first."""
    out = []
    if torch.cuda.is_available():
        out.append('cuda')
    if torch.backends.mps.is_available():
        out.append('mps')
    out.append('cpu')
    return out


@dataclasses.dataclass
class InferenceOptions:
    device: str = 'auto'
    compute: str = 'fp32'
    weights: str = 'fp32'
    compile: bool = False
    batch_size: int = 512
    threads: int | None = None

    def __post_init__(self):
        if self.compute not in COMPUTE_DTYPES:
            raise ValueError(f'compute must be one of {tuple(COMPUTE_DTYPES)}')
        if self.weights not in WEIGHT_DTYPES:
            raise ValueError(f'weights must be one of {WEIGHT_DTYPES}')

    @property
    def is_reference(self):
        r"""True for the settings that reproduce original deepmatcher scores."""
        return self.compute == 'fp32' and self.weights == 'fp32' and not self.compile

    def describe(self):
        parts = [f'device={self.device}', f'compute={self.compute}', f'weights={self.weights}']
        if self.compile:
            parts.append('compile')
        return ' '.join(parts)

    def to_dict(self):
        return dataclasses.asdict(self)


# ----------------------------------------------------------------------------
# Reduced-precision weight storage
# ----------------------------------------------------------------------------
def _fp8_lut():
    codes = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    lut = codes.view(torch.float8_e4m3fn).to(torch.float32)
    return torch.nan_to_num(lut, nan=0.0)  # 0x7f / 0xff are NaN codes we never emit


def _quantize(weight, fmt):
    r"""Return (stored, scale) for a 2-D weight. ``scale`` is per row (or None)."""
    w = weight.detach().float().cpu()
    if fmt in ('fp16', 'bf16'):
        return w.to(COMPUTE_DTYPES[fmt]), None
    if fmt == 'fp8':
        amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        scale = amax / FP8_MAX
        q = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        return q.view(torch.uint8), scale.squeeze(1)
    raise ValueError(fmt)


class _StoredWeight(nn.Module):
    """Holds a weight in reduced precision and decodes it (rows or whole)."""

    def __init__(self, weight, fmt):
        super().__init__()
        self.fmt = fmt
        stored, scale = _quantize(weight, fmt)
        self.register_buffer('stored', stored)
        self.register_buffer('scale', scale if scale is not None else torch.empty(0))
        self.register_buffer('lut', _fp8_lut() if fmt == 'fp8' else torch.empty(0))

    def decode(self, rows=None, dtype=torch.float32):
        stored = self.stored if rows is None else self.stored[rows]
        if self.fmt == 'fp8':
            values = self.lut[stored.long()]
            scale = self.scale if rows is None else self.scale[rows]
            return (values * scale.unsqueeze(-1)).to(dtype)
        return stored.to(dtype)


class QuantizedEmbedding(nn.Module):
    r"""Drop-in :class:`torch.nn.Embedding` replacement with reduced-precision storage.

    Only the looked-up rows are decoded, so lookups stay cheap even for fp8.
    """

    def __init__(self, embedding, fmt):
        super().__init__()
        self.num_embeddings, self.embedding_dim = embedding.weight.shape
        self.padding_idx = embedding.padding_idx
        self.weight_store = _StoredWeight(embedding.weight, fmt)

    def forward(self, input):
        dtype = torch.get_autocast_dtype(input.device.type) \
            if torch.is_autocast_enabled(input.device.type) else torch.float32
        return self.weight_store.decode(input, dtype=dtype)


class QuantizedLinear(nn.Module):
    r"""Drop-in :class:`torch.nn.Linear` replacement with reduced-precision storage."""

    def __init__(self, linear, fmt):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.weight_store = _StoredWeight(linear.weight, fmt)
        self.bias = None if linear.bias is None else nn.Parameter(
            linear.bias.detach().clone(), requires_grad=False)

    def forward(self, input):
        weight = self.weight_store.decode(dtype=input.dtype)
        bias = None if self.bias is None else self.bias.to(input.dtype)
        return F.linear(input, weight, bias)


def quantize_weights(model, fmt):
    r"""Replace embeddings and linear layers of ``model`` with reduced-precision ones.

    Recurrent layers keep fp32 weights (their fused kernels need them).
    Returns the number of replaced modules.
    """
    if fmt == 'fp32':
        return 0
    if fmt not in WEIGHT_DTYPES:
        raise ValueError(fmt)
    replaced = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Embedding):
                setattr(parent, name, QuantizedEmbedding(child, fmt))
                replaced += 1
            elif isinstance(child, nn.Linear):
                setattr(parent, name, QuantizedLinear(child, fmt))
                replaced += 1
    return replaced


def model_nbytes(model):
    r"""Bytes held by a model's parameters and buffers (shared storage counted once)."""
    seen, total = set(), 0
    for t in list(model.parameters()) + list(model.buffers()):
        key = t.untyped_storage().data_ptr()
        if key not in seen:
            seen.add(key)
            total += t.untyped_storage().nbytes()
    return total


# ----------------------------------------------------------------------------
# Predictor
# ----------------------------------------------------------------------------
@contextlib.contextmanager
def nan_checks_disabled():
    import deepmatcher as dm
    previous = dm._check_nan
    dm._check_nan = False
    try:
        yield
    finally:
        dm._check_nan = previous


def load_model(path, **model_kwargs):
    r"""Load a ``.pth`` checkpoint or a converted ``.safetensors`` file on CPU."""
    from .models.core import MatchingModel
    from .serialization import load_safetensors
    with nan_checks_disabled():
        if str(path).endswith('.safetensors'):
            return load_safetensors(path, **model_kwargs)
        model = MatchingModel(**{'attr_summarizer': 'hybrid', **model_kwargs})
        model.load_state(path, map_location='cpu')
        return model


class Predictor:
    r"""Owns one model for inference and scores datasets with :class:`InferenceOptions`.

    Unlike :meth:`MatchingModel.run_prediction` it does not deep-copy the model
    for every call (the model is only ever used for inference here).
    """

    def __init__(self, model, options=None):
        self.model = model.eval()
        self.options = options or InferenceOptions()
        self.device = select_device(self.options.device)
        if self.options.threads:
            torch.set_num_threads(self.options.threads)
        self.stats = {}

    @classmethod
    def load(cls, path, options=None, **model_kwargs):
        start = time.perf_counter()
        predictor = cls(load_model(path, **model_kwargs), options)
        predictor.stats['load_seconds'] = time.perf_counter() - start
        return predictor

    def process_unlabeled(self, path, ignore_columns=None, vector_cache=None):
        r""":func:`deepmatcher.data.process_unlabeled` for this model.

        Columns the model was trained to ignore are always ignored as well.
        ``vector_cache`` overrides the training machine's embeddings cache path.
        """
        from .data import process_unlabeled
        if vector_cache:
            self.model.meta.embeddings_cache = os.path.expanduser(vector_cache)
        ignore = sorted(set(ignore_columns or []) | set(self.model.meta.ignore_columns or []))
        start = time.perf_counter()
        dataset = process_unlabeled(path=path, trained_model=self.model, ignore_columns=ignore)
        self.stats['process_seconds'] = time.perf_counter() - start
        return dataset

    def _autocast(self):
        dtype = COMPUTE_DTYPES[self.options.compute]
        if dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def predict(self, dataset, output_attributes=False, progress_style='bar', log_freq=5):
        r"""Score ``dataset``; same output as :meth:`MatchingModel.run_prediction`."""
        from .runner import Runner
        opts = self.options
        model = self.model
        start = time.perf_counter()
        with nan_checks_disabled():
            # Embeddings must cover the dataset's extended vocabulary.
            model._reset_embeddings(dataset.vocabs)
            self.stats['quantized_modules'] = quantize_weights(model, opts.weights)
        model.to(self.device).eval()
        self.stats['model_bytes'] = model_nbytes(model)
        runner_model = model
        if opts.compile:
            runner_model = torch.compile(model, dynamic=True)
            runner_model.meta = model.meta  # Runner reads model.meta
        with torch.inference_mode(), self._autocast():
            predictions = Runner._run('PREDICT', runner_model, dataset,
                                      return_predictions=True, device=self.device,
                                      batch_size=opts.batch_size,
                                      progress_style=progress_style, log_freq=log_freq)
        self.stats['predict_seconds'] = time.perf_counter() - start
        self.stats['pairs'] = len(predictions)
        table = pd.DataFrame(predictions, columns=(dataset.id_field, 'match_score'))
        table = table.set_index(dataset.id_field)
        if output_attributes:
            raw = pd.read_csv(dataset.path).set_index(dataset.id_field)
            raw.index = raw.index.astype('str')
            table = table.join(raw)
        return table
