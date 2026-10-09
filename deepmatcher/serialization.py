r"""Pickle-free model files: convert deepmatcher checkpoints to safetensors.

A classic deepmatcher checkpoint (``MatchingModel.save_state``) is a dill pickle:
loading one executes arbitrary code, it needs the exact class layout of the
library that wrote it, and it carries the optimizer state (roughly doubling its
size). This module writes the same model as a single ``.safetensors`` file:

* every tensor (weights, vocabulary vectors, SIF word probabilities and principal
  components, the init batch) is stored as a plain tensor;
* vocabularies are stored as UTF-8 byte tensors plus offsets, so token text of
  any size or content round-trips exactly;
* everything else (field layout, tokenizer settings, model constructor kwargs)
  is JSON in the safetensors header.

Weight tensors that share storage (left/right fields share one embedding) are
stored once and restored as shared. Loading never unpickles anything.

Usage::

    from deepmatcher.serialization import convert, load_safetensors
    convert('model.pth', 'model.safetensors', model_kwargs={'attr_summarizer': 'hybrid'})
    model = load_safetensors('model.safetensors')

or from the shell: ``deepmatcher-convert model.pth --verify``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from collections import Counter, OrderedDict, defaultdict

import numpy as np
import torch

from .batch import AttrTensor, MatchingBatch
from .data.dataset import MatchingDataset
from .data.field import MatchingVocab
from .utils import Bunch

logger = logging.getLogger(__name__)

FORMAT = 'deepmatcher-safetensors'
FORMAT_VERSION = 1

# Plain (JSON-able) fields of the training metadata Bunch, stored verbatim.
_PLAIN_META_FIELDS = (
    'path', 'column_naming', 'corresponding_field', 'all_left_fields', 'all_right_fields',
    'canonical_text_fields', 'all_text_fields', 'label_field', 'id_field', 'embeddings',
    'embeddings_cache', 'train_pca', 'ignore_columns', 'tokenize', 'lowercase',
    'include_lengths')
# Fields handled explicitly below.
_STRUCTURED_META_FIELDS = ('text_fields', 'vocabs', 'orig_metadata', 'init_batch')
# Checkpoint entries that only matter for resuming training.
_TRAINING_ONLY_STATE = ('optimizer_state',)
_ATTR_TENSOR_PARTS = AttrTensor._fields  # ('data', 'lengths', 'word_probs', 'pc')


class SafetensorsFormatError(ValueError):
    pass


def _require_safetensors():
    try:
        import safetensors.torch  # noqa: F401
    except ImportError as e:  # pragma: no cover - dependency is declared
        raise ImportError('safetensors is required: pip install safetensors') from e


# ----------------------------------------------------------------------------
# String lists <-> tensors
# ----------------------------------------------------------------------------
def _encode_strings(strings):
    data = [s.encode('utf-8') for s in strings]
    offsets = np.zeros(len(data) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(b) for b in data])
    buf = np.frombuffer(b''.join(data), dtype=np.uint8).copy()
    return torch.from_numpy(buf), torch.from_numpy(offsets)


def _decode_strings(buf, offsets):
    raw = buf.numpy().tobytes()
    off = offsets.tolist()
    return [raw[off[i]:off[i + 1]].decode('utf-8') for i in range(len(off) - 1)]


# ----------------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------------
def _save_embeddings(embeddings, tensors):
    r"""Describe the training embeddings: a name like 'fasttext.en.bin' (the usual
    case, resolved from the vector cache at inference), or an in-memory
    :class:`Vectors` table, which is stored as tensors."""
    from ._torchtext.vocab import Vectors
    from .data.field import FastTextBinary
    items = embeddings if isinstance(embeddings, list) else [embeddings]
    out = []
    for i, emb in enumerate(items):
        if isinstance(emb, str):
            out.append({'kind': 'name', 'name': emb})
        elif isinstance(emb, Vectors) and not isinstance(emb, FastTextBinary):
            if emb.unk_init is not torch.Tensor.zero_:
                raise SafetensorsFormatError('embeddings with a custom unk_init are not supported')
            p = f'embeddings.{i}.'
            tensors[p + 'itos'], tensors[p + 'itos_offsets'] = _encode_strings(emb.itos)
            tensors[p + 'vectors'] = emb.vectors.detach().cpu().contiguous().clone()
            out.append({'kind': 'vectors', 'class': type(emb).__name__, 'dim': emb.dim})
        else:
            raise SafetensorsFormatError(
                f'cannot store embeddings of type {type(emb).__name__}; '
                'train with an embeddings name (e.g. "fasttext.en.bin") instead')
    return {'list': isinstance(embeddings, list), 'items': out}


def _load_embeddings(spec, tensors):
    from ._torchtext.vocab import Vectors
    if spec is None or isinstance(spec, str):  # files written before this field existed
        return spec
    items = []
    for i, item in enumerate(spec['items']):
        if item['kind'] == 'name':
            items.append(item['name'])
            continue
        p = f'embeddings.{i}.'
        vec = Vectors.__new__(Vectors)
        vec.itos = _decode_strings(tensors[p + 'itos'], tensors[p + 'itos_offsets'])
        vec.stoi = {t: n for n, t in enumerate(vec.itos)}
        vec.vectors = tensors[p + 'vectors']
        vec.dim = item['dim']
        vec.unk_init = torch.Tensor.zero_
        items.append(vec)
    return items if spec['list'] else items[0]


_JSON_HEADER_KEYS = ('model_kwargs', 'meta', 'vocabs', 'orig_metadata', 'init_batch',
                     'aliases', 'state', 'dropped', 'source')


def _tag_tuples(value):
    # JSON has no tuples; keep them distinguishable so they round-trip exactly.
    if isinstance(value, tuple):
        return {'__tuple__': [_tag_tuples(v) for v in value]}
    if isinstance(value, list):
        return [_tag_tuples(v) for v in value]
    if isinstance(value, dict):
        return {k: _tag_tuples(v) for k, v in value.items()}
    return value


def _untag_tuples(value):
    if isinstance(value, dict):
        if set(value) == {'__tuple__'}:
            return tuple(_untag_tuples(v) for v in value['__tuple__'])
        return {k: _untag_tuples(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_untag_tuples(v) for v in value]
    return value


def _find_equal(tensor, tensors, prefix):
    t = tensor.detach().cpu()
    for name, candidate in tensors.items():
        if (name.startswith(prefix) and candidate.shape == t.shape
                and candidate.dtype == t.dtype and torch.equal(candidate, t)):
            return name
    return None


def _tensor_key(t):
    return (t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), t.stride(),
            t.dtype)


def save_safetensors(model, path, model_kwargs, source=None):
    r"""Write a loaded :class:`~deepmatcher.MatchingModel` to ``path`` as safetensors.

    Args:
        model: A model restored with :meth:`MatchingModel.load_state` (or trained).
        path: Output ``.safetensors`` path.
        model_kwargs (dict): Constructor kwargs needed to rebuild the architecture,
            e.g. ``{'attr_summarizer': 'hybrid'}``. Checkpoints don't record these.
        source (dict): Optional provenance stored in the header.
    """
    _require_safetensors()
    from safetensors.torch import save_file

    if not hasattr(model, 'state_meta'):
        raise SafetensorsFormatError('model has no training metadata (state_meta); '
                                     'load it with load_state() or train it first')
    meta = model.state_meta
    unknown = set(vars(meta)) - set(_PLAIN_META_FIELDS) - set(_STRUCTURED_META_FIELDS)
    if unknown:
        raise SafetensorsFormatError(f'unsupported metadata fields: {sorted(unknown)}')

    tensors = {}
    header = {}

    # --- weights, de-duplicated by storage -----------------------------------
    aliases, seen = {}, {}
    for name, t in model.state_dict().items():
        key = _tensor_key(t)
        if key in seen:
            aliases[name] = seen[key]
        else:
            seen[key] = name
            tensors['model.' + name] = t.detach().cpu().contiguous().clone()

    # --- vocabularies (left/right fields usually share one object) -----------
    vocab_groups, vocab_index = [], {}
    for field, vocab in meta.vocabs.items():
        if id(vocab) in vocab_index:
            vocab_groups[vocab_index[id(vocab)]]['fields'].append(field)
            continue
        i = len(vocab_groups)
        vocab_index[id(vocab)] = i
        unknown = set(vars(vocab)) - {'freqs', 'itos', 'unk_index', 'stoi', 'vectors'}
        if unknown:
            raise SafetensorsFormatError(f'vocab {field}: unsupported attributes {unknown}')
        p = f'vocab.{i}.'
        tensors[p + 'itos'], tensors[p + 'itos_offsets'] = _encode_strings(vocab.itos)
        freq_tokens = list(vocab.freqs.keys())
        tensors[p + 'freq_tokens'], tensors[p + 'freq_offsets'] = _encode_strings(freq_tokens)
        tensors[p + 'freq_counts'] = torch.tensor([vocab.freqs[t] for t in freq_tokens],
                                                  dtype=torch.int64)
        stoi_from_itos = {t: n for n, t in enumerate(vocab.itos)}
        stoi_matches = dict(vocab.stoi) == stoi_from_itos
        if not stoi_matches:
            stoi_tokens = list(vocab.stoi.keys())
            tensors[p + 'stoi_tokens'], tensors[p + 'stoi_offsets'] = _encode_strings(stoi_tokens)
            tensors[p + 'stoi_ids'] = torch.tensor([vocab.stoi[t] for t in stoi_tokens],
                                                   dtype=torch.int64)
        has_vectors = vocab.vectors is not None
        vectors_alias = None
        if has_vectors:
            # The embedding layer is initialized from these vectors (and frozen), so
            # they are usually stored already; keep one copy.
            vectors_alias = _find_equal(vocab.vectors, tensors, prefix='model.')
            if vectors_alias is None:
                tensors[p + 'vectors'] = vocab.vectors.detach().cpu().contiguous().clone()
        vocab_groups.append({
            'fields': [field], 'unk_index': getattr(vocab, 'unk_index', None),
            'stoi_from_itos': stoi_matches, 'has_vectors': has_vectors,
            'vectors_alias': vectors_alias,
            'stoi_default_unk': isinstance(vocab.stoi, defaultdict)
                                and vocab.stoi.default_factory is not None})

    # --- SIF metadata ---------------------------------------------------------
    orig = meta.orig_metadata
    unknown = set(orig) - {'word_probs', 'totals', 'pc'}
    if unknown:
        raise SafetensorsFormatError(f'unsupported orig_metadata keys: {sorted(unknown)}')
    meta_info = {'keys': list(orig.keys())}
    if 'word_probs' in orig:
        for field, probs in orig['word_probs'].items():
            ids = list(probs.keys())
            tensors[f'word_probs.{field}.ids'] = torch.tensor(ids, dtype=torch.int64)
            tensors[f'word_probs.{field}.values'] = torch.tensor([probs[k] for k in ids],
                                                                 dtype=torch.float64)
        meta_info['word_probs_fields'] = list(orig['word_probs'].keys())
    if 'totals' in orig:
        meta_info['totals'] = orig['totals']
    if 'pc' in orig:
        pc_info = {}
        for field, pc in orig['pc'].items():
            if pc is None:
                pc_info[field] = None
                continue
            arr = np.asarray(pc)
            tensors[f'pc.{field}'] = torch.from_numpy(np.ascontiguousarray(arr).copy())
            pc_info[field] = 'ndarray' if isinstance(pc, np.ndarray) else 'tensor'
        meta_info['pc'] = pc_info

    # --- init batch -------------------------------------------------------------
    batch_info = OrderedDict()
    for attr, value in vars(meta.init_batch).items():
        if isinstance(value, AttrTensor):
            parts = {}
            for part in _ATTR_TENSOR_PARTS:
                t = getattr(value, part)
                parts[part] = t is not None
                if t is not None:
                    tensors[f'init_batch.{attr}.{part}'] = t.detach().cpu().contiguous().clone()
            batch_info[attr] = {'kind': 'attr', 'parts': parts}
        elif isinstance(value, torch.Tensor):
            tensors[f'init_batch.{attr}'] = value.detach().cpu().contiguous().clone()
            batch_info[attr] = {'kind': 'tensor'}
        elif isinstance(value, (list, tuple)) and all(
                isinstance(v, (str, int, float, bool)) or v is None for v in value):
            # e.g. the id column of the batch, kept as raw values
            batch_info[attr] = {'kind': 'list', 'tuple': isinstance(value, tuple),
                                'value': list(value)}
        else:
            raise SafetensorsFormatError(f'init_batch.{attr}: unsupported type {type(value)}')

    # --- other checkpoint state -------------------------------------------------
    extra_state, dropped = {}, []
    for name in sorted(model._train_buffers):
        if name == 'state_meta':
            continue
        if name in _TRAINING_ONLY_STATE:
            dropped.append(name)
            continue
        value = getattr(model, name, None)
        if isinstance(value, torch.Tensor):
            tensors[f'state.{name}'] = value.detach().cpu().contiguous().clone()
            extra_state[name] = 'tensor'
        elif isinstance(value, (int, float, str, bool)) or value is None:
            extra_state[name] = {'value': value}
        else:
            raise SafetensorsFormatError(f'checkpoint entry {name!r}: unsupported type '
                                         f'{type(value)}')

    plain = {k: getattr(meta, k) for k in _PLAIN_META_FIELDS if hasattr(meta, k)}
    plain['text_fields'] = {k: list(v) for k, v in meta.text_fields.items()}
    plain['embeddings'] = _save_embeddings(meta.embeddings, tensors)
    from . import __version__
    header = {
        'format': FORMAT,
        'format_version': str(FORMAT_VERSION),
        'deepmatcher_version': __version__,
        'model_kwargs': json.dumps(model_kwargs),
        'meta': json.dumps(_tag_tuples(plain)),
        'vocabs': json.dumps(vocab_groups),
        'orig_metadata': json.dumps(meta_info),
        'init_batch': json.dumps(batch_info),
        'aliases': json.dumps(aliases),
        'state': json.dumps(extra_state),
        'dropped': json.dumps(dropped),
        'source': json.dumps(source or {}),
    }
    tmp = str(path) + '.tmp'
    save_file(tensors, tmp, metadata=header)
    os.replace(tmp, path)
    return path


# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------
def read_header(path):
    r"""Return the decoded safetensors header of a converted model (no tensors)."""
    _require_safetensors()
    from safetensors import safe_open
    with safe_open(str(path), framework='pt') as f:
        raw = f.metadata() or {}
    if raw.get('format') != FORMAT:
        raise SafetensorsFormatError(f'{path} is not a {FORMAT} file')
    return {k: json.loads(v) if k in _JSON_HEADER_KEYS else v for k, v in raw.items()}


def load_safetensors(path, device='cpu', **model_kwargs_override):
    r"""Rebuild a :class:`~deepmatcher.MatchingModel` from a converted safetensors file.

    Equivalent to ``MatchingModel(**model_kwargs).load_state(pth)`` on the original
    checkpoint, minus the optimizer state. Nothing is unpickled.
    """
    _require_safetensors()
    from safetensors.torch import load_file

    from .models.core import MatchingModel

    header = read_header(path)
    if int(header['format_version']) > FORMAT_VERSION:
        raise SafetensorsFormatError(
            f'{path} uses format version {header["format_version"]}, this deepmatcher '
            f'reads up to {FORMAT_VERSION}')
    tensors = load_file(str(path), device='cpu')

    plain = _untag_tuples(header['meta'])
    meta = Bunch(**{k: v for k, v in plain.items() if k != 'text_fields'})
    meta.text_fields = OrderedDict((k, tuple(v)) for k, v in plain['text_fields'].items())
    meta.embeddings = _load_embeddings(plain.get('embeddings'), tensors)

    # Vocabularies.
    vocabs = OrderedDict()
    for i, group in enumerate(header['vocabs']):
        p = f'vocab.{i}.'
        vocab = MatchingVocab.__new__(MatchingVocab)
        tokens = _decode_strings(tensors[p + 'freq_tokens'], tensors[p + 'freq_offsets'])
        vocab.freqs = Counter(dict(zip(tokens, tensors[p + 'freq_counts'].tolist())))
        vocab.itos = _decode_strings(tensors[p + 'itos'], tensors[p + 'itos_offsets'])
        vocab.unk_index = group['unk_index']
        if group['stoi_from_itos']:
            stoi_items = ((t, n) for n, t in enumerate(vocab.itos))
        else:
            stoi_items = zip(
                _decode_strings(tensors[p + 'stoi_tokens'], tensors[p + 'stoi_offsets']),
                tensors[p + 'stoi_ids'].tolist())
        vocab.stoi = (defaultdict(vocab._default_unk_index) if group['stoi_default_unk']
                      else defaultdict())
        vocab.stoi.update(stoi_items)
        if not group['has_vectors']:
            vocab.vectors = None
        elif group.get('vectors_alias'):
            # Own copy: process_unlabeled extends vocab vectors in place.
            vocab.vectors = tensors[group['vectors_alias']].clone()
        else:
            vocab.vectors = tensors[p + 'vectors']
        for field in group['fields']:
            vocabs[field] = vocab
    # Keep the original field order.
    meta.vocabs = OrderedDict((f, vocabs[f]) for f in _vocab_field_order(header, vocabs))

    # SIF metadata.
    info = header['orig_metadata']
    orig = OrderedDict()
    for key in info['keys']:
        if key == 'word_probs':
            orig[key] = OrderedDict(
                (f, dict(zip(tensors[f'word_probs.{f}.ids'].tolist(),
                             tensors[f'word_probs.{f}.values'].tolist())))
                for f in info['word_probs_fields'])
        elif key == 'totals':
            orig[key] = info['totals']
        elif key == 'pc':
            orig[key] = OrderedDict()
            for f, kind in info['pc'].items():
                if kind is None:
                    orig[key][f] = None
                else:
                    t = tensors[f'pc.{f}']
                    orig[key][f] = t.numpy() if kind == 'ndarray' else t
    meta.orig_metadata = orig

    # Init batch.
    batch = MatchingBatch.__new__(MatchingBatch)
    for attr, spec in header['init_batch'].items():
        if spec['kind'] == 'attr':
            parts = [tensors[f'init_batch.{attr}.{part}'] if spec['parts'][part] else None
                     for part in _ATTR_TENSOR_PARTS]
            setattr(batch, attr, AttrTensor(*parts))
        elif spec['kind'] == 'list':
            setattr(batch, attr, tuple(spec['value']) if spec['tuple'] else list(spec['value']))
        else:
            setattr(batch, attr, tensors[f'init_batch.{attr}'])
    meta.init_batch = batch

    # Rebuild exactly like MatchingModel.load_state().
    model_kwargs = dict(header['model_kwargs'])
    model_kwargs.update(model_kwargs_override)
    model = MatchingModel(**model_kwargs)
    model._register_train_buffer('state_meta', meta)
    for name, spec in header['state'].items():
        value = tensors[f'state.{name}'] if spec == 'tensor' else spec['value']
        model._register_train_buffer(name, value)

    train_info = copy.copy(model.state_meta)
    train_info.metadata = train_info.orig_metadata
    MatchingDataset.finalize_metadata(train_info)
    model.initialize(train_info, model.state_meta.init_batch)

    state_dict = OrderedDict()
    aliases = header['aliases']
    for name in model.state_dict().keys():
        source = aliases.get(name, name)
        state_dict[name] = tensors['model.' + source]
    model.load_state_dict(state_dict)
    return model.to(device)


def _vocab_field_order(header, vocabs):
    order = header['meta'].get('all_text_fields') or []
    return [f for f in order if f in vocabs] + [f for f in vocabs if f not in order]


# ----------------------------------------------------------------------------
# Convert + verify
# ----------------------------------------------------------------------------
def sha256sum(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def convert(pth_path, out_path=None, model_kwargs=None, verify=True):
    r"""Convert a ``save_state`` checkpoint to safetensors (and optionally verify it).

    Returns:
        dict: ``{'output', 'source_bytes', 'output_bytes', 'verified', 'differences'}``
    """
    from .models.core import MatchingModel

    model_kwargs = model_kwargs or {'attr_summarizer': 'hybrid'}
    out_path = out_path or os.path.splitext(pth_path)[0] + '.safetensors'
    with _nan_checks_disabled():
        model = MatchingModel(**model_kwargs)
        model.load_state(pth_path, map_location='cpu')
        source = {'file': os.path.basename(pth_path), 'sha256': sha256sum(pth_path),
                  'bytes': os.path.getsize(pth_path)}
        save_safetensors(model, out_path, model_kwargs, source=source)
        result = {'output': out_path, 'source_bytes': source['bytes'],
                  'output_bytes': os.path.getsize(out_path), 'verified': None,
                  'differences': []}
        if verify:
            reloaded = load_safetensors(out_path)
            result['differences'] = compare_models(model, reloaded)
            result['verified'] = not result['differences']
    return result


def compare_models(a, b):
    r"""List every difference between two loaded models' weights and metadata.

    Tensors are compared bit-for-bit (dtype, shape and values); an empty list
    means the models are interchangeable.
    """
    diffs = []
    sa, sb = a.state_dict(), b.state_dict()
    if list(sa) != list(sb):
        diffs.append(f'state_dict keys differ: {set(sa) ^ set(sb)}')
    for k in sa:
        if k in sb and not _tensor_equal(sa[k], sb[k]):
            diffs.append(f'weight {k} differs')
    _compare_values('meta', _meta_view(a.state_meta), _meta_view(b.state_meta), diffs)
    for name in sorted(set(a._train_buffers) | set(b._train_buffers)):
        if name in ('state_meta',) + _TRAINING_ONLY_STATE:
            continue
        _compare_values(name, getattr(a, name, None), getattr(b, name, None), diffs)
    return diffs


def _meta_view(meta):
    view = {k: getattr(meta, k) for k in _PLAIN_META_FIELDS if hasattr(meta, k)}
    view['text_fields'] = {k: tuple(v) for k, v in meta.text_fields.items()}
    view['embeddings'] = _embeddings_view(meta.embeddings)
    view['vocabs'] = {f: {'itos': v.itos, 'stoi': dict(v.stoi), 'freqs': dict(v.freqs),
                          'freq_order': list(v.freqs), 'unk_index': v.unk_index,
                          'vectors': v.vectors,
                          'stoi_default': v.stoi.default_factory is not None}
                      for f, v in meta.vocabs.items()}
    view['vocab_sharing'] = sorted(sorted(f for f, v in meta.vocabs.items() if v is shared)
                                   for shared in {id(v): v for v in meta.vocabs.values()}.values())
    view['orig_metadata'] = meta.orig_metadata
    view['init_batch'] = {k: (tuple(v) if isinstance(v, AttrTensor) else v)
                          for k, v in vars(meta.init_batch).items()}
    return view


def _embeddings_view(embeddings):
    items = embeddings if isinstance(embeddings, list) else [embeddings]
    view = [e if isinstance(e, str) else
            {'itos': e.itos, 'stoi': dict(e.stoi), 'vectors': e.vectors, 'dim': e.dim}
            for e in items]
    return view if isinstance(embeddings, list) else view[0]


def _tensor_equal(x, y):
    return (x.dtype == y.dtype and x.shape == y.shape
            and torch.equal(x.cpu(), y.cpu()))


def _compare_values(path, x, y, diffs):
    if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
        if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
                and _tensor_equal(x, y)):
            diffs.append(f'{path} differs')
    elif isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        if not (type(x) is type(y) and x.dtype == y.dtype and np.array_equal(x, y)):
            diffs.append(f'{path} differs')
    elif isinstance(x, dict) and isinstance(y, dict):
        if list(x.keys()) != list(y.keys()) and set(x) != set(y):
            diffs.append(f'{path}: keys differ')
            return
        for k in x:
            _compare_values(f'{path}.{k}', x[k], y[k], diffs)
    elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        if len(x) != len(y) or type(x) is not type(y):
            diffs.append(f'{path}: length/type differs')
            return
        for i, (u, v) in enumerate(zip(x, y)):
            _compare_values(f'{path}[{i}]', u, v, diffs)
    elif x != y or type(x) is not type(y):
        diffs.append(f'{path}: {x!r} != {y!r}')


class _nan_checks_disabled:
    def __enter__(self):
        import deepmatcher as dm
        self._prev = dm._check_nan
        dm._check_nan = False

    def __exit__(self, *exc):
        import deepmatcher as dm
        dm._check_nan = self._prev
        return False
