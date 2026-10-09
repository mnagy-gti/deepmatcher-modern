# deepmatcher-modern

[DeepMatcher](https://github.com/anhaidgroup/deepmatcher) (Mudgal et al., SIGMOD 2018) entity
matching, ported to **current PyTorch** and Python, with:

- **CUDA, Apple Metal (MPS) and CPU** inference from one code base
- **safetensors** model files: pickle-free, about 2.5× smaller, verified bit for bit against the checkpoint
- optional **fp16 / bf16 compute**, **fp16 / bf16 / fp8 weight storage** and **`torch.compile`**
- the original API and the original `.pth` checkpoints, unchanged

The import name stays `deepmatcher`. Existing checkpoints pickle `deepmatcher.*` classes, and
they load as they are.

## Why a port was needed

Upstream DeepMatcher (last released 2021) is built on `torchtext.legacy`. torchtext removed that
API in 0.12 and has since been discontinued, which pins DeepMatcher to **torch ≤ 1.10 /
Python ≤ 3.9**. This package vendors the pure-Python torchtext pieces DeepMatcher uses, under
`deepmatcher._torchtext`, and fixes what broke or degraded on newer torch:

| Change | Effect |
|---|---|
| `torch.load(..., weights_only=False)` for checkpoints and dataset caches | torch ≥ 2.6 refuses them otherwise |
| Batch metadata follows the data's device, not only CUDA | MPS works |
| One host transfer per batch instead of per token/row (`batch.py`, `runner.py`) | MPS batch loading went from 103 s to 0.4 s; removes a sync per row on CUDA |
| NaN checks raise `FloatingPointError` instead of calling `pdb.set_trace()`, can actually be disabled (`global` was missing), and use tensor hooks instead of deprecated module backward hooks | Unattended jobs no longer hang; inference skips a device sync per module |
| `SoftNLLLoss` keeps the element-wise KLDiv mean explicitly | Training stays the same once torch changes `'mean'` |
| No `logging.basicConfig()` at import | Host applications keep their own logging |
| nltk data fetched only when missing, including `punkt_tab` | Inference works with current nltk and offline |
| torchtext's process-wide TLS-verification bypass in `Vectors.cache()` removed | Downloads verify certificates |

All 43 upstream tests pass, ported from nose to pytest.

## Install

```bash
pip install "deepmatcher-modern @ git+https://github.com/mnagy-gti/deepmatcher-modern"
```

Requires Python ≥ 3.10 and torch ≥ 2.4. Install torch first from the index that matches
your hardware (CUDA, CPU or the default macOS wheels).

## Convert checkpoints to safetensors

```bash
deepmatcher-convert model.pth                      # writes model.safetensors, verified
deepmatcher-convert *.pth --out-dir converted/
deepmatcher-convert model.pth --attr-summarizer rnn --model-kwargs '{"hidden_size": 150}'
```

A checkpoint does not record the `MatchingModel` constructor arguments, so pass them if they
weren't the defaults (`attr_summarizer='hybrid'`). They are stored in the safetensors header,
which makes the file self-describing.

What gets written:

- **every tensor**: weights, vocabulary vectors, SIF word probabilities and principal components,
  the init batch. Tensors that share storage, such as left/right fields sharing one embedding
  or vocabulary vectors equal to the embedding table, are stored once.
- **vocabularies** as UTF-8 byte tensors plus offsets, so any token round-trips exactly
- **everything else** (field layout, tokenizer settings, constructor kwargs, provenance with the
  source file's SHA-256) as JSON in the header
- the **optimizer state is dropped**; it only matters for resuming training

Verification reloads the written file and compares every weight, vocabulary, metadata entry
and the init batch bit for bit against the checkpoint. The converter exits with status 1 on any
difference.

```python
from deepmatcher.serialization import convert, load_safetensors, read_header
model = load_safetensors("model.safetensors")   # nothing is unpickled
```

## Inference runtime

```python
from deepmatcher.runtime import InferenceOptions, Predictor

predictor = Predictor.load("model.safetensors",          # or a .pth
                           InferenceOptions(device="auto",  # cuda > mps > cpu
                                            compute="fp32", weights="fp32", compile=False))
dataset = predictor.process_unlabeled("pairs.csv", vector_cache="~/.vector_cache")
scores = predictor.predict(dataset, output_attributes=True)
print(predictor.stats)   # load/process/predict seconds, model bytes, pairs
```

With the defaults (fp32, no compile), `Predictor.predict` returns exactly what
`MatchingModel.run_prediction` returns. It just doesn't deep-copy the model on every call.

| Option | Values | Notes |
|---|---|---|
| `device` | `auto`, `cuda`, `mps`, `cpu` | MPS needs a native macOS Python (not Docker) |
| `compute` | `fp32`, `fp16`, `bf16` | `torch.autocast`; a win on tensor-core GPUs, usually not on CPU |
| `weights` | `fp32`, `fp16`, `bf16`, `fp8` | Storage of embeddings and linear layers (embeddings are ~90 % of the model). `fp8` is e4m3 with a per-row scale, decoded through a 256-entry lookup table, so it runs on any device including MPS. It saves memory, not time |
| `compile` | `True` / `False` | `torch.compile(dynamic=True)`; tens of seconds of compile time up front |

Anything other than fp32 without compile moves scores slightly. Compare against an fp32 run
before using it for real decisions.

## Numerical parity with the original

These checks used GTI's production body-matching models (4 models, 3 countries, 7,788 candidate
pairs) with **deepmatcher 0.1.2 / torch 1.10.2** as the baseline:

| Run | vs baseline |
|---|---|
| torch 2.14.1, CPU (macOS), `.pth` | every match decision identical, max \|Δscore\| 5.0e-6 |
| torch 2.14.1, CPU (Linux arm64, same platform as the baseline) | identical decisions, max \|Δ\| 5.0e-6 |
| torch 2.14.1, MPS | identical decisions, max \|Δ\| 6.7e-6 |
| `.safetensors` vs `.pth` (same device) | **bit-identical** |

The remaining ~1e-6 differences are float32 kernel differences between torch builds. They are
the same size as the gap between macOS and Linux on the *same* torch version (3e-6).

## Tests

```bash
pip install -e ".[dev]"
pytest                 # upstream suite + safetensors / runtime / device tests
pytest -m "not slow"   # skip the torch.compile check
```

## License

BSD-3-Clause, the same as upstream DeepMatcher. See `LICENSE`. The vendored torchtext code is
BSD-3-Clause too; see `deepmatcher/_torchtext/LICENSE`.
