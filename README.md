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

## Training

`MatchingModel.run_train` (the original API) still works exactly as before. `deepmatcher.training`
adds a modern loop around the same models, with checkpoints in the original `.pth` format, so
every existing loader (and `deepmatcher-convert`) reads them.

```python
import deepmatcher as dm
from deepmatcher.training import TrainConfig, train, finetune
from deepmatcher.evaluation import evaluate

train_ds, valid_ds, test_ds = dm.data.process(path, train="train.csv", validation="valid.csv",
                                              test="test.csv", embeddings="fasttext.en.bin")
model = dm.MatchingModel(attr_summarizer="hybrid-transformer")
result = train(model, train_ds, valid_ds, "runs/exp1",
               TrainConfig(epochs=10, device="auto", seed=42, lr_schedule="cosine"),
               on_epoch=lambda m: print(m["epoch"], m["val_f1"]),   # UI hooks
               on_batch=lambda b: None)                             # every ~20 batches
report = evaluate(model, test_ds, device="auto", group_by="country")
report.best_f1_threshold, report.threshold_for_precision(0.99), report.groups
```

`train` writes to `out_dir`: `best.pth` (best validation score; EMA weights when EMA is on),
`last.pth` (raw weights + optimizer, what `resume=True` continues from), `ema.pt`,
`metrics.jsonl` (one JSON object per epoch: `epoch, train_loss, train_seconds, val_precision,
val_recall, val_f1` at threshold 0.5, `val_best_f1_threshold, val_best_f1, val_roc_auc,
val_pr_auc, lr, eval_seconds, timestamp, is_best`) and `train_state.json` (config, epoch, best
score, schedule state, seeds). Checkpoints hold CPU tensors, so a file written on MPS or CUDA
loads anywhere.

**Reproducibility and resume.** With `seed` set, initialization is seeded and every epoch
re-seeds torch/numpy/random and the example shuffle from `(seed, epoch)`. An interrupted run
continued with `train(..., resume=True)` (same data, `epochs` = the total) therefore gives the
same metrics as an uninterrupted run (checked bit for bit on CPU; MPS/CUDA can differ in the last
digits through nondeterministic kernels). Without a `last.pth`, `resume=True` starts fresh.
The training loop shuffles examples completely each epoch; the original loop shuffles batches
within chunks of 100 batches, which does not mix a small file at all.

**Fine-tuning.** `finetune(base_model_path, train_ds, valid_ds, out_dir, config, model_kwargs=...)`
loads a `.pth` or `.safetensors` checkpoint, calls `MatchingModel.rebind(train_ds)`, then trains
with a fresh optimizer. The pretrained embedding vectors are frozen in this model, so rebinding
rebuilds the tables from the new vocabulary (new tokens get their vector instead of `<unk>`;
nothing learned is lost) and every other weight is kept. Caveat: the SIF word probabilities and
principal components, and the metadata stored in the new checkpoint, come from the **new**
training set. That shifts the aggregation statistics under weights fitted to the old ones; with
little new data use `TrainConfig(freeze=["attr_summarizers"], lr=1e-4)` and compare with the base
model on a validation set from the target domain. Attributes and embedding size must match.

**Per-example weights.** If the CSVs have a numeric `weight` column (the same column in all splits;
configurable with `process(weight_attr=...)`, ignored when absent), batches expose `batch.weight`
and `TrainConfig(loss="weighted")` or `sample_weights=True` use it. The loss is the weighted mean
`sum(w*l) / (sum(w) * 2)`, equal to the unweighted loss when all weights are 1.

**Gradient accumulation** is real here (the effective batch is `batch_size * grad_accumulation`).
The original `run_train` accepted `grad_accumulation=` through `**kwargs` and silently ignored it.

| `TrainConfig` field | Default | Notes |
|---|---|---|
| `epochs`, `batch_size`, `grad_accumulation` | 10, 32, 1 | |
| `device` | `auto` | `auto` (cuda > mps > cpu), `cuda`, `mps`, `cpu` |
| `compute` | `fp32` | `bf16` / `fp16` autocast for forward and backward; fp16 on CUDA uses a gradient scaler |
| `seed` | 42 | `None` = unseeded |
| `lr`, `optimizer`, `weight_decay` | 1e-3, `adam`, 0 | `adam` (eps 1e-9 as in the original), `adamw`, `sgd` |
| `lr_schedule`, `warmup_epochs` | `plateau`, 0 | `plateau` (x0.8 after an epoch without a new best), `cosine`, `onecycle`, `constant`, `legacy` (the original's real behavior: x0.8 after every epoch) |
| `loss`, `focal_gamma`, `sample_weights` | `soft_nll`, 2.0, False | `soft_nll`, `focal`, `weighted` |
| `label_smoothing`, `pos_neg_ratio` | 0.05, 1.0 | as in `run_train` |
| `grad_clip` | 5.0 | `0`/`None` disables |
| `early_stopping_patience` | `None` | epochs without a new best |
| `ema_decay` | `None` | validate and save `best.pth` with averaged weights |
| `select_metric` | `f1` | what picks `best.pth`: `f1` (at 0.5, like the original), `best_f1`, `pr_auc`, `roc_auc` |
| `token_dropout` | 0.0 | delete each input token with this probability while training |
| `freeze` | `[]` | parameter-name prefixes that stay fixed |
| `checkpoint_every`, `eval_batch_size` | 1, 512 | |

**Evaluation.** `evaluate(model, dataset, device, group_by=..., group_bin=...)` returns an
`EvalResult`: probabilities and labels, precision/recall/F1 at 0.5, a threshold sweep (0.01 to
0.99), `best_f1_threshold`, `threshold_for_precision(target)`, `threshold_for_recall(target)`, ROC
and PR AUC, confusion counts, and a per-group table (for example `group_by="jaro_winkler_similarity",
group_bin=0.1`, or a categorical column such as `country`). `agreement(probs_a, probs_b)`
reports the decision agreement rate, flips and the largest score difference between two models.

### Methods

`deepmatcher.methods.METHODS` is an annotated registry (`Method`: description, rationale,
reference, hyperparameters with defaults and ranges, cost note, `legacy`/`new` status) meant to be
rendered by user interfaces. The "new" entries are hypotheses to compare on a validation set, not
proven upgrades.

| Name | Kind | Status | Select with |
|---|---|---|---|
| `sif`, `rnn`, `attention`, `hybrid` | summarizer | legacy | `MatchingModel(attr_summarizer=...)` |
| `hybrid-transformer` | summarizer | new | Hybrid with `TransformerSelfAttention` + `MultiHeadCrossAttention` |
| `gru`, `lstm`, `self-attention` | contextualizer | legacy | `word_contextualizer=` |
| `transformer` | contextualizer | new | `word_contextualizer='transformer'` or `word_contextualizers.TransformerSelfAttention(heads=4, layers=1, ...)` |
| `decomposable-attention` | comparator | legacy | `word_comparator=` |
| `multi-head-attention` | comparator | new | `word_comparator='multi-head-attention'` or `word_comparators.MultiHeadCrossAttention(heads=4)` |
| `soft_nll` | loss | legacy | `TrainConfig(loss=...)` |
| `focal`, `weighted` | loss | new | `TrainConfig(loss=...)` |
| `plateau` | schedule | legacy | `TrainConfig(lr_schedule=...)` |
| `cosine`, `onecycle`, `constant` | schedule | new | `TrainConfig(lr_schedule=...)` |
| `token-dropout`, `ema` | regularizer | new | `TrainConfig(token_dropout=..., ema_decay=...)` |

The string `'self-attention'` keeps meaning the original DeepMatcher module (so existing code and
checkpoints do not change); the transformer encoder is `'transformer'`. A character n-gram
fallback was left out on purpose: the fastText vectors already carry subword information.

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
