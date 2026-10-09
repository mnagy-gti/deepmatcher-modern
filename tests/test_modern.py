"""Tests for what deepmatcher-modern adds: safetensors files, the inference
runtime (devices, reduced precision, torch.compile) and the porting fixes."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import pathname2url

import pytest
import torch

import deepmatcher as dm
from deepmatcher.data.field import FastText
from deepmatcher.data.process import process, process_unlabeled
from deepmatcher.runtime import (InferenceOptions, Predictor, QuantizedEmbedding, _fp8_lut,
                                 _quantize, available_devices, model_nbytes, select_device)
from deepmatcher.serialization import (SafetensorsFormatError, compare_models, convert,
                                       load_safetensors, read_header)

DATASETS = Path(__file__).parent / "upstream" / "test_datasets"
SUMMARIZERS = ["sif", "rnn", "attention", "hybrid"]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """One tiny trained checkpoint per attribute summarizer."""
    root = tmp_path_factory.mktemp("dm")
    data = root / "data"
    shutil.copytree(DATASETS, data)
    url_base = urljoin("file:", pathname2url(str(data))) + os.path.sep
    vectors = FastText("fasttext_sample.vec.zip", url_base=url_base, cache=str(root / "vec"))
    train, valid, test = process(
        path=str(data), cache="train_cache.pth", train="test_train.csv",
        validation="test_valid.csv", test="test_test.csv", embeddings=vectors,
        embeddings_cache_path="", ignore_columns=("left_id", "right_id"))
    models = {}
    for name in SUMMARIZERS:
        torch.manual_seed(0)
        path = root / f"{name}.pth"
        model = dm.MatchingModel(attr_summarizer=name)
        model.run_train(train, valid, epochs=1, batch_size=8, best_save_path=str(path),
                        pos_neg_ratio=3)
        models[name] = path
    return {"root": root, "models": models, "unlabeled": data / "test_unlabeled.csv",
            "test": test}


def _load_pth(path, summarizer):
    model = dm.MatchingModel(attr_summarizer=summarizer)
    model.load_state(str(path))
    return model


def _scores(model, unlabeled):
    dataset = process_unlabeled(str(unlabeled), trained_model=model,
                                ignore_columns=("left_id", "right_id"))
    return model.run_prediction(dataset)["match_score"]


# ---------------------------------------------------------------- porting
def test_runs_without_torchtext_installed():
    code = ("import sys; sys.modules['torchtext'] = None; import deepmatcher, "
            "deepmatcher.runtime, deepmatcher.serialization; print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=Path(__file__).parent.parent)
    assert out.stdout.strip() == "ok", out.stderr


def test_nan_check_switches_work():
    dm.disable_nan_checks()
    assert dm._check_nan is False
    dm.enable_nan_checks()
    assert dm._check_nan is True


# ------------------------------------------------------------ safetensors
@pytest.mark.parametrize("summarizer", SUMMARIZERS)
def test_safetensors_round_trip_is_exact(trained, summarizer, tmp_path):
    pth = trained["models"][summarizer]
    out = tmp_path / f"{summarizer}.safetensors"
    result = convert(str(pth), str(out), model_kwargs={"attr_summarizer": summarizer})
    assert result["verified"] is True and result["differences"] == []
    assert result["output_bytes"] < result["source_bytes"]  # no optimizer state

    original = _load_pth(pth, summarizer)
    restored = load_safetensors(out)
    assert compare_models(original, restored) == []
    a, b = _scores(original, trained["unlabeled"]), _scores(restored, trained["unlabeled"])
    assert a.index.equals(b.index) and (a.to_numpy() == b.to_numpy()).all()


def test_header_is_self_describing(trained, tmp_path):
    out = tmp_path / "m.safetensors"
    convert(str(trained["models"]["hybrid"]), str(out))
    header = read_header(out)
    assert header["format"] == "deepmatcher-safetensors" and header["format_version"] == "1"
    assert header["model_kwargs"] == {"attr_summarizer": "hybrid"}
    assert header["dropped"] == ["optimizer_state"]
    assert header["source"]["file"] == "hybrid.pth" and len(header["source"]["sha256"]) == 64


def test_rejects_foreign_safetensors(tmp_path):
    from safetensors.torch import save_file
    path = tmp_path / "other.safetensors"
    save_file({"w": torch.zeros(2)}, str(path))
    with pytest.raises(SafetensorsFormatError):
        read_header(path)


def test_compare_models_detects_a_changed_weight(trained):
    a = _load_pth(trained["models"]["sif"], "sif")
    b = _load_pth(trained["models"]["sif"], "sif")
    assert compare_models(a, b) == []
    name, tensor = next((k, v) for k, v in b.state_dict().items() if v.is_floating_point())
    tensor.view(-1)[0] += 1
    assert any(name in d for d in compare_models(a, b))


def test_convert_cli(trained, tmp_path):
    out = subprocess.run([sys.executable, "-m", "deepmatcher.convert",
                          str(trained["models"]["rnn"]), "--attr-summarizer", "rnn",
                          "--out-dir", str(tmp_path)], capture_output=True, text=True,
                         cwd=Path(__file__).parent.parent)
    assert out.returncode == 0, out.stderr
    assert "verified" in out.stdout and (tmp_path / "rnn.safetensors").exists()


# ---------------------------------------------------------------- runtime
def test_select_device():
    assert select_device("cpu").type == "cpu"
    assert select_device("auto").type == available_devices()[0]
    with pytest.raises(ValueError):
        select_device("tpu")
    with pytest.raises(ValueError):
        InferenceOptions(compute="int4")


def _predictor_scores(trained, summarizer="hybrid", **options):
    predictor = Predictor.load(str(trained["models"][summarizer]),
                               InferenceOptions(**{"device": "cpu", **options}),
                               attr_summarizer=summarizer)
    dataset = predictor.process_unlabeled(str(trained["unlabeled"]),
                                          ignore_columns=["left_id", "right_id"])
    return predictor.predict(dataset)["match_score"], predictor


@pytest.mark.parametrize("summarizer", SUMMARIZERS)
def test_reference_predictor_matches_run_prediction(trained, summarizer):
    reference = _scores(_load_pth(trained["models"][summarizer], summarizer), trained["unlabeled"])
    scores, predictor = _predictor_scores(trained, summarizer)
    assert predictor.options.is_reference
    assert (scores.to_numpy() == reference.loc[scores.index].to_numpy()).all()


@pytest.mark.parametrize("compute,weights,tolerance", [
    ("fp16", "fp32", 0.02), ("bf16", "fp32", 0.05),
    ("fp32", "fp16", 0.01), ("fp32", "bf16", 0.05), ("fp32", "fp8", 0.15), ("bf16", "fp8", 0.2),
])
def test_reduced_precision_stays_close(trained, compute, weights, tolerance):
    reference, ref_pred = _predictor_scores(trained)
    scores, predictor = _predictor_scores(trained, compute=compute, weights=weights)
    assert (scores - reference.loc[scores.index]).abs().max() < tolerance
    if weights != "fp32":
        assert predictor.stats["quantized_modules"] > 0
        assert predictor.stats["model_bytes"] < ref_pred.stats["model_bytes"]


def test_fp8_codec():
    lut = _fp8_lut()
    codes = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    finite = ~torch.isnan(codes.view(torch.float8_e4m3fn).float())
    assert torch.equal(lut[finite], codes.view(torch.float8_e4m3fn).float()[finite])
    w = torch.randn(64, 32) * torch.logspace(-3, 1, 64).unsqueeze(1)
    stored, scale = _quantize(w, "fp8")
    decoded = lut[stored.long()] * scale.unsqueeze(1)
    rel = ((decoded - w).abs() / w.abs().amax(dim=1, keepdim=True)).max()
    assert rel < 2 ** -4  # 3 mantissa bits, per-row scale
    emb = torch.nn.Embedding(64, 32)
    emb.weight.data.copy_(w)
    q = QuantizedEmbedding(emb, "fp8")
    idx = torch.tensor([[0, 5, 63]])
    assert torch.allclose(q(idx), decoded[idx])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple Metal")
def test_mps_matches_cpu(trained):
    cpu, _ = _predictor_scores(trained)
    mps, predictor = _predictor_scores(trained, device="mps")
    assert predictor.device.type == "mps"
    assert (mps - cpu.loc[mps.index]).abs().max() < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_cpu(trained):
    cpu, _ = _predictor_scores(trained)
    cuda, predictor = _predictor_scores(trained, device="cuda")
    assert predictor.device.type == "cuda"
    assert (cuda - cpu.loc[cuda.index]).abs().max() < 1e-5


@pytest.mark.slow
def test_torch_compile_matches_eager(trained):
    eager, _ = _predictor_scores(trained)
    compiled, _ = _predictor_scores(trained, compile=True)
    assert (compiled - eager.loc[compiled.index]).abs().max() < 1e-5


def test_model_nbytes_counts_shared_storage_once():
    a = torch.nn.Linear(10, 10)
    b = torch.nn.Module()
    b.x, b.y = a, a
    assert model_nbytes(b) == model_nbytes(a)
