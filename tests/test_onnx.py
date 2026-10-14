import pytest
import torch
from retinatrust.models.build_model import build_model
from retinatrust.inference.onnx_export import export_to_onnx, benchmark_inference

def test_onnx_export_and_benchmark(tmp_path):
    model, _ = build_model("resnet18", pretrained=False)
    onnx_file = tmp_path / "test_model.onnx"

    exported_path = export_to_onnx(model, str(onnx_file), opset_version=18)
    assert exported_path.is_file()
    assert exported_path.stat().st_size > 1000

    bench = benchmark_inference(model, exported_path, num_runs=3)
    assert "onnx_latency_ms" in bench
    assert "speedup_ratio" in bench
    assert bench["speedup_ratio"] > 0.0
