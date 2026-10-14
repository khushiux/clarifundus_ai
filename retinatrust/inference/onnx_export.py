from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from retinatrust.config import resolve_path
from retinatrust.models.build_model import build_model
from retinatrust.utils.logging_utils import setup_logger

logger = setup_logger("retinatrust.onnx_export")

def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_size: Tuple[int, int] = (224, 224),
    opset_version: int = 17,
) -> Path:
    """Exports a PyTorch model to ONNX format with dynamic batch axis."""
    out_file = resolve_path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1], dtype=torch.float32)

    logger.info(f"Exporting PyTorch graph to ONNX: {out_file}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(out_file),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False,
    )

    # Validate ONNX graph integrity
    onnx_model = onnx.load(str(out_file))
    onnx.checker.check_model(onnx_model)
    logger.info(f"ONNX export successful and verified: {out_file}")
    return out_file

def benchmark_inference(
    pytorch_model: nn.Module,
    onnx_path: Path,
    input_size: Tuple[int, int] = (224, 224),
    num_runs: int = 30,
) -> Dict[str, Any]:
    """Benchmarks inference latency between PyTorch CPU and ONNX Runtime."""
    pytorch_model.eval()
    dummy_torch = torch.randn(1, 3, input_size[0], input_size[1], dtype=torch.float32)
    dummy_numpy = dummy_torch.numpy()

    # PyTorch CPU Warmup & Benchmark
    with torch.no_grad():
        for _ in range(5):
            _ = pytorch_model(dummy_torch)

        torch_times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = pytorch_model(dummy_torch)
            torch_times.append(time.perf_counter() - t0)

    avg_torch_ms = float(np.mean(torch_times) * 1000.0)

    # ONNX Runtime Benchmark
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    for _ in range(5):
        _ = session.run(None, {input_name: dummy_numpy})

    onnx_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_numpy})
        onnx_times.append(time.perf_counter() - t0)

    avg_onnx_ms = float(np.mean(onnx_times) * 1000.0)
    speedup = float(avg_torch_ms / max(avg_onnx_ms, 1e-4))

    logger.info(f"Benchmark Results (over {num_runs} iterations):")
    logger.info(f"PyTorch CPU:    {avg_torch_ms:.2f} ms")
    logger.info(f"ONNX Runtime:   {avg_onnx_ms:.2f} ms (Speedup: {speedup:.2f}x)")

    return {
        "pytorch_latency_ms": round(avg_torch_ms, 2),
        "onnx_latency_ms": round(avg_onnx_ms, 2),
        "speedup_ratio": round(speedup, 2),
        "onnx_file": str(onnx_path),
    }
