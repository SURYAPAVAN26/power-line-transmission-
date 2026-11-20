"""
Export Script — Insulator Defect Detection
Exports a trained YOLOv8 model to ONNX and TorchScript,
then benchmarks inference speed for each format.

Usage:
    python training/export.py --weights models/best.pt --imgsz 640
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 model to ONNX and TorchScript"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/best.pt",
        help="Path to trained .pt weights",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Export image size",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Export batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for benchmarking: cpu | 0",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=False,
        help="Use FP16 half-precision for ONNX/TorchScript (GPU only)",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=100,
        help="Number of inference runs for speed benchmark",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save exported models",
    )
    return parser.parse_args()


def benchmark_pt(model, imgsz: int, device: str, n_runs: int) -> float:
    """Benchmark PyTorch .pt model speed, return mean ms per image."""
    import torch
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    # warmup
    for _ in range(5):
        model(dummy, verbose=False)
    start = time.perf_counter()
    for _ in range(n_runs):
        model(dummy, verbose=False)
    elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000


def benchmark_onnx(onnx_path: Path, imgsz: int, n_runs: int) -> float:
    """Benchmark ONNX model speed, return mean ms per image."""
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        inp_name = sess.get_inputs()[0].name
        dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
        # warmup
        for _ in range(5):
            sess.run(None, {inp_name: dummy})
        start = time.perf_counter()
        for _ in range(n_runs):
            sess.run(None, {inp_name: dummy})
        elapsed = time.perf_counter() - start
        return (elapsed / n_runs) * 1000
    except ImportError:
        print("[warn] onnxruntime not installed; skipping ONNX benchmark.")
        return float("nan")


def benchmark_torchscript(ts_path: Path, imgsz: int, n_runs: int) -> float:
    """Benchmark TorchScript model speed, return mean ms per image."""
    try:
        import torch

        model = torch.jit.load(str(ts_path), map_location="cpu")
        model.eval()
        dummy = torch.zeros(1, 3, imgsz, imgsz)
        # warmup
        with torch.no_grad():
            for _ in range(5):
                model(dummy)
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                model(dummy)
        elapsed = time.perf_counter() - start
        return (elapsed / n_runs) * 1000
    except Exception as e:
        print(f"[warn] TorchScript benchmark failed: {e}")
        return float("nan")


def print_benchmark_table(results: dict):
    print("\n" + "=" * 55)
    print("  EXPORT BENCHMARK RESULTS")
    print("=" * 55)
    print(f"  {'Format':<20} {'Latency (ms/img)':>18} {'FPS':>10}")
    print("-" * 55)
    for fmt, ms in results.items():
        fps = 1000 / ms if ms and not (ms != ms) else float("nan")  # nan check
        fps_str = f"{fps:.1f}" if fps == fps else "N/A"
        ms_str = f"{ms:.2f}" if ms == ms else "N/A"
        print(f"  {fmt:<20} {ms_str:>18} {fps_str:>10}")
    print("=" * 55 + "\n")


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed. Run: pip install ultralytics")

    # ── Resolve paths ───────────────────────────────────────────────────────
    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = ROOT / weights
    if not weights.exists():
        sys.exit(f"Weights not found: {weights}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    benchmark_results = {}

    # ── Benchmark base .pt ──────────────────────────────────────────────────
    print("\n[1/3] Benchmarking PyTorch .pt …")
    pt_ms = benchmark_pt(model, args.imgsz, args.device, args.benchmark_runs)
    benchmark_results["PyTorch (.pt)"] = pt_ms
    print(f"      → {pt_ms:.2f} ms/img  ({1000/pt_ms:.1f} FPS)")

    # ── Export ONNX ─────────────────────────────────────────────────────────
    print("\n[2/3] Exporting to ONNX …")
    try:
        onnx_path = model.export(
            format="onnx",
            imgsz=args.imgsz,
            batch=args.batch,
            half=args.half,
            dynamic=True,
            simplify=True,
            opset=17,
        )
        onnx_path = Path(onnx_path)
        # Copy to output_dir
        dest_onnx = output_dir / onnx_path.name
        if onnx_path != dest_onnx:
            import shutil
            shutil.copy2(onnx_path, dest_onnx)
        print(f"      → Saved: {dest_onnx}")
        onnx_ms = benchmark_onnx(dest_onnx, args.imgsz, args.benchmark_runs)
        benchmark_results["ONNX"] = onnx_ms
        if onnx_ms == onnx_ms:
            print(f"      → {onnx_ms:.2f} ms/img  ({1000/onnx_ms:.1f} FPS)")
    except Exception as e:
        print(f"[error] ONNX export failed: {e}")
        benchmark_results["ONNX"] = float("nan")

    # ── Export TorchScript ──────────────────────────────────────────────────
    print("\n[3/3] Exporting to TorchScript …")
    try:
        ts_path = model.export(
            format="torchscript",
            imgsz=args.imgsz,
            batch=args.batch,
            half=args.half,
        )
        ts_path = Path(ts_path)
        dest_ts = output_dir / ts_path.name
        if ts_path != dest_ts:
            import shutil
            shutil.copy2(ts_path, dest_ts)
        print(f"      → Saved: {dest_ts}")
        ts_ms = benchmark_torchscript(dest_ts, args.imgsz, args.benchmark_runs)
        benchmark_results["TorchScript"] = ts_ms
        if ts_ms == ts_ms:
            print(f"      → {ts_ms:.2f} ms/img  ({1000/ts_ms:.1f} FPS)")
    except Exception as e:
        print(f"[error] TorchScript export failed: {e}")
        benchmark_results["TorchScript"] = float("nan")

    # ── Summary ─────────────────────────────────────────────────────────────
    print_benchmark_table(benchmark_results)
    print(f"Exported models saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
