"""
Evaluation Script — Insulator Defect Detection
Loads a trained YOLOv8 model, runs validation on the test split,
prints mAP50 / mAP50-95, and saves a confusion matrix + PR curve.

Usage:
    python training/evaluate.py --weights models/best.pt \
                                 --data training/dataset.yaml \
                                 --split test
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # project root


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv8 model for insulator defect detection"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/best.pt",
        help="Path to model weights (.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="training/dataset.yaml",
        help="Path to dataset YAML",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold (low for full PR curve)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="IoU threshold for NMS",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device: cpu | 0 | 0,1 …",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/evaluate",
        help="Directory to save evaluation artefacts",
    )
    return parser.parse_args()


def print_metrics(results):
    """Pretty-print key metrics from an Ultralytics val Results object."""
    rd = results.results_dict
    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    print(f"  {'Metric':<25} {'Value':>10}")
    print("-" * 50)

    metric_map = {
        "metrics/precision(B)":    "Precision",
        "metrics/recall(B)":       "Recall",
        "metrics/mAP50(B)":        "mAP@0.50",
        "metrics/mAP50-95(B)":     "mAP@0.50:0.95",
        "fitness":                 "Fitness",
    }
    for key, label in metric_map.items():
        val = rd.get(key, None)
        if val is not None:
            print(f"  {label:<25} {val:>10.4f}")
    print("=" * 50 + "\n")


def save_confusion_matrix(results, output_dir: Path, class_names: list):
    """Save a styled confusion matrix PNG using seaborn."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = results.confusion_matrix.matrix  # (nc+1, nc+1) numpy array
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".0f",
            cmap="Blues",
            xticklabels=class_names + ["background"],
            yticklabels=class_names + ["background"],
            ax=ax,
        )
        ax.set_title("Confusion Matrix — Insulator Defect Detection", fontsize=13)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        plt.tight_layout()
        dest = output_dir / "confusion_matrix.png"
        fig.savefig(dest, dpi=150)
        plt.close(fig)
        print(f"[✓] Confusion matrix saved → {dest}")
    except Exception as e:
        print(f"[warn] Could not save confusion matrix: {e}")


def save_pr_curve(results, output_dir: Path, class_names: list):
    """Save per-class Precision-Recall curves."""
    try:
        import matplotlib.pyplot as plt

        # Ultralytics stores curves on the validator
        validator = getattr(results, "validator", None)
        if validator is None:
            print("[warn] Validator not accessible; skipping PR curve.")
            return

        # results.curves_results contains (px, py, ap) per class
        px = validator.metrics.curves_results[0] if hasattr(validator.metrics, "curves_results") else None
        if px is None:
            print("[warn] PR curve data not available from this Ultralytics version.")
            return

        fig, ax = plt.subplots(figsize=(9, 6))
        colors = ["#e74c3c", "#2ecc71"]
        for i, name in enumerate(class_names):
            x = validator.metrics.curves_results[0]  # recall
            y = validator.metrics.curves_results[1][i]  # precision for class i
            ap = validator.metrics.curves_results[2][i]
            ax.plot(x, y, color=colors[i % len(colors)], label=f"{name} (AP={ap:.3f})")

        ax.set_xlabel("Recall", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_title("Precision-Recall Curve — Insulator Defect Detection", fontsize=13)
        ax.legend(loc="lower left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        dest = output_dir / "pr_curve.png"
        fig.savefig(dest, dpi=150)
        plt.close(fig)
        print(f"[✓] PR curve saved → {dest}")
    except Exception as e:
        print(f"[warn] Could not save PR curve: {e}")


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

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not data_path.exists():
        sys.exit(f"Dataset YAML not found: {data_path}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Class names from YAML ───────────────────────────────────────────────
    import yaml
    with open(data_path) as f:
        ds_cfg = yaml.safe_load(f)
    class_names = list(ds_cfg.get("names", {0: "defect", 1: "insulator"}).values())

    # ── Load & validate ─────────────────────────────────────────────────────
    print(f"\nLoading weights: {weights}")
    model = YOLO(str(weights))

    print(f"Running validation on [{args.split}] split …")
    results = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=str(output_dir),
        name="val_run",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    # ── Print metrics ───────────────────────────────────────────────────────
    print_metrics(results)

    # ── Save artefacts ──────────────────────────────────────────────────────
    save_confusion_matrix(results, output_dir, class_names)
    save_pr_curve(results, output_dir, class_names)

    print(f"\nAll artefacts saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
