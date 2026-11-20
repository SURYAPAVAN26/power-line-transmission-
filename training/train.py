"""
YOLOv8 Training Script for Insulator Defect Detection
Usage:
    python training/train.py --data training/dataset.yaml --model yolov8n.pt --epochs 100
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # project root


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model for insulator defect detection"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="training/dataset.yaml",
        help="Path to dataset YAML file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLOv8 model variant or path to pretrained weights",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (pixels)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (-1 for auto-batch)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="insulator_defect",
        help="Experiment name (saved under runs/detect/)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from last checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to run on: cpu, 0, 0,1,2,3 (empty = auto)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (epochs)",
    )
    parser.add_argument(
        "--augment-config",
        type=str,
        default="training/augment_config.yaml",
        help="Path to augmentation config YAML (merged into train kwargs)",
    )
    return parser.parse_args()


def load_augment_config(path: str) -> dict:
    """Load augmentation overrides from YAML, returning empty dict on failure."""
    try:
        import yaml  # PyYAML
        p = Path(path)
        if not p.exists():
            print(f"[warn] Augmentation config not found at {path}, using defaults.")
            return {}
        with open(p) as f:
            cfg = yaml.safe_load(f) or {}
        # Keep only keys recognised as Ultralytics hyperparameters
        hyp_keys = {
            "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
            "shear", "perspective", "flipud", "fliplr", "mosaic", "mixup",
            "copy_paste", "auto_augment", "erasing", "crop_fraction",
        }
        return {k: v for k, v in cfg.items() if k in hyp_keys}
    except Exception as e:
        print(f"[warn] Could not load augment config: {e}")
        return {}


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "ultralytics not installed. Run: pip install ultralytics"
        )

    import yaml

    # ── Resolve paths relative to project root ─────────────────────────────
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not data_path.exists():
        sys.exit(f"Dataset YAML not found: {data_path}")

    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Insulator Defect Detection — YOLOv8 Training")
    print(f"{'='*60}")
    print(f"  Data   : {data_path}")
    print(f"  Model  : {args.model}")
    print(f"  Epochs : {args.epochs}")
    print(f"  ImgSz  : {args.imgsz}")
    print(f"  Batch  : {args.batch}")
    print(f"  Name   : {args.name}")
    print(f"  Device : {args.device or 'auto'}")
    print(f"{'='*60}\n")

    model = YOLO(args.model)

    # ── Build train kwargs ─────────────────────────────────────────────────
    aug_overrides = load_augment_config(args.augment_config)

    train_kwargs = dict(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        resume=args.resume,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(ROOT / "runs" / "detect"),
        exist_ok=True,
        # Logging
        verbose=True,
        plots=True,
        save=True,
        save_period=10,          # checkpoint every 10 epochs
        # Validation
        val=True,
        # Optimiser
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # Augmentation defaults (may be overridden by augment_config)
        **aug_overrides,
    )

    # ── Train ──────────────────────────────────────────────────────────────
    results = model.train(**train_kwargs)

    # ── Copy best weights to models/ ───────────────────────────────────────
    run_dir = Path(results.save_dir)
    best_src = run_dir / "weights" / "best.pt"
    if best_src.exists():
        dest = models_dir / "best.pt"
        shutil.copy2(best_src, dest)
        print(f"\n[✓] Best weights copied to: {dest}")
    else:
        print(f"\n[!] best.pt not found at {best_src}; check run directory.")

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Training complete!")
    print(f"  Run dir  : {run_dir}")
    print(f"  Best     : {models_dir / 'best.pt'}")
    print(f"  mAP50    : {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  mAP50-95 : {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
