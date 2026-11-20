"""
data/validate_final.py
======================
Validates the final merged YOLO dataset before training.
Checks:
  - Image/label count per split
  - Every image has a matching label
  - All bounding boxes are valid (0-1 range, no zero-area boxes)
  - Class distribution
  - No corrupt images (via PIL open check)
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def validate_dataset(dataset_path: Path):
    print(f"\n{'='*60}")
    print(f"  VALIDATING: {dataset_path.resolve()}")
    print(f"{'='*60}")

    if not dataset_path.exists():
        print(f"[ERROR] Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    total_images = 0
    total_labels = 0
    total_boxes = 0
    invalid_boxes = 0
    class_counts = defaultdict(int)
    missing_labels = []
    missing_images = []
    corrupt_images = []
    errors = []

    for split in ("train", "val", "test"):
        img_dir = dataset_path / "images" / split
        lbl_dir = dataset_path / "labels" / split

        if not img_dir.exists():
            print(f"  [WARN] Split '{split}' has no images/ folder — skipping")
            continue

        images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
        labels = {p.stem: p for p in lbl_dir.iterdir() if p.suffix == ".txt"} if lbl_dir.exists() else {}

        split_missing_lbl = [img_dir / (s + ".jpg") for s in images if s not in labels]
        split_missing_img = [lbl_dir / (s + ".txt") for s in labels if s not in images]

        # Validate label contents
        split_boxes = 0
        split_invalid = 0
        for stem, lbl_path in labels.items():
            for line in lbl_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"Malformed line in {lbl_path}: {line!r}")
                    continue
                try:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                except ValueError:
                    errors.append(f"Non-numeric values in {lbl_path}: {line!r}")
                    continue
                split_boxes += 1
                class_counts[cls_id] += 1
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                    split_invalid += 1
                    errors.append(f"Out-of-range bbox in {lbl_path.name}: {line!r}")

        # Quick image corruption check
        split_corrupt = []
        try:
            from PIL import Image
            for img_path in list(images.values())[:20]:  # check first 20 per split
                try:
                    with Image.open(img_path) as im:
                        im.verify()
                except Exception as e:
                    split_corrupt.append(img_path.name)
        except ImportError:
            pass

        total_images += len(images)
        total_labels += len(labels)
        total_boxes += split_boxes
        invalid_boxes += split_invalid
        missing_labels.extend(split_missing_lbl)
        missing_images.extend(split_missing_img)
        corrupt_images.extend(split_corrupt)

        matched = len(images) - len(split_missing_lbl)
        status = "✅" if not split_missing_lbl and not split_invalid else "⚠️"
        print(f"\n  [{split.upper()}] {status}")
        print(f"    Images  : {len(images):5d}   Labels : {len(labels):5d}   Matched: {matched:5d}")
        print(f"    Boxes   : {split_boxes:5d}   Invalid: {split_invalid:5d}")
        if split_missing_lbl:
            print(f"    ⚠ {len(split_missing_lbl)} images without labels")
        if split_missing_img:
            print(f"    ⚠ {len(split_missing_img)} labels without images (orphans)")
        if split_corrupt:
            print(f"    ⚠ {len(split_corrupt)} possibly corrupt images")

    print(f"\n  {'─'*50}")
    print(f"  TOTALS")
    print(f"    Images       : {total_images}")
    print(f"    Labels       : {total_labels}")
    print(f"    Bounding boxes: {total_boxes}")
    print(f"    Invalid boxes: {invalid_boxes}")
    print(f"\n  CLASS DISTRIBUTION:")
    total_b = sum(class_counts.values()) or 1
    class_names = {0: "defect", 1: "insulator"}
    for cls_id, cnt in sorted(class_counts.items()):
        name = class_names.get(cls_id, f"class_{cls_id}")
        bar = "█" * int(30 * cnt / total_b)
        print(f"    {cls_id} ({name:12s}): {cnt:5d}  {bar}")

    if errors:
        print(f"\n  ⚠ ERRORS FOUND ({len(errors)}):")
        for e in errors[:10]:
            print(f"    {e}")
        if len(errors) > 10:
            print(f"    ... and {len(errors)-10} more")

    if missing_labels:
        print(f"\n  ⚠ {len(missing_labels)} images without labels (may be background negatives)")

    ok = invalid_boxes == 0 and len(corrupt_images) == 0 and len(errors) == 0
    print(f"\n  Overall: {'✅ VALID' if ok else '⚠️  ISSUES FOUND — review before training'}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="../datasets/CPLID_Final",
                        help="Path to the final merged YOLO dataset")
    args = parser.parse_args()
    ok = validate_dataset(Path(args.dataset))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
