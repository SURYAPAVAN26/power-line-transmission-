"""
master_merge.py
===============
Master script: validates, fixes, harmonises, and merges all insulator
defect datasets into one clean YOLO dataset.

Datasets handled
----------------
  1. CPLID           – Pascal VOC XMLs → YOLO (267 matched pairs)
  2. CPLID_YOLO      – existing YOLO conversion (restructures subfolder layout)
  3. FaultInsulator_Converted
  4. Insulator Defect.v1i.yolov8   (3-class → 2-class remap)
  5. InsulatorDefect_Converted
  6. fault-insulator.v1i.yolov8    (8-class → 2-class remap  ⚠ UPDATE CLASS_REMAP_8)
  7. Merged_Dataset  – re-validated

Output
------
  datasets/CPLID_Final/
    images/train|val|test/
    labels/train|val|test/
    dataset.yaml

Usage
-----
  python master_merge.py

Run from your project root where datasets/ lives.
"""

import os, sys, shutil, random, math, hashlib, csv
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG  — edit these before running
# ═══════════════════════════════════════════════════════════════════════════════

ROOT        = Path("datasets")
OUTPUT_DIR  = Path("datasets/CPLID_Final")

# Final unified class map
FINAL_CLASSES = {0: "defect", 1: "insulator"}

# ── Per-dataset class remaps ──────────────────────────────────────────────────
# Maps source_class_id → final_class_id.  None = drop that class.

# CPLID XML class names → final id
CPLID_NAME_MAP = {
    "defect"   : 0,
    "insulator": 1,
}

# CPLID_YOLO (class 0 = insulator, class 1 = defect based on label samples)
# Samples show class 0 has 656 boxes (insulator), class 1 has 123 (defect)
CPLID_YOLO_REMAP = {
    0: 1,   # was "insulator" → final 1
    1: 0,   # was "defect"    → final 0
}

# FaultInsulator_Converted: 2 classes, 0=defect 1=insulator
FAULT_CONV_REMAP = {0: 0, 1: 1}

# Insulator Defect.v1i.yolov8: 3 classes
# 0=broken → defect, 1=insulator → insulator, 2=pollution-flashover → defect
INSULATOR_DEFECT_REMAP = {
    0: 0,    # broken → defect
    1: 1,    # insulator → insulator
    2: 0,    # pollution-flashover → defect
}

# InsulatorDefect_Converted: 2 classes
INSULATOR_DEFECT_CONV_REMAP = {0: 0, 1: 1}

# fault-insulator.v1i.yolov8: 8 classes
# ⚠️  IMPORTANT: Check the YAML / README in that folder and fill in the mapping!
# Current placeholder: treat class 6 & 7 as insulator, all others as defect.
# Replace with accurate mapping once you know the class names.
FAULT_8CLASS_REMAP = {
    0: 0,    # unknown defect type → defect
    1: 0,    # unknown defect type → defect
    2: 0,    # unknown defect type → defect
    3: 0,    # unknown defect type → defect
    4: 0,    # unknown defect type → defect
    5: None, # unknown → DROP until confirmed (high count, likely insulator)
    6: 1,    # likely insulator string → insulator
    7: 1,    # likely insulator string → insulator
}

# Merged_Dataset: 2 classes, 0=defect 1=insulator
MERGED_REMAP = {0: 0, 1: 1}

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test = 1 - TRAIN - VAL = 0.15

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

RANDOM_SEED = 42

# ═══════════════════════════════════════════════════════════════════════════════


def file_hash(path: Path, chunk=65536) -> str:
    """MD5 of first chunk — fast enough for dedup."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(chunk))
    return h.hexdigest()


def find_images_in(folder: Path) -> dict[str, Path]:
    """stem.lower() → path for every image under folder."""
    result = {}
    for p in folder.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            result[p.stem.lower()] = p
    return result


def find_labels_in(folder: Path) -> dict[str, Path]:
    """stem.lower() → path for every .txt under folder."""
    return {p.stem.lower(): p for p in folder.rglob("*.txt")}


def parse_yolo_label(path: Path, remap: dict) -> list[str]:
    """Read YOLO label, apply class remap, return valid lines."""
    lines = []
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        final = remap.get(cls_id, None)
        if final is None:
            continue
        lines.append(f"{final} " + " ".join(parts[1:]))
    return lines


def parse_xml_to_yolo(xml_path: Path, name_map: dict, img_w: int = 0, img_h: int = 0):
    """
    Convert Pascal VOC XML to list of YOLO label strings.
    Returns (w, h, [lines]) or (0, 0, []) on failure.
    """
    try:
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        w = int(size.findtext("width", 0)) if size is not None else img_w
        h = int(size.findtext("height", 0)) if size is not None else img_h
        if w == 0 or h == 0:
            return 0, 0, []
        lines = []
        for obj in root.findall("object"):
            name = obj.findtext("name", "").strip().lower()
            final = name_map.get(name)
            if final is None:
                continue
            bb = obj.find("bndbox")
            if bb is None:
                continue
            xmin = float(bb.findtext("xmin", 0))
            ymin = float(bb.findtext("ymin", 0))
            xmax = float(bb.findtext("xmax", 0))
            ymax = float(bb.findtext("ymax", 0))
            xmin, xmax = max(0, xmin), min(w, xmax)
            ymin, ymax = max(0, ymin), min(h, ymax)
            if xmax <= xmin or ymax <= ymin:
                continue
            cx = ((xmin + xmax) / 2) / w
            cy = ((ymin + ymax) / 2) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            lines.append(f"{final} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        return w, h, lines
    except Exception as e:
        return 0, 0, []


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-dataset collectors
# ═══════════════════════════════════════════════════════════════════════════════

Sample = tuple  # (img_path, label_lines: list[str], source_tag: str)

def collect_cplid() -> list[Sample]:
    """CPLID: match XML annotations to images by stem (with zero-pad variants)."""
    base = ROOT / "CPLID"
    if not base.exists():
        print("  [SKIP] datasets/CPLID not found")
        return []

    all_imgs = find_images_in(base)

    def fuzzy(stem):
        for s in [stem, stem.lstrip("0") or stem, stem.zfill(3), stem.zfill(4)]:
            if s in all_imgs:
                return all_imgs[s]
        return None

    samples = []
    skipped = 0
    for xml_path in base.rglob("*.xml"):
        _, w, h, _ = 0, 0, 0, []
        w, h, lines = parse_xml_to_yolo(xml_path, CPLID_NAME_MAP)
        if not lines:
            skipped += 1
            continue
        img = fuzzy(xml_path.stem.lower())
        if img is None:
            skipped += 1
            continue
        samples.append((img, lines, "CPLID"))

    print(f"  CPLID          : {len(samples):4d} samples  ({skipped} skipped)")
    return samples


def collect_cplid_yolo() -> list[Sample]:
    """CPLID_YOLO: match within each subfolder (Defective / Normal)."""
    base = ROOT / "CPLID_YOLO"
    if not base.exists():
        print("  [SKIP] datasets/CPLID_YOLO not found")
        return []

    samples = []
    skipped = 0
    for sub in ["Defective_Insulators", "Normal_Insulators"]:
        img_dir = base / sub / "images"
        lbl_dir = base / sub / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            continue
        imgs = find_images_in(img_dir)
        lbls = find_labels_in(lbl_dir)
        for stem, img_path in imgs.items():
            if stem not in lbls:
                skipped += 1
                continue
            lines = parse_yolo_label(lbls[stem], CPLID_YOLO_REMAP)
            if not lines:
                skipped += 1
                continue
            samples.append((img_path, lines, "CPLID_YOLO"))

    print(f"  CPLID_YOLO     : {len(samples):4d} samples  ({skipped} skipped)")
    return samples


def collect_split_dataset(name: str, folder_name: str, remap: dict,
                           splits=("train", "valid", "val", "test")) -> list[Sample]:
    """Generic collector for datasets with train/valid/test split folders."""
    base = ROOT / folder_name
    if not base.exists():
        print(f"  [SKIP] datasets/{folder_name} not found")
        return []

    samples = []
    skipped = 0
    found_splits = [s for s in splits if (base / s).exists()]

    for split in found_splits:
        img_dir = base / split / "images"
        lbl_dir = base / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            continue
        imgs = find_images_in(img_dir)
        lbls = find_labels_in(lbl_dir)
        for stem, img_path in imgs.items():
            if stem not in lbls:
                skipped += 1
                continue
            lines = parse_yolo_label(lbls[stem], remap)
            if not lines:
                skipped += 1
                continue
            samples.append((img_path, lines, name))

    print(f"  {name:<22s}: {len(samples):4d} samples  ({skipped} skipped)")
    return samples


def collect_merged() -> list[Sample]:
    """Merged_Dataset: handles both split folders AND fasterrcnn_train_300."""
    base = ROOT / "Merged_Dataset"
    if not base.exists():
        print("  [SKIP] datasets/Merged_Dataset not found")
        return []

    samples = []
    skipped = 0
    # All sub-dirs that contain images/ and labels/ subdirs
    for sub in base.iterdir():
        if not sub.is_dir():
            continue
        img_dir = sub / "images"
        lbl_dir = sub / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            continue
        imgs = find_images_in(img_dir)
        lbls = find_labels_in(lbl_dir)
        for stem, img_path in imgs.items():
            if stem not in lbls:
                skipped += 1
                continue
            lines = parse_yolo_label(lbls[stem], MERGED_REMAP)
            if not lines:
                skipped += 1
                continue
            samples.append((img_path, lines, "Merged_Dataset"))

    print(f"  {'Merged_Dataset':<22s}: {len(samples):4d} samples  ({skipped} skipped)")
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
#  Deduplication
# ═══════════════════════════════════════════════════════════════════════════════

def deduplicate(samples: list[Sample]) -> list[Sample]:
    seen = {}
    unique = []
    dupes = 0
    for img_path, lines, src in samples:
        h = file_hash(img_path)
        if h not in seen:
            seen[h] = True
            unique.append((img_path, lines, src))
        else:
            dupes += 1
    print(f"\n  Dedup: {len(samples)} → {len(unique)} unique  ({dupes} duplicates removed)")
    return unique


# ═══════════════════════════════════════════════════════════════════════════════
#  Write output
# ═══════════════════════════════════════════════════════════════════════════════

def write_output(samples: list[Sample]):
    random.seed(RANDOM_SEED)
    random.shuffle(samples)

    n = len(samples)
    n_train = math.floor(n * TRAIN_RATIO)
    n_val   = math.floor(n * VAL_RATIO)
    splits = {
        "train": samples[:n_train],
        "val"  : samples[n_train : n_train + n_val],
        "test" : samples[n_train + n_val :],
    }

    for split in ("train", "val", "test"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    class_counts = defaultdict(int)
    source_counts = defaultdict(int)
    written = 0

    for split, items in splits.items():
        for img_path, lines, src in items:
            stem = img_path.stem
            dst_img = OUTPUT_DIR / "images" / split / (stem + img_path.suffix)
            # Avoid name collision by appending source tag
            if dst_img.exists():
                stem = f"{stem}_{src[:4]}"
                dst_img = OUTPUT_DIR / "images" / split / (stem + img_path.suffix)
            shutil.copy2(img_path, dst_img)
            (OUTPUT_DIR / "labels" / split / (stem + ".txt")).write_text("\n".join(lines))
            for line in lines:
                cls = int(line.split()[0])
                class_counts[cls] += 1
            source_counts[src] += 1
            written += 1

    # dataset.yaml
    nc = len(FINAL_CLASSES)
    names_yaml = "\n".join(f"  {k}: {v}" for k, v in sorted(FINAL_CLASSES.items()))
    yaml_txt = f"""# Merged Insulator Defect Dataset — YOLOv8/v11
# Generated by master_merge.py

path: {OUTPUT_DIR.resolve().as_posix()}
train: images/train
val:   images/val
test:  images/test

nc: {nc}
names:
{names_yaml}

# Sources merged:
{chr(10).join(f'#   {k}: {v}' for k, v in source_counts.items())}
"""
    (OUTPUT_DIR / "dataset.yaml").write_text(yaml_txt)

    # Audit CSV
    with open(OUTPUT_DIR / "split_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "image", "source"])
        for split, items in splits.items():
            for img_path, _, src in items:
                w.writerow([split, img_path.name, src])

    print(f"\n  Written: {written} samples")
    print(f"  train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")
    print(f"\n  Class distribution in output:")
    for cls_id, count in sorted(class_counts.items()):
        print(f"    Class {cls_id} ({FINAL_CLASSES[cls_id]}): {count} boxes")
    print(f"\n  Source breakdown:")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {src:<30s}: {count}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if not ROOT.exists():
        print(f"[ERROR] 'datasets/' folder not found in CWD: {Path.cwd()}")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("  MASTER MERGE PIPELINE — Insulator Defect Datasets")
    print("=" * 65)
    print(f"\nCollecting from {ROOT.resolve()}\n")

    all_samples: list[Sample] = []

    all_samples += collect_cplid()
    all_samples += collect_cplid_yolo()
    all_samples += collect_split_dataset(
        "FaultInsulator",  "FaultInsulator_Converted",  FAULT_CONV_REMAP)
    all_samples += collect_split_dataset(
        "InsulatorDefect_v1", "Insulator Defect.v1i.yolov8", INSULATOR_DEFECT_REMAP)
    all_samples += collect_split_dataset(
        "InsulatorDefect_Conv", "InsulatorDefect_Converted", INSULATOR_DEFECT_CONV_REMAP)
    all_samples += collect_split_dataset(
        "FaultIns_8cls", "fault-insulator.v1i.yolov8", FAULT_8CLASS_REMAP)
    all_samples += collect_merged()

    print(f"\nTotal before dedup: {len(all_samples)}")

    if not all_samples:
        print("[ERROR] No samples collected. Check ROOT path and dataset folders.")
        sys.exit(1)

    unique = deduplicate(all_samples)

    print(f"\nWriting output to {OUTPUT_DIR.resolve()} ...")
    write_output(unique)

    print(f"\n  dataset.yaml   → {OUTPUT_DIR / 'dataset.yaml'}")
    print(f"  split_manifest → {OUTPUT_DIR / 'split_manifest.csv'}")
    print("\n" + "=" * 65)
    print("  NEXT STEPS")
    print("=" * 65)
    print("""
  1. Inspect fault-insulator.v1i.yolov8 YAML for 8-class names, then
     update FAULT_8CLASS_REMAP in this script and re-run.

  2. Train:
     yolo train \\
       model=yolov8m.pt \\
       data=datasets/CPLID_Final/dataset.yaml \\
       epochs=150 imgsz=640 batch=16 \\
       cls=2.0  \\
       workers=4

  3. Validate:
     yolo val model=runs/detect/train/weights/best.pt \\
               data=datasets/CPLID_Final/dataset.yaml
""")


if __name__ == "__main__":
    main()
