"""
data/verify_cplid_xml.py
========================
Standalone CPLID XML verification (original verify script,
now accepts --root argument for integration into the pipeline).
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="../datasets/CPLID",
                        help="Path to the CPLID dataset root")
    args = parser.parse_args()

    root = Path(args.root)
    print("\n" + "=" * 50)
    print("  CPLID XML VERIFICATION")
    print("=" * 50)

    # Collect all images
    images = {}
    for ext in IMG_EXTS:
        for p in root.rglob(f"*{ext}"):
            images[p.stem.lower()] = p

    # Collect all XMLs
    xmls = list(root.rglob("*.xml"))

    matched = []
    unmatched = []
    class_counts = defaultdict(int)
    category_counts = defaultdict(int)

    for xml_path in xmls:
        try:
            tree = ET.parse(xml_path)
            r = tree.getroot()
            fn = r.findtext("filename") or xml_path.stem
            stem = Path(fn).stem.lower()

            # fuzzy match
            img = None
            for s in [stem, stem.lstrip("0") or stem, stem.zfill(3), stem.zfill(4)]:
                if s in images:
                    img = images[s]
                    break

            # Determine category
            cat = "unknown"
            for part in xml_path.parts:
                if "defective" in part.lower():
                    cat = "Defective_Insulators"
                    break
                elif "normal" in part.lower():
                    cat = "Normal_Insulators"
                    break
            category_counts[cat] += 1

            # Count objects
            for obj in r.findall("object"):
                name = obj.findtext("name", "").strip().lower()
                class_counts[name] += 1

            if img:
                matched.append(xml_path)
            else:
                unmatched.append(xml_path)

        except Exception as e:
            unmatched.append(xml_path)

    print(f"\nTotal CPLID images   : {len(images)}")
    print(f"Total XML files      : {len(xmls)}")
    print(f"Matching XMLs        : {len(matched)}")
    print(f"Unmatched XMLs       : {len(unmatched)}")
    print(f"\nObject classes:")
    for name, cnt in sorted(class_counts.items()):
        print(f"  {name:20s}: {cnt}")
    print(f"\nAnnotations by category:")
    for cat, cnt in sorted(category_counts.items()):
        print(f"  {cat:30s}: {cnt}")
    if unmatched:
        print(f"\nSample unmatched XMLs:")
        for x in unmatched[:10]:
            print(f"  {x}")
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
