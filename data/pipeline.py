# Insulator Defect Detector — Data Pipeline
# Copies the scripts built during dataset analysis into the project.

"""
data/pipeline.py
================
Orchestrates the full data preparation pipeline:
  Step 1: Verify CPLID XML annotations
  Step 2: Convert CPLID XMLs → YOLO format
  Step 3: Merge all datasets into CPLID_Final/
  Step 4: Validate final dataset

Usage (from project root):
  python data/pipeline.py --datasets-root ../datasets --output ../datasets/CPLID_Final
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd=None):
    print(f"\n▶ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Insulator defect data pipeline")
    parser.add_argument("--datasets-root", default="../datasets",
                        help="Path to datasets/ folder (default: ../datasets)")
    parser.add_argument("--output", default="../datasets/CPLID_Final",
                        help="Output path for merged dataset (default: ../datasets/CPLID_Final)")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip XML verification step")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip merge step (use existing CPLID_Final)")
    args = parser.parse_args()

    root = Path(args.datasets_root)
    output = Path(args.output)
    scripts = Path(__file__).parent

    print("=" * 60)
    print("  INSULATOR DEFECT — DATA PIPELINE")
    print("=" * 60)
    print(f"  Datasets root : {root.resolve()}")
    print(f"  Output        : {output.resolve()}")

    if not args.skip_verify:
        print("\n[Step 1/3] Verifying CPLID XML annotations...")
        run([sys.executable, str(scripts / "verify_cplid_xml.py"),
             "--root", str(root / "CPLID")])

    if not args.skip_merge:
        print("\n[Step 2/3] Running master merge & conversion...")
        run([sys.executable, str(scripts / "master_merge.py"),
             "--datasets-root", str(root),
             "--output", str(output)])

    print("\n[Step 3/3] Validating final dataset...")
    run([sys.executable, str(scripts / "validate_final.py"),
         "--dataset", str(output)])

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Dataset ready at: {output.resolve()}")
    print("  Next step: python training/train.py --data training/dataset.yaml\n")


if __name__ == "__main__":
    main()
