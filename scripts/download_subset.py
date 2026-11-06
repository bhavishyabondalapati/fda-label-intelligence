"""Download openFDA drug label bulk files and build a reproducible ~5K-label subset.

Outputs (all rebuilt from scratch by this script):
  data/raw/drug-label-XXXX-of-YYYY.json.zip   raw downloads (gitignored, cached)
  data/processed/labels.jsonl                  one line per label: metadata + split
  data/processed/sections.jsonl                one line per (label, section): the text
  data/processed/manifest.json                 seed, export date, file checksums, counts

The logic lives in src/ingest/subset.py (also used by the ingest service).

Usage:
  python scripts/download_subset.py                 # full run, ~1.8 GB download
  python scripts/download_subset.py --size 5000 --seed 42
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.ingest.subset import build_subset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=5000, help="number of labels to keep")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--no-fetch", action="store_true", help="use files already in raw-dir")
    args = ap.parse_args()
    manifest = build_subset(args.raw_dir, args.out_dir, args.size, args.seed, fetch=not args.no_fetch)
    manifest.pop("raw_file_sha256")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
