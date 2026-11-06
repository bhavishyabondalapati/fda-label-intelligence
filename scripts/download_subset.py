"""Download openFDA drug label bulk files and build a reproducible ~5K-label subset.

Outputs (all rebuilt from scratch by this script):
  data/raw/drug-label-XXXX-of-YYYY.json.zip   raw downloads (gitignored, cached)
  data/processed/labels.jsonl                  one line per label: metadata + split
  data/processed/sections.jsonl                one line per (label, section): the text
  data/processed/manifest.json                 seed, export date, file checksums, counts

How the sample stays reproducible: every eligible label gets a hash key from
(seed, set_id). We keep the N labels with the smallest keys. That is a uniform
random sample, identical on every run for the same openFDA export, and it never
needs more than one bulk file in memory at a time.

Usage:
  python scripts/download_subset.py                 # full run, ~1.8 GB download
  python scripts/download_subset.py --size 5000 --seed 42
"""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.ingest.openfda import is_human_rx, parse_label, sample_key  # noqa: E402

INDEX_URL = "https://api.fda.gov/download.json"


def fetch_partition_list():
    info = requests.get(INDEX_URL, timeout=60).json()["results"]["drug"]["label"]
    return info["export_date"], info["partitions"]


def download(url: str, dest: Path):
    """Download with a progress bar; skip if already complete."""
    head = requests.head(url, timeout=60, allow_redirects=True)
    size = int(head.headers.get("content-length", 0))
    if dest.exists() and size and dest.stat().st_size == size:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f, tqdm(total=size, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_results(zip_path: Path) -> list:
    """Each zip holds one JSON file shaped {"meta": ..., "results": [labels]}."""
    with zipfile.ZipFile(zip_path) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            return json.load(f)["results"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=5000, help="number of labels to keep")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    args = ap.parse_args()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    export_date, partitions = fetch_partition_list()
    print(f"openFDA export {export_date}: {len(partitions)} files")

    kept = {}          # set_id -> (key, label_meta, sections)
    checksums = {}
    n_seen = n_rx = 0
    for p in partitions:
        url = p["file"]
        zip_path = args.raw_dir / url.rsplit("/", 1)[-1]
        download(url, zip_path)
        checksums[zip_path.name] = sha256_of(zip_path)

        results = load_results(zip_path)
        n_seen += len(results)
        # Cheap pass first: compute hash keys for Rx labels, keep only the best `size`.
        cands = []
        for raw in results:
            if is_human_rx(raw) and raw.get("set_id") and raw["set_id"] not in kept:
                cands.append((sample_key(args.seed, raw["set_id"]), raw))
        n_rx += len(cands)
        cands.sort(key=lambda c: c[0])
        # Expensive pass (full parsing) only for the labels that could make the cut.
        for key, raw in cands[: args.size]:
            parsed = parse_label(raw, args.seed)
            if parsed:
                kept[raw["set_id"]] = (key, parsed[0], parsed[1])
        # Trim to the global best `size`.
        best = sorted(kept.items(), key=lambda kv: kv[1][0])[: args.size]
        kept = dict(best)
        del results, cands
        print(f"  {zip_path.name}: kept {len(kept)} (Rx seen so far: {n_rx})")

    ordered = sorted(kept.values(), key=lambda v: v[0])
    labels_path = args.out_dir / "labels.jsonl"
    sections_path = args.out_dir / "sections.jsonl"
    with open(labels_path, "w") as lf, open(sections_path, "w") as sf:
        for _, meta, sections in ordered:
            lf.write(json.dumps(meta) + "\n")
            for s in sections:
                sf.write(json.dumps(s) + "\n")

    splits = {}
    for _, meta, _ in ordered:
        splits[meta["split"]] = splits.get(meta["split"], 0) + 1
    manifest = {
        "openfda_export_date": export_date,
        "seed": args.seed,
        "target_size": args.size,
        "labels_total_in_export": n_seen,
        "human_rx_labels_seen": n_rx,
        "labels_kept": len(ordered),
        "sections_kept": sum(len(s) for _, _, s in ordered),
        "split_counts": splits,
        "raw_file_sha256": checksums,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
