"""Build the reproducible label subset from openFDA bulk files (shared by the script and the ingest service).

How the sample stays reproducible: every eligible label gets a hash key from (seed, set_id).
We keep the N labels with the smallest keys: a uniform random sample, identical on every
run for the same openFDA export, that never needs more than one bulk file in memory.
"""
import hashlib
import json
import zipfile
from pathlib import Path

from src.ingest.openfda import is_human_rx, parse_label, sample_key

INDEX_URL = "https://api.fda.gov/download.json"


def fetch_partition_list():
    import requests
    info = requests.get(INDEX_URL, timeout=60).json()["results"]["drug"]["label"]
    return info["export_date"], info["partitions"]


def download(url: str, dest: Path):
    """Download with a progress bar; skip if already complete."""
    import requests
    from tqdm import tqdm
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
        with z.open(z.namelist()[0]) as f:
            return json.load(f)["results"]


def build_subset(raw_dir: Path, out_dir: Path, size=5000, seed=42, fetch=True, log=print):
    """Write labels.jsonl, sections.jsonl and manifest.json into out_dir; return the manifest.

    fetch=True downloads the bulk files listed by openFDA into raw_dir (cached).
    fetch=False uses whatever *.json.zip files are already in raw_dir.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if fetch:
        export_date, partitions = fetch_partition_list()
        zips = []
        for p in partitions:
            zip_path = raw_dir / p["file"].rsplit("/", 1)[-1]
            download(p["file"], zip_path)
            zips.append(zip_path)
    else:
        export_date = "local-files"
        zips = sorted(raw_dir.glob("*.json.zip"))
    log(f"openFDA export {export_date}: {len(zips)} files")

    kept = {}          # set_id -> (key, label_meta, sections)
    checksums = {}
    n_seen = n_rx = 0
    for zip_path in zips:
        checksums[zip_path.name] = sha256_of(zip_path)
        results = load_results(zip_path)
        n_seen += len(results)
        # Cheap pass first: compute hash keys for Rx labels, keep only the best `size`.
        cands = []
        for raw in results:
            if is_human_rx(raw) and raw.get("set_id") and raw["set_id"] not in kept:
                cands.append((sample_key(seed, raw["set_id"]), raw))
        n_rx += len(cands)
        cands.sort(key=lambda c: c[0])
        # Expensive pass (full parsing) only for the labels that could make the cut.
        for key, raw in cands[:size]:
            parsed = parse_label(raw, seed)
            if parsed:
                kept[raw["set_id"]] = (key, parsed[0], parsed[1])
        kept = dict(sorted(kept.items(), key=lambda kv: kv[1][0])[:size])   # global best `size`
        del results, cands
        log(f"  {zip_path.name}: kept {len(kept)} (Rx seen so far: {n_rx})")

    ordered = sorted(kept.values(), key=lambda v: v[0])
    with open(out_dir / "labels.jsonl", "w") as lf, open(out_dir / "sections.jsonl", "w") as sf:
        for _, meta, sections in ordered:
            lf.write(json.dumps(meta) + "\n")
            for s in sections:
                sf.write(json.dumps(s) + "\n")

    splits = {}
    for _, meta, _ in ordered:
        splits[meta["split"]] = splits.get(meta["split"], 0) + 1
    manifest = {
        "openfda_export_date": export_date,
        "seed": seed,
        "target_size": size,
        "labels_total_in_export": n_seen,
        "human_rx_labels_seen": n_rx,
        "labels_kept": len(ordered),
        "sections_kept": sum(len(s) for _, _, s in ordered),
        "split_counts": splits,
        "raw_file_sha256": checksums,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
