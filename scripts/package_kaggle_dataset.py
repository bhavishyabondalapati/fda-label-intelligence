"""Package the Phase 4 training files for a PRIVATE Kaggle dataset. Local, free, uploads nothing.

Builds kaggle/dataset/ (train.jsonl, val.jsonl, train.py, dataset-metadata.json) and kaggle/fda-extraction-train.zip,
after checking that the files contain only extraction_train labels and no secret from .env.
"""
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval.score import read_jsonl  # noqa: E402

OUT = ROOT / "kaggle" / "dataset"
SLUG = "fda-extraction-train"


def env_secrets():
    """Values in .env that look like secrets (never printed)."""
    p = ROOT / ".env"
    vals = []
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                v = line.split("=", 1)[1].strip().strip("'\"")
                if len(v) >= 8:
                    vals.append(v)
    return vals


def main():
    labels = {l["set_id"]: l["split"] for l in read_jsonl(ROOT / "data" / "processed" / "labels.jsonl")}
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("train.jsonl", "val.jsonl"):
        shutil.copy(ROOT / "data" / "finetune" / f, OUT / f)
    shutil.copy(ROOT / "sagemaker" / "train.py", OUT / "train.py")
    (OUT / "dataset-metadata.json").write_text(json.dumps({
        "title": SLUG, "id": f"bhavishya12345678/{SLUG}", "licenses": [{"name": "CC0-1.0"}]}, indent=2))

    # checks
    splits, n = set(), {}
    for f in ("train.jsonl", "val.jsonl"):
        rows = list(read_jsonl(OUT / f))
        n[f] = len(rows)
        splits |= {labels[r["set_id"]] for r in rows}
    assert splits == {"extraction_train"}, f"unexpected splits in the data: {splits}"
    secrets = env_secrets()
    blob_hits = 0
    for p in OUT.iterdir():
        text = p.read_text()
        blob_hits += sum(s in text for s in secrets)
        assert not re.search(r"\b(AKIA|ASIA)[A-Z0-9]{16}\b", text), f"AWS key pattern in {p.name}"
        for marker in ("PINECONE_API_KEY", "aws_secret", "BEGIN PRIVATE KEY"):
            assert marker not in text, f"{marker} found in {p.name}"
    assert blob_hits == 0, "a value from .env appears in the dataset files"

    zpath = ROOT / "kaggle" / f"{SLUG}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            z.write(p, p.name)
    print(json.dumps({"rows": n, "splits": sorted(splits), "env_values_checked": len(secrets), "zip": str(zpath),
                      "zip_mb": round(zpath.stat().st_size / 1e6, 1)}, indent=2))


if __name__ == "__main__":
    main()
