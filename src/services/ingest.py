"""Ingest service: parse openFDA labels and build the label subset.

POST /ingest/label   one raw openFDA label JSON -> parsed metadata + sections
POST /ingest/subset  download (cached) + sample + parse the whole subset (takes minutes)
GET  /ingest/manifest  what the current subset contains
"""
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.ingest.openfda import parse_label
from src.ingest.subset import build_subset
from src.services.common import DATA_DIR, ROOT

app = FastAPI(title="ingest-service")


class LabelRequest(BaseModel):
    raw: dict
    seed: int = 42


class SubsetRequest(BaseModel):
    size: int = 5000
    seed: int = 42
    fetch: bool = True   # False = use only files already in data/raw


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest/label")
def ingest_label(req: LabelRequest):
    parsed = parse_label(req.raw, req.seed)
    if parsed is None:
        raise HTTPException(422, "label is not a usable human prescription label")
    meta, sections = parsed
    return {"label": meta, "sections": sections}


@app.post("/ingest/subset")
def ingest_subset(req: SubsetRequest):
    manifest = build_subset(Path(ROOT / "data" / "raw"), DATA_DIR, req.size, req.seed, fetch=req.fetch,
                            log=lambda *_: None)
    manifest.pop("raw_file_sha256", None)
    return manifest


@app.get("/ingest/manifest")
def manifest():
    path = DATA_DIR / "manifest.json"
    if not path.exists():
        raise HTTPException(404, "no subset built yet")
    m = json.loads(path.read_text())
    m.pop("raw_file_sha256", None)
    return m
