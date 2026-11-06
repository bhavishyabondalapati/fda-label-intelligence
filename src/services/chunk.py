"""Chunk service: POST /chunk {"labels": [{set_id, drug_name, route, sections:[{section,text}]}]}
-> naive fixed-size chunks (same chunker as the Phase 0 baseline)."""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.chunk.naive import CHUNK_SIZE, OVERLAP
from src.chunk.records import chunk_label


class Section(BaseModel):
    section: str
    text: str


class LabelIn(BaseModel):
    set_id: str
    drug_name: str
    route: str = ""
    sections: list[Section]


class ChunkRequest(BaseModel):
    labels: list[LabelIn] = Field(min_length=1, max_length=500)


app = FastAPI(title="chunk-service")


@app.get("/health")
def health():
    return {"status": "ok", "chunk_size": CHUNK_SIZE, "overlap": OVERLAP}


@app.post("/chunk")
def chunk(req: ChunkRequest):
    out = []
    for lab in req.labels:
        out.extend(chunk_label(lab.set_id, lab.drug_name, lab.route, [s.model_dump() for s in lab.sections]))
    return {"n_chunks": len(out), "chunks": out}
