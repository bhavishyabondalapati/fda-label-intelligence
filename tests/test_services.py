"""Service tests use FastAPI's TestClient with fakes: no model, no network, no Pinecone."""
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from src.services import chunk as chunk_svc
from src.services import embed as embed_svc
from src.services import ingest as ingest_svc
from src.services import query as query_svc
from src.store.chunk_store import ChunkStore

LONG = "word " * 40


# ---------- embed ----------
class FakeEmbedder:
    name, dim, device = "fake", 4, "cpu"

    def encode(self, texts):
        return np.array([[len(t), 1, 0, 0] for t in texts], dtype="float32")


def test_embed_service_returns_one_vector_per_text():
    with TestClient(embed_svc.create_app(FakeEmbedder())) as c:
        assert c.get("/health").json()["model"] == "fake"
        r = c.post("/embed", json={"texts": ["a", "abc"]}).json()
        assert r["dim"] == 4 and r["vectors"] == [[1, 1, 0, 0], [3, 1, 0, 0]]


def test_embed_service_rejects_empty_input():
    with TestClient(embed_svc.create_app(FakeEmbedder())) as c:
        assert c.post("/embed", json={"texts": []}).status_code == 422


# ---------- chunk ----------
def test_chunk_service_chunks_with_ids_offsets_and_primary_section():
    body = {"labels": [{"set_id": "S1", "drug_name": "Zed", "route": "ORAL",
                        "sections": [{"section": "a", "text": "A" * 1500}, {"section": "b", "text": "B" * 600}]}]}
    c = TestClient(chunk_svc.app)
    r = c.post("/chunk", json=body).json()
    chunks = r["chunks"]
    assert r["n_chunks"] == len(chunks) >= 3
    assert [x["chunk_id"] for x in chunks][:2] == ["S1:0", "S1:1"]
    assert chunks[0]["start"] == 0 and chunks[0]["section"] == "a"
    assert chunks[-1]["section"] == "b" and chunks[-1]["drug"] == "Zed" and chunks[-1]["route"] == "ORAL"


# ---------- ingest ----------
def make_raw(rx=True):
    raw = {"set_id": "abc", "openfda": {"brand_name": ["Zed"], "product_type": ["HUMAN PRESCRIPTION DRUG" if rx else "HUMAN OTC DRUG"]}}
    for s in ("indications_and_usage", "contraindications", "dosage_and_administration", "description"):
        raw[s] = [LONG]
    return raw


def test_ingest_label_parses_and_rejects():
    c = TestClient(ingest_svc.app)
    ok = c.post("/ingest/label", json={"raw": make_raw()}).json()
    assert ok["label"]["drug_name"] == "Zed" and len(ok["sections"]) == 4
    assert c.post("/ingest/label", json={"raw": make_raw(rx=False)}).status_code == 422


def test_ingest_subset_from_local_zip(tmp_path, monkeypatch):
    import json
    import zipfile
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    with zipfile.ZipFile(raw_dir / "drug-label-0001-of-0001.json.zip", "w") as z:
        z.writestr("x.json", json.dumps({"results": [make_raw(), make_raw(rx=False)]}))
    from src.ingest.subset import build_subset
    m = build_subset(raw_dir, tmp_path / "out", size=10, seed=1, fetch=False, log=lambda *_: None)
    assert m["labels_kept"] == 1 and m["human_rx_labels_seen"] == 1
    assert (tmp_path / "out" / "sections.jsonl").read_text().count("\n") == 4


# ---------- query ----------
class FakeIndex:
    def __init__(self):
        self.calls = []

    def query(self, vector, top_k, include_metadata, namespace):
        self.calls.append((vector, top_k, namespace))
        return SimpleNamespace(matches=[SimpleNamespace(id="S1:1", score=0.9), SimpleNamespace(id="S1:0", score=0.8),
                                        SimpleNamespace(id="GONE:0", score=0.7)])


def test_query_service_embeds_searches_and_looks_up_text(tmp_path):
    store = ChunkStore(tmp_path / "c.sqlite")
    store.create()
    store.add_many([{"chunk_id": f"S1:{i}", "set_id": "S1", "drug": "Zed", "route": "ORAL", "section": "a",
                     "start": i * 10, "end": i * 10 + 10, "text": f"text{i}"} for i in range(2)])
    index = FakeIndex()
    app = query_svc.create_app(embed_fn=lambda t: [1.0, 0.0], index=index, store=store, namespace="ns")
    with TestClient(app) as c:
        r = c.post("/query", json={"question": "what?", "top_k": 3}).json()
    assert [x["chunk_id"] for x in r["results"]] == ["S1:1", "S1:0"]       # order kept; id missing locally is dropped
    assert r["results"][0]["text"] == "text1" and r["results"][0]["start"] == 10
    assert set(r["timings_ms"]) == {"embed", "pinecone", "lookup", "total"}
    assert index.calls == [([1.0, 0.0], 3, "ns")]


def test_query_service_validates_input(tmp_path):
    app = query_svc.create_app(embed_fn=lambda t: [0.0], index=FakeIndex(), store=ChunkStore(tmp_path / "c.sqlite"))
    with TestClient(app) as c:
        assert c.post("/query", json={"question": ""}).status_code == 422
        assert c.post("/query", json={"question": "x", "top_k": 0}).status_code == 422


# ---------- chunk store ----------
def test_chunk_store_roundtrip_and_order(tmp_path):
    store = ChunkStore(tmp_path / "c.sqlite")
    store.create()
    store.add_many([{"chunk_id": f"S:{i}", "set_id": "S", "drug": "D", "route": "", "section": "a",
                     "start": 0, "end": 1, "text": f"t{i}"} for i in (2, 0, 1)])
    assert store.count() == 3
    assert [cid for cid, _ in store.iter_in_order()] == ["S:2", "S:0", "S:1"]    # insertion order preserved
    assert store.get_many(["S:0", "nope"]).keys() == {"S:0"}
