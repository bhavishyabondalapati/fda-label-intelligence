"""Query service: POST /query {"question": "...", "top_k": 10}

    question -> embed service (HTTP) -> Pinecone search -> chunk text lookup (SQLite)

Returns the ranked chunks plus a timing breakdown in milliseconds.
"""
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.services import common


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=100)


def create_app(embed_fn=None, index=None, store=None, namespace=None) -> FastAPI:
    """Dependencies can be injected for tests:
      embed_fn(text) -> list[float]     index.query(...) like Pinecone     store.get_many(ids) like ChunkStore
    """
    state = {"embed_fn": embed_fn, "index": index, "store": store, "http": None}
    ns = namespace if namespace is not None else common.PINECONE_NAMESPACE

    def get_index():
        if state["index"] is None:                     # connect lazily if startup could not
            from src.retrieve.pinecone_store import get_client, open_index
            state["index"] = open_index(get_client(), common.PINECONE_INDEX)
        return state["index"]

    @asynccontextmanager
    async def lifespan(app):
        if state["embed_fn"] is None:
            # One persistent HTTP connection to the embed service (no reconnect per query).
            state["http"] = httpx.Client(base_url=common.EMBED_URL, timeout=30)
            state["embed_fn"] = lambda text: state["http"].post("/embed", json={"texts": [text]}).raise_for_status().json()["vectors"][0]
        if state["store"] is None:
            from src.store.chunk_store import ChunkStore
            state["store"] = ChunkStore(common.CHUNK_DB)
        try:
            get_index()                                # open the Pinecone connection before the first query
        except Exception:
            pass                                       # reported by /health; retried on first query
        yield
        if state["http"] is not None:
            state["http"].close()

    app = FastAPI(title="query-service", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "pinecone_connected": state["index"] is not None, "namespace": ns}

    @app.post("/query")
    def query(req: QueryRequest):
        t0 = time.perf_counter()
        try:
            vec = state["embed_fn"](req.question)
        except httpx.HTTPError as e:
            raise HTTPException(502, f"embed service unavailable: {type(e).__name__}")
        t1 = time.perf_counter()
        res = get_index().query(vector=vec, top_k=req.top_k, include_metadata=True, namespace=ns)
        t2 = time.perf_counter()
        matches = res.matches
        rows = state["store"].get_many([m.id for m in matches])
        t3 = time.perf_counter()
        results = []
        for m in matches:
            row = rows.get(m.id)
            if row is None:                            # vector in Pinecone but text missing locally
                continue
            results.append({"chunk_id": m.id, "score": float(m.score), **{k: v for k, v in row.items() if k != "chunk_id"}})
        return {
            "question": req.question,
            "results": results,
            "timings_ms": {
                "embed": round((t1 - t0) * 1000, 3),
                "pinecone": round((t2 - t1) * 1000, 3),
                "lookup": round((t3 - t2) * 1000, 3),
                "total": round((t3 - t0) * 1000, 3),
            },
        }

    return app


app = create_app()
