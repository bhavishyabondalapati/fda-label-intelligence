"""Thin wrapper around Pinecone: create the serverless index, upsert with retries, query.

Only small metadata is stored in Pinecone (set_id, drug, route, section, chunk_id).
The API key is read from the environment and is never printed.
"""
import os
import time

DIM = 384
METRIC = "cosine"
# Pinecone's free (Starter) plan only allows serverless indexes in AWS us-east-1.
CLOUD, REGION = "aws", "us-east-1"
MAX_META_STR = 120


def get_client():
    from pinecone import Pinecone
    key = os.getenv("PINECONE_API_KEY")
    if not key:
        raise RuntimeError("PINECONE_API_KEY is not set (put it in .env)")
    return Pinecone(api_key=key)


def ensure_index(pc, name):
    """Create the index if missing (384 dims, cosine, serverless) and wait until ready."""
    from pinecone import ServerlessSpec
    existing = {i["name"] for i in pc.list_indexes()}
    if name not in existing:
        pc.create_index(name=name, dimension=DIM, metric=METRIC, spec=ServerlessSpec(cloud=CLOUD, region=REGION))
    while not pc.describe_index(name).status["ready"]:
        time.sleep(2)
    return pc.Index(name)


def open_index(pc, name):
    """Open an existing index (no control-plane create). Raises if it does not exist."""
    return pc.Index(name)


def to_metadata(chunk_id, set_id, drug, route, section):
    return {"chunk_id": chunk_id, "set_id": set_id, "drug": (drug or "")[:MAX_META_STR],
            "route": (route or "")[:MAX_META_STR], "section": section or ""}


def to_metadata_v2(chunk_id, set_id, drug, route, section):
    """Phase 2 adds two normalised fields so queries can filter on drug (+ route)."""
    from src.eval.gold import norm_name
    meta = to_metadata(chunk_id, set_id, drug, route, section)
    meta["drug_norm"] = norm_name(drug or "")[:MAX_META_STR]
    meta["route_norm"] = (route or "").strip().lower()[:MAX_META_STR]
    return meta


def upsert_with_retries(index, records, namespace, attempts=6):
    """records: list of {"id", "values", "metadata"}. Retries with exponential backoff."""
    delay = 1.0
    for attempt in range(1, attempts + 1):
        try:
            index.upsert(vectors=records, namespace=namespace)
            return
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30)
