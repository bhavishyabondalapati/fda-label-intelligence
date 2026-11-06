"""Settings shared by the services. Values come from environment variables / .env.

Never print or log secret values (PINECONE_API_KEY).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

PORTS = {"ingest": 8001, "chunk": 8002, "embed": 8003, "query": 8004}

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data" / "processed"))
CHUNK_DB = Path(os.getenv("CHUNK_DB", DATA_DIR / "chunks.sqlite"))

EMBED_URL = os.getenv("EMBED_URL", f"http://127.0.0.1:{PORTS['embed']}")
CHUNK_URL = os.getenv("CHUNK_URL", f"http://127.0.0.1:{PORTS['chunk']}")

PINECONE_INDEX = os.getenv("PINECONE_INDEX", "fda-labels")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "naive-v1")   # one namespace per pipeline version
