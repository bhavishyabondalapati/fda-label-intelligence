"""Local chunk store: chunk text and positions in SQLite, looked up by chunk_id.

Pinecone holds only the vector plus tiny metadata; the text (bulky) stays here.
"""
import sqlite3
from pathlib import Path

COLUMNS = ("chunk_id", "set_id", "drug", "route", "section", "start", "end", "text")


class ChunkStore:
    def __init__(self, path):
        self.path = Path(path)

    def _connect(self, read_only=False):
        if read_only:
            return sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.path)

    def create(self):
        """Start a fresh store (drops any existing one)."""
        self.path.unlink(missing_ok=True)
        with self._connect() as con:
            con.execute("""CREATE TABLE chunks (
                n INTEGER PRIMARY KEY AUTOINCREMENT,   -- insertion order = embedding order
                chunk_id TEXT UNIQUE NOT NULL, set_id TEXT, drug TEXT, route TEXT,
                section TEXT, start INTEGER, "end" INTEGER, text TEXT)""")

    def add_many(self, chunks):
        rows = [(c["chunk_id"], c["set_id"], c["drug"], c["route"], c["section"], c["start"], c["end"], c["text"])
                for c in chunks]
        with self._connect() as con:
            con.executemany(
                'INSERT INTO chunks (chunk_id,set_id,drug,route,section,start,"end",text) VALUES (?,?,?,?,?,?,?,?)', rows)

    def count(self):
        con = self._connect(read_only=True)
        try:
            return con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        finally:
            con.close()

    def get_many(self, chunk_ids):
        """Return {chunk_id: row dict} for the ids that exist."""
        if not chunk_ids:
            return {}
        marks = ",".join("?" * len(chunk_ids))
        con = self._connect(read_only=True)
        try:
            rows = con.execute(
                f'SELECT chunk_id,set_id,drug,route,section,start,"end",text FROM chunks WHERE chunk_id IN ({marks})',
                list(chunk_ids)).fetchall()
        finally:
            con.close()
        return {r[0]: dict(zip(COLUMNS, r)) for r in rows}

    def iter_in_order(self, batch=5000):
        """Yield (chunk_id, text) in insertion order, for embedding and for index parity checks."""
        con = self._connect(read_only=True)
        try:
            cur = con.execute("SELECT chunk_id, text FROM chunks ORDER BY n")
            while rows := cur.fetchmany(batch):
                yield from rows
        finally:
            con.close()

    def metadata_in_order(self, batch=5000):
        """Yield (chunk_id, set_id, drug, route, section) in insertion order (for Pinecone upsert)."""
        con = self._connect(read_only=True)
        try:
            cur = con.execute("SELECT chunk_id, set_id, drug, route, section FROM chunks ORDER BY n")
            while rows := cur.fetchmany(batch):
                yield from rows
        finally:
            con.close()
