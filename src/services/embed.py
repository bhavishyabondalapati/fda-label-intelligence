"""Embed service: POST /embed {"texts": [...]} -> vectors. The model loads ONCE at startup."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=1024)


class MiniLMEmbedder:
    """The real embedder: loads all-MiniLM-L6-v2 once, on Apple's GPU ("mps") when available."""

    def __init__(self):
        from src.embed.local import MODEL_NAME, load_model, pick_device
        self.name = MODEL_NAME
        self.device = pick_device()
        self._model = load_model(self.device)
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts):
        from src.embed.local import embed_texts
        return embed_texts(self._model, texts, show_progress=False)


def create_app(embedder=None) -> FastAPI:
    """embedder: optional object with .encode(texts)->array and .dim/.name/.device (tests pass a fake)."""

    @asynccontextmanager
    async def lifespan(app):
        app.state.model = embedder if embedder is not None else MiniLMEmbedder()
        yield

    app = FastAPI(title="embed-service", lifespan=lifespan)

    @app.get("/health")
    def health():
        m = app.state.model
        return {"status": "ok", "model": m.name, "dim": m.dim, "device": m.device}

    # `async def` on purpose: it runs on the event loop thread, one request at a time, which
    # keeps the single Apple-GPU ("mps") model from being called from several threads at once.
    @app.post("/embed")
    async def embed(req: EmbedRequest):
        vectors = app.state.model.encode(req.texts)
        return {"dim": int(vectors.shape[1]), "vectors": vectors.tolist()}

    return app


app = create_app()
