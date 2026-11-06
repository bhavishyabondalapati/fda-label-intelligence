"""Local embeddings with all-MiniLM-L6-v2 (384 dimensions, free, runs on the Mac)."""

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def pick_device() -> str:
    """Apple Silicon GPU ("mps") if available, otherwise CPU. Never CUDA on this Mac."""
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(device=None):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME, device=device or pick_device())


def embed_texts(model, texts, batch_size=64, show_progress=True):
    """Return a float32 array (n, 384) of L2-normalised vectors.

    Normalised vectors make a plain dot product equal to cosine similarity.
    """
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    ).astype("float32")
