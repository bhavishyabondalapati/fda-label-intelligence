"""Cross-encoder reranker. A bi-encoder (MiniLM) embeds question and chunk separately; a
cross-encoder reads them TOGETHER, which is slower but much better at judging relevance. So we use
it only on the top candidates from the fast first stage."""
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker(device=None):
    from sentence_transformers import CrossEncoder
    from src.embed.local import pick_device
    return CrossEncoder(MODEL_NAME, device=device or pick_device(), max_length=320)


def rerank(model, question, candidates, texts, batch_size=50):
    """candidates: list of ids; texts: list of their texts. Returns ids sorted best first."""
    if not candidates:
        return []
    scores = model.predict([(question, t) for t in texts], batch_size=batch_size, show_progress_bar=False)
    return [candidates[i] for i in sorted(range(len(candidates)), key=lambda j: -scores[j])]
