"""Reciprocal rank fusion: merge several ranked lists into one without comparing their scores."""


def rrf(rankings, k=60, top=100):
    """rankings: list of ranked id lists (best first). Score = sum over lists of 1/(k+rank)."""
    score = {}
    for ranked in rankings:
        for rank, i in enumerate(ranked, start=1):
            score[i] = score.get(i, 0.0) + 1.0 / (k + rank)
    return [i for i, _ in sorted(score.items(), key=lambda kv: -kv[1])][:top]
