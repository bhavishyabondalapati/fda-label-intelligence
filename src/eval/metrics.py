"""Metric helpers. Definitions match PLAN.md exactly."""

MIN_OVERLAP_CHARS = 200


def chunk_hits_gold(chunk_set_id, chunk_start, chunk_end, gold_set_id, gold_span):
    """True if a naive chunk lies in the gold label AND really covers the gold section.

    Overlap must be at least min(200, section length) characters, so a chunk that
    only touches the section boundary by a few characters does not count.
    """
    if chunk_set_id != gold_set_id:
        return False
    g_start, g_end = gold_span
    overlap = min(chunk_end, g_end) - max(chunk_start, g_start)
    return overlap >= min(MIN_OVERLAP_CHARS, g_end - g_start)


def recall_at_k(hit_flags_per_query):
    """hit_flags_per_query: list of lists of bools, one list per query (top-k in rank order).

    A query is a hit if ANY of its top-k results is a hit.
    """
    if not hit_flags_per_query:
        return 0.0
    return sum(any(flags) for flags in hit_flags_per_query) / len(hit_flags_per_query)


def relative_improvement(new, baseline):
    """(new - baseline) / baseline. Reported as a fraction, e.g. 0.30 = +30%."""
    if baseline == 0:
        raise ValueError("baseline is 0: relative improvement is undefined")
    return (new - baseline) / baseline
