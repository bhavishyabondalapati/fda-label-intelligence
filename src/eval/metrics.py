"""Metric helpers. Definitions match PLAN.md exactly."""

MIN_OVERLAP_CHARS = 200


def overlaps_section(chunk_start, chunk_end, span):
    """True if a chunk covers a section: overlap >= min(200, section length) characters.

    So a chunk that only grazes the section boundary by a few characters does not count.
    """
    s_start, s_end = span
    overlap = min(chunk_end, s_end) - max(chunk_start, s_start)
    return overlap >= min(MIN_OVERLAP_CHARS, s_end - s_start)


def chunk_hits_gold(chunk_set_id, chunk_start, chunk_end, gold_set_id, gold_span):
    """STRICT hit: the chunk is in the exact gold label AND covers the gold section."""
    return chunk_set_id == gold_set_id and overlaps_section(chunk_start, chunk_end, gold_span)


def chunk_hits_relevant(chunk_key, chunk_start, chunk_end, gold_key, chunk_section_span):
    """RELAXED hit (primary metric): the chunk's label has the same (drug name, route) as
    the gold label, and the chunk covers that label's own copy of the gold section.

    chunk_section_span is the span of the gold section inside the chunk's own label
    (None if that label has no such section).
    """
    return (chunk_key == gold_key and chunk_section_span is not None
            and overlaps_section(chunk_start, chunk_end, chunk_section_span))


def recall_at_k(hit_flags_per_query):
    """hit_flags_per_query: list of lists of bools, one list per query (top-k in rank order).

    A query is a hit if ANY of its top-k results is a hit.
    """
    if not hit_flags_per_query:
        return 0.0
    return sum(any(flags) for flags in hit_flags_per_query) / len(hit_flags_per_query)


def mrr(hit_flags_per_query):
    """Mean reciprocal rank: average of 1/rank of the first hit (0 if no hit in the list)."""
    if not hit_flags_per_query:
        return 0.0
    total = 0.0
    for flags in hit_flags_per_query:
        for rank, hit in enumerate(flags, start=1):
            if hit:
                total += 1 / rank
                break
    return total / len(hit_flags_per_query)


def relative_improvement(new, baseline):
    """(new - baseline) / baseline. Reported as a fraction, e.g. 0.30 = +30%."""
    if baseline == 0:
        raise ValueError("baseline is 0: relative improvement is undefined")
    return (new - baseline) / baseline
