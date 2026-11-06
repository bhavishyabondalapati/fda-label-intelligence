"""Naive fixed-size chunking: the Phase 0 BASELINE.

It glues all sections of a label into one string and slices it into equal
windows, ignoring section boundaries and adding no context. That is what
a quick first RAG prototype does, and it is what later phases must beat.
"""

CHUNK_SIZE = 1000   # characters
OVERLAP = 100       # characters shared by neighbouring chunks
SECTION_SEPARATOR = "\n\n"


def join_label(sections):
    """sections: list of {"section", "text"} in order.

    Returns (full_text, spans) where spans maps section name -> (start, end)
    character offsets inside full_text. Spans are how we know which section a
    chunk came from, even though the chunker itself ignores sections.
    """
    parts, spans, pos = [], {}, 0
    for s in sections:
        if parts:
            pos += len(SECTION_SEPARATOR)
        spans[s["section"]] = (pos, pos + len(s["text"]))
        parts.append(s["text"])
        pos += len(s["text"])
    return SECTION_SEPARATOR.join(parts), spans


def naive_chunks(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """Yield (start, end, chunk_text) windows covering the whole text."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    step = size - overlap
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        yield start, end, text[start:end]
        if end == len(text):
            break
        start += step
