"""Turn a label's sections into chunk records (the unit that gets embedded and indexed)."""
from src.chunk.naive import CHUNK_SIZE, OVERLAP, join_label, naive_chunks


def primary_section(spans, start, end):
    """The section that the chunk overlaps most (chunks can straddle sections)."""
    best, best_overlap = "", 0
    for name, (a, b) in spans.items():
        overlap = min(end, b) - max(start, a)
        if overlap > best_overlap:
            best, best_overlap = name, overlap
    return best


def chunk_label(set_id, drug_name, route, sections, size=CHUNK_SIZE, overlap=OVERLAP):
    """sections: list of {"section", "text"}. Returns a list of chunk dicts.

    chunk_id is "<set_id>:<n>" where n is the chunk's position inside the label, so ids are
    stable and sort in the same order as the Phase 0 baseline chunks.
    """
    text, spans = join_label(sections)
    out = []
    for n, (start, end, chunk_text) in enumerate(naive_chunks(text, size, overlap)):
        out.append({
            "chunk_id": f"{set_id}:{n}",
            "set_id": set_id,
            "drug": drug_name,
            "route": route,
            "section": primary_section(spans, start, end),
            "start": start,
            "end": end,
            "text": chunk_text,
        })
    return out
