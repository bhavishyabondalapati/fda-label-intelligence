"""Phase 2 chunking variants. Both keep the SAME character coordinates as the baseline
(offsets into the label's joined text, see naive.join_label), so the frozen hit rule applies
to them unchanged.

  * section_chunks: split on section boundaries FIRST, then by size inside each section.
    A chunk never mixes two sections, and every chunk knows its exact section.
  * header_for: the contextual header "<drug> (<route>) - <section>:" put in front of the text
    that gets embedded (the stored/returned chunk text stays without it).
"""
from src.chunk.naive import CHUNK_SIZE, OVERLAP, join_label, naive_chunks
from src.chunk.records import primary_section


def header_for(drug, route, section):
    route = (route or "").strip().lower()
    name = f"{drug} ({route})" if route else drug
    return f"{name} - {section.replace('_', ' ')}:"


def section_chunks(set_id, drug, route, sections, size=CHUNK_SIZE, overlap=OVERLAP):
    """Same record format as records.chunk_label, but section-aware."""
    _, spans = join_label(sections)
    out, n = [], 0
    for s in sections:
        base = spans[s["section"]][0]
        for start, end, text in naive_chunks(s["text"], size, overlap):
            out.append({"chunk_id": f"{set_id}:{n}", "set_id": set_id, "drug": drug, "route": route,
                        "section": s["section"], "start": base + start, "end": base + end, "text": text})
            n += 1
    return out


def naive_records(set_id, drug, route, sections, size=CHUNK_SIZE, overlap=OVERLAP):
    from src.chunk.records import chunk_label
    return chunk_label(set_id, drug, route, sections, size, overlap)


def embed_text(chunk, header=True):
    """The string that is embedded (and BM25-indexed) for a chunk."""
    if not header:
        return chunk["text"]
    return header_for(chunk["drug"], chunk["route"], chunk["section"]) + " " + chunk["text"]
