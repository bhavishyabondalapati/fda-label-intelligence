"""Tests for the Phase 2 retrieval pieces (small hand-made data, no models, no network)."""
from src.chunk.naive import join_label
from src.chunk.variants import embed_text, header_for, section_chunks
from src.eval.metrics import overlaps_section
from src.retrieve.bm25 import BM25
from src.retrieve.drug_detect import DrugDetector
from src.retrieve.fusion import rrf

SECTIONS = [{"section": "indications_and_usage", "text": "A" * 2500},
            {"section": "overdosage", "text": "B" * 300}]


def test_section_chunks_never_mix_sections_and_keep_joined_coordinates():
    _, spans = join_label(SECTIONS)
    chunks = section_chunks("L1", "drugx", "ORAL", SECTIONS)
    for c in chunks:
        a, b = spans[c["section"]]
        assert a <= c["start"] < c["end"] <= b          # inside its own section
        assert set(c["text"]) == ({"A"} if c["section"] == "indications_and_usage" else {"B"})
    # the overdosage section is covered, so the frozen hit rule can find it
    over = [c for c in chunks if c["section"] == "overdosage"]
    assert any(overlaps_section(c["start"], c["end"], spans["overdosage"]) for c in over)


def test_header_text():
    assert header_for("Lidocaine", "TRANSDERMAL", "boxed_warning") == "Lidocaine (transdermal) - boxed warning:"
    assert header_for("Lidocaine", "", "overdosage") == "Lidocaine - overdosage:"
    c = {"drug": "X", "route": "ORAL", "section": "overdosage", "text": "body"}
    assert embed_text(c, header=False) == "body" and embed_text(c).endswith(" body")


def test_bm25_ranks_the_chunk_with_the_rare_word_first():
    docs = ["common words about tablets", "warfarin bleeding risk tablets", "other tablets text"]
    bm = BM25(docs)
    assert bm.top("warfarin risk", k=3)[0] == 1
    assert bm.top("warfarin", k=3, allowed=[0, 2]) == []        # filter removes the only match


def test_drug_detector_longest_match_and_route():
    labels = [{"drug_name": "Lidocaine", "route": "TOPICAL"}, {"drug_name": "Lidocaine", "route": "INFILTRATION"},
              {"drug_name": "Lidocaine Hydrochloride", "route": "INJECTION"}, {"drug_name": "Aspirin", "route": "ORAL"}]
    det = DrugDetector(labels)
    assert det.detect("What are the warnings for Lidocaine Hydrochloride?") == ("lidocaine hydrochloride", None)
    assert det.detect("What is Lidocaine (topical) indicated for?") == ("lidocaine", "topical")
    assert det.detect("How should Lidocaine (banana) be stored?") == ("lidocaine", None)   # not a route of it
    assert det.detect("What is the best way to relax?") is None                        # nothing confident


def test_rrf_prefers_items_ranked_high_in_both_lists():
    assert rrf([[1, 2, 3], [3, 1, 4]])[:2] == [1, 3]
