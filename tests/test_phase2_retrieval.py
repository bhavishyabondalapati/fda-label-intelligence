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


# ---------- the whole pipeline with fakes (no Pinecone, no models) ----------
class FakeIndex:
    """Pretends to be Pinecone: returns a fixed ranking and remembers the last query it got."""
    def __init__(self, ids):
        self.ids, self.last = ids, None

    def query(self, **kw):
        from types import SimpleNamespace
        self.last = kw
        return SimpleNamespace(matches=[SimpleNamespace(id=i, score=1 - n / 10) for n, i in enumerate(self.ids)])


class FakeStore:
    def __init__(self, rows):
        self.rows = {r["chunk_id"]: r for r in rows}

    def get_many(self, ids):
        return {i: self.rows[i] for i in ids if i in self.rows}


def _rows():
    mk = lambda i, drug, text: {"chunk_id": f"c{i}", "set_id": f"s{i}", "drug": drug, "route": "ORAL",
                                "section": "overdosage", "start": 0, "end": len(text), "text": text}
    return [mk(0, "Aspirin", "aspirin overdose text"), mk(1, "Warfarin", "warfarin bleeding overdose"),
            mk(2, "Warfarin", "warfarin tablets supplied")]


def test_pipeline_filter_hybrid_and_rerank_order():
    from src.retrieve.pipeline import Pipeline
    rows = _rows()
    det = DrugDetector([{"drug_name": r["drug"], "route": r["route"]} for r in rows])
    bm25 = BM25([r["text"] for r in rows])

    class Reranker:                                   # prefers the chunk that says "bleeding"
        def predict(self, pairs, **kw):
            return [1.0 if "bleeding" in t else 0.0 for _, t in pairs]

    index = FakeIndex(["c0", "c1", "c2"])
    p = Pipeline(lambda q: [0.0], index, "ns", FakeStore(rows), header=False, detector=det, bm25=bm25,
                 reranker=Reranker(), rows=rows)
    results, timings, detected = p.search("What happens in an overdose of Warfarin?", top_k=2)
    assert detected == ("warfarin", None)
    assert index.last["filter"] == {"drug_norm": {"$eq": "warfarin"}}      # drug detected -> Pinecone filter
    assert results[0]["chunk_id"] == "c1" and len(results) == 2            # reranker put "bleeding" first
    assert {"embed", "pinecone", "bm25", "lookup", "rerank", "total"} <= set(timings)


def test_pipeline_without_switches_is_plain_dense_search():
    from src.retrieve.pipeline import Pipeline
    rows = _rows()
    index = FakeIndex(["c2", "c0", "c1"])
    p = Pipeline(lambda q: [0.0], index, "ns", FakeStore(rows), use_filter=False, use_hybrid=False,
                 use_rerank=False)
    results, timings, detected = p.search("anything", top_k=2)
    assert [r["chunk_id"] for r in results] == ["c2", "c0"] and detected is None
    assert "filter" not in index.last and index.last["top_k"] == 2
