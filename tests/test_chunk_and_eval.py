import pytest

from src.chunk.naive import SECTION_SEPARATOR, join_label, naive_chunks
from src.eval.gold import TEMPLATES, build_gold
from src.eval.metrics import chunk_hits_gold, chunk_hits_relevant, recall_at_k, relative_improvement


def test_join_label_spans_point_at_section_text():
    secs = [{"section": "a", "text": "AAAA"}, {"section": "b", "text": "BBBBBB"}]
    text, spans = join_label(secs)
    assert text == "AAAA" + SECTION_SEPARATOR + "BBBBBB"
    for s in secs:
        start, end = spans[s["section"]]
        assert text[start:end] == s["text"]


def test_naive_chunks_cover_text_with_overlap():
    text = "x" * 2500
    chunks = list(naive_chunks(text, size=1000, overlap=100))
    assert chunks[0][:2] == (0, 1000)
    assert chunks[1][0] == 900                       # next chunk starts 100 chars back
    assert chunks[-1][1] == len(text)                # last chunk reaches the end
    assert all(text[s:e] == c for s, e, c in chunks)


def test_naive_chunks_rejects_bad_overlap():
    with pytest.raises(ValueError):
        list(naive_chunks("abc", size=10, overlap=10))


def test_chunk_hits_gold_rules():
    span = (1000, 3000)
    assert chunk_hits_gold("L", 1500, 2500, "L", span)           # inside
    assert not chunk_hits_gold("OTHER", 1500, 2500, "L", span)   # wrong label
    assert not chunk_hits_gold("L", 0, 1050, "L", span)          # 50-char graze does not count
    assert chunk_hits_gold("L", 0, 1300, "L", span)              # 300-char overlap counts
    assert chunk_hits_gold("L", 0, 5000, "L", (10, 60))          # tiny section fully covered


def test_recall_at_k_and_relative_improvement():
    assert recall_at_k([[False, True], [False, False], [True]]) == pytest.approx(2 / 3)
    assert recall_at_k([]) == 0.0
    assert relative_improvement(0.65, 0.5) == pytest.approx(0.30)
    with pytest.raises(ValueError):
        relative_improvement(0.5, 0)


def _fake_labels(n):
    labels, sections = [], {}
    for i in range(n):
        sid = f"id{i}"
        labels.append({
            "set_id": sid, "drug_name": f"Drug{i}", "brand_name": f"Drug{i}", "generic_name": f"gen{i}",
            "manufacturer": "Acme", "route": "ORAL", "split": "retrieval_eval" if i % 2 == 0 else "extraction_train",
        })
        sections[sid] = {s: "t" * 400 for s in TEMPLATES}
    return labels, sections


def test_build_gold_is_deterministic_and_uses_only_eval_split():
    labels, sections = _fake_labels(100)
    g1 = build_gold(labels, sections, n_questions=20, seed=1)
    g2 = build_gold(labels, sections, n_questions=20, seed=1)
    assert g1 == g2 and len(g1) == 20
    eval_ids = {lab["set_id"] for lab in labels if lab["split"] == "retrieval_eval"}
    assert all(g["set_id"] in eval_ids for g in g1)
    assert all(g["drug_name"] in g["question"] for g in g1)


def test_build_gold_questions_do_not_name_the_manufacturer():
    labels, sections = _fake_labels(20)
    gold = build_gold(labels, sections, n_questions=5, seed=1)
    assert gold and all("Acme" not in g["question"] for g in gold)
    assert all(g["route"] == "ORAL" for g in gold)


def test_build_gold_puts_route_in_question_for_multi_route_names():
    labels, sections = _fake_labels(4)
    for lab in labels:                      # same name on every label, all retrieval_eval
        lab["brand_name"] = lab["drug_name"] = "Same"
        lab["split"] = "retrieval_eval"
    gold = build_gold(labels, sections, n_questions=4, seed=1)
    assert len(gold) == 4 and not any(g["route_in_question"] for g in gold)   # one route: plain name
    labels[0]["route"], labels[1]["route"] = "TOPICAL", "TOPICAL"             # now two routes exist
    gold = build_gold(labels, sections, n_questions=4, seed=1)
    assert len(gold) == 4 and all(g["route_in_question"] for g in gold)
    assert all(f"({g['route'].lower()})" in g["question"] for g in gold)


def test_build_gold_skips_multi_route_label_with_no_route():
    labels, sections = _fake_labels(2)
    for lab in labels:
        lab["brand_name"] = lab["drug_name"] = "Same"
        lab["split"] = "retrieval_eval"
    labels[0]["route"] = ""                 # "Same" now exists as "" and "ORAL": the "" one cannot be asked
    gold = build_gold(labels, sections, n_questions=2, seed=1)
    assert [g["set_id"] for g in gold] == ["id1"]


def test_relaxed_hit_accepts_other_manufacturers_copy_of_same_drug_and_route():
    key = ("xylocaine", "injection")
    own_span = (100, 900)                                    # that label's copy of the gold section
    assert chunk_hits_relevant(key, 150, 1150, key, own_span)                 # other label, same drug+route
    assert not chunk_hits_relevant(("xylocaine", "topical"), 150, 1150, key, own_span)   # different route
    assert not chunk_hits_relevant(("lidocaine", "injection"), 150, 1150, key, own_span)  # different name
    assert not chunk_hits_relevant(key, 150, 1150, key, None)                 # label lacks that section
    assert not chunk_hits_relevant(key, 2000, 3000, key, own_span)            # chunk is outside the section


def test_mrr():
    from src.eval.metrics import mrr
    assert mrr([[True], [False, True], [False, False]]) == pytest.approx((1 + 0.5 + 0) / 3)
