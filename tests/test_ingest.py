from src.ingest.openfda import (
    assign_split,
    clean_text,
    is_human_rx,
    parse_label,
    sample_key,
)

LONG = "word " * 40  # 200 chars, above the minimum section length


def make_raw(set_id="abc", brand=("Zestril",), generic=("lisinopril",), n_sections=5, rx=True):
    sections = [
        "indications_and_usage",
        "contraindications",
        "dosage_and_administration",
        "adverse_reactions",
        "description",
        "how_supplied",
    ][:n_sections]
    raw = {
        "set_id": set_id,
        "effective_time": "20240101",
        "openfda": {
            "brand_name": list(brand),
            "generic_name": list(generic),
            "manufacturer_name": ["Acme Pharma"],
            "product_type": ["HUMAN PRESCRIPTION DRUG LABEL" if rx else "HUMAN OTC DRUG LABEL"],
        },
    }
    for s in sections:
        raw[s] = [LONG, "second   paragraph\n with   spaces"]
    return raw


def test_clean_text_joins_and_collapses_whitespace():
    assert clean_text(["a  b", "c\n\nd"]) == "a b c d"
    assert clean_text("  x   y ") == "x y"


def test_is_human_rx():
    assert is_human_rx(make_raw())
    assert not is_human_rx(make_raw(rx=False))
    assert not is_human_rx({})


def test_parse_label_extracts_sections_and_metadata():
    meta, sections = parse_label(make_raw(), seed=42)
    assert meta["set_id"] == "abc"
    assert meta["drug_name"] == "Zestril"
    assert meta["manufacturer"] == "Acme Pharma"
    assert [s["section"] for s in sections] == meta["sections"]
    assert len(sections) == 5
    assert "  " not in sections[0]["text"]


def test_drug_name_falls_back_to_generic():
    meta, _ = parse_label(make_raw(brand=()), seed=42)
    assert meta["drug_name"] == "lisinopril"


def test_unusable_labels_are_rejected():
    assert parse_label(make_raw(rx=False), seed=42) is None
    assert parse_label(make_raw(n_sections=3), seed=42) is None      # too few sections
    assert parse_label(make_raw(brand=(), generic=()), seed=42) is None  # no name


def test_sampling_key_and_split_are_deterministic():
    assert sample_key(42, "x") == sample_key(42, "x")
    assert sample_key(42, "x") != sample_key(43, "x")
    assert assign_split(42, "x") == assign_split(42, "x")


def test_split_shares_are_roughly_right():
    counts = {"retrieval_eval": 0, "extraction_test": 0, "extraction_train": 0}
    n = 5000
    for i in range(n):
        counts[assign_split(42, f"id-{i}")] += 1
    assert 0.07 < counts["retrieval_eval"] / n < 0.13
    assert 0.07 < counts["extraction_test"] / n < 0.13
    assert counts["extraction_train"] / n > 0.74
