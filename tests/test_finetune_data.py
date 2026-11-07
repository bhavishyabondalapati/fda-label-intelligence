import json

from src.extract.validate import quote_ok
from src.finetune.data import build_example, find_quote, find_route_quote, squash

TEXT = ("WARNING: serious heart risk.\n\nSildenafil citrate tablets are a phosphodiesterase inhibitor indicated for "
        "erectile dysfunction.   Take by mouth with water. " * 3)
LABEL = {"generic_name": "SILDENAFIL", "substance_name": ["SILDENAFIL CITRATE"], "route": "ORAL",
         "pharm_class_epc": [], "sections": ["boxed_warning", "indications_and_usage"]}
SECS = {"boxed_warning": "WARNING: serious heart risk."}


def test_quotes_are_verbatim_and_contain_the_value():
    src = squash(TEXT)
    q = find_quote(src, "sildenafil citrate")
    assert q and "sildenafil citrate" in q.lower() and quote_ok(q, src.lower()) and 3 <= len(q) <= 300
    assert find_quote(src, "tadalafil") is None
    r = find_route_quote(src, "ORAL")
    assert r and quote_ok(r, src.lower())


def test_example_targets_match_gold_and_quotes_pass_validator():
    ex = build_example(LABEL, SECS, TEXT)
    c = json.loads(ex["completion"])
    assert c["generic_name"]["value"] == "sildenafil" and c["is_combination"]["value"] is False
    assert c["route"]["value"] == "ORAL" and c["has_boxed_warning"]["value"] is True
    src = squash(TEXT).lower()
    for f in ("generic_name", "active_ingredients", "route", "has_boxed_warning"):
        assert c[f]["quotes"] and all(quote_ok(q, src) for q in c[f]["quotes"])


def test_empty_gold_pharm_class_is_masked_but_found_class_is_not():
    ex = build_example(LABEL, SECS, TEXT)                        # gold empty -> masked
    (a, b), = ex["mask_spans"]
    assert ex["completion"][a:b] == '{"value": [], "quotes": []' and ex["completion"][b:].startswith(', "confidence"')
    lab = dict(LABEL, pharm_class_epc=["Phosphodiesterase Inhibitor [EPC]"])
    ex2 = build_example(lab, SECS, TEXT)
    assert ex2["mask_spans"] == [] and json.loads(ex2["completion"])["pharm_class"]["value"] == ["Phosphodiesterase Inhibitor"]
    lab3 = dict(LABEL, pharm_class_epc=["Not In The Text [EPC]"])   # non-empty gold but absent from text -> masked
    assert build_example(lab3, SECS, TEXT)["mask_spans"]


def test_missing_value_keeps_example_with_empty_quotes():
    lab = dict(LABEL, generic_name="TADALAFIL")
    c = json.loads(build_example(lab, SECS, TEXT)["completion"])
    assert c["generic_name"]["quotes"] == [] and c["generic_name"]["value"] == "tadalafil"


def test_unknown_route_becomes_other():
    c = json.loads(build_example(dict(LABEL, route="PARENTERAL"), SECS, TEXT)["completion"])
    assert c["route"]["value"] == "OTHER"
