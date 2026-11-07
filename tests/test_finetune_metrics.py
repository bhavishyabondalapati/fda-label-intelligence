from src.finetune.metrics import bootstrap, counts, micro_f1, parse_reply, values_from, FIELDS5

LABEL = {"generic_name": "SILDENAFIL", "substance_name": ["SILDENAFIL CITRATE"], "route": "ORAL",
         "pharm_class_epc": [], "sections": ["boxed_warning"]}


def entry(v):
    return {"value": v, "quotes": [], "confidence": "high"}


GOOD = {"generic_name": entry("sildenafil"), "active_ingredients": entry(["sildenafil citrate"]),
        "is_combination": entry(False), "route": entry("ORAL"), "has_boxed_warning": entry(True), "pharm_class": entry([])}


def test_parse_reply_handles_fences_and_garbage():
    assert parse_reply('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_reply("I cannot do that") is None
    assert parse_reply("[1, 2]") is None


def test_perfect_output_scores_one_and_unparsed_scores_zero():
    v = values_from(GOOD)
    assert micro_f1([counts(v, LABEL, FIELDS5)]) == 1.0
    assert micro_f1([counts(None, LABEL, FIELDS5)]) == 0.0


def test_lenient_coercions():
    bad = dict(GOOD, active_ingredients=entry("sildenafil citrate"), has_boxed_warning=entry("yes"), pharm_class=entry("[Not Applicable]"))
    v = values_from(bad)
    assert v["active_ingredients"] == ["sildenafil citrate"] and v["has_boxed_warning"] is None
    assert v["pharm_class"] == ["[Not Applicable]"]


def test_bootstrap_interval_orders_and_is_deterministic():
    a = [(0, 1, 1)] * 20 + [(1, 0, 0)] * 10
    b = [(1, 0, 0)] * 30
    r1, r2 = bootstrap(a, b, n=200), bootstrap(a, b, n=200)
    assert r1 == r2 and r1["abs_ci95"][0] > 0
