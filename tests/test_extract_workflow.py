import json

from langchain_core.messages import AIMessage

from src.extract.graph import build_graph, parse_json
from src.extract.retrieve import TOTAL_CAP, select_text
from src.extract.validate import quote_ok, route_decisions, validate_extraction
from src.generation.usage import UsageLedger

SRC = ("WARNING: serious heart risk. Sildenafil citrate tablets are indicated for erectile dysfunction. "
       "Take by mouth with   water.")


def field(value, quotes, conf="high"):
    return {"value": value, "quotes": quotes, "confidence": conf}


def good():
    return {
        "generic_name": field("sildenafil", ["Sildenafil citrate tablets"]),
        "active_ingredients": field(["sildenafil citrate"], ["Sildenafil citrate tablets"]),
        "is_combination": field(False, []),
        "route": field("ORAL", ["Take by mouth with water"]),
        "has_boxed_warning": field(True, ["WARNING: serious heart risk"]),
        "pharm_class": field([], []),
    }


def test_quote_must_be_verbatim_ignoring_case_and_whitespace():
    s = " ".join(SRC.lower().split())
    assert quote_ok("take BY mouth with water", s)
    assert not quote_ok("Take it by mouth", s)
    assert not quote_ok("ab", s)                  # too short


def test_all_good_fields_pass_gates_except_absence():
    v = validate_extraction(good(), SRC)
    assert all(v[f]["gates_pass"] for f in ("generic_name", "active_ingredients", "is_combination", "route", "has_boxed_warning"))
    assert not v["pharm_class"]["gates_pass"] and "absence" in v["pharm_class"]["reasons"][0]


def test_made_up_quote_fails_even_with_high_confidence():
    e = good()
    e["route"] = field("ORAL", ["Swallow whole with a glass of water"], "high")
    v = validate_extraction(e, SRC)
    assert not v["route"]["gates_pass"]
    assert not route_decisions(v, "low")["route"]["approved"]        # confidence cannot rescue it


def test_confidence_only_adds_a_requirement():
    e = good()
    e["route"]["confidence"] = "low"
    v = validate_extraction(e, SRC)
    assert v["route"]["gates_pass"]
    assert route_decisions(v, "low")["route"]["approved"]
    assert not route_decisions(v, "medium")["route"]["approved"]


def test_value_must_be_supported_by_its_quote():
    e = good()
    e["generic_name"] = field("tadalafil", ["Sildenafil citrate tablets"])
    assert not validate_extraction(e, SRC)["generic_name"]["gates_pass"]
    e = good()
    e["route"] = field("INTRAVENOUS", ["Take by mouth with water"])
    assert not validate_extraction(e, SRC)["route"]["gates_pass"]


def test_is_combination_must_match_ingredient_count():
    e = good()
    e["is_combination"] = field(True, [])
    v = validate_extraction(e, SRC)
    assert not v["is_combination"]["gates_pass"] and not v["active_ingredients"]["gates_pass"]


def test_generic_and_ingredients_must_agree():
    e = good()
    e["generic_name"] = field("tadalafil", ["Sildenafil citrate tablets"])
    e["active_ingredients"] = field(["sildenafil citrate"], ["Sildenafil citrate tablets"])
    v = validate_extraction(e, SRC)
    assert not v["generic_name"]["gates_pass"]


def test_invalid_json_flags_everything():
    v = validate_extraction(None, SRC)
    assert not any(x["gates_pass"] for x in v.values())


def test_parse_json_handles_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json("nope") is None


def test_select_text_caps_and_orders():
    secs = {"description": "d" * 9000, "boxed_warning": "WARNING " * 10, "indications_and_usage": "i" * 5000}
    text, segs = select_text(secs)
    assert len(text) <= TOTAL_CAP + 10
    assert segs[0][0] == "boxed_warning"
    name, a, b = segs[1]
    assert text[a:b] == "d" * 4000 and name == "description"


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.n = replies, 0

    def invoke(self, messages):
        r = self.replies[min(self.n, len(self.replies) - 1)]
        self.n += 1
        return AIMessage(content=r, usage_metadata={"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200})


def test_graph_end_to_end_with_summary_and_ledger(tmp_path):
    ledger = UsageLedger(tmp_path / "u.jsonl", 1.0, 1.0, 5.0, "fake")
    llm = FakeLLM([json.dumps(good()), "Check pharm_class: nothing is quoted."])
    g = build_graph(llm, ledger, {"x": {"indications_and_usage": SRC}})
    out = g.invoke({"set_id": "x", "min_conf": "medium", "do_summary": True})
    assert out["decisions"]["route"]["approved"] and not out["decisions"]["pharm_class"]["approved"]
    assert out["summary"].startswith("Check pharm_class")
    assert [u["tag"] for u in out["usage"]] == ["extract", "summarize"]
    assert ledger.total() > 0


def test_graph_skips_summary_when_nothing_queued(tmp_path):
    ledger = UsageLedger(tmp_path / "u.jsonl", 1.0, 1.0, 5.0, "fake")
    e = good()
    e["pharm_class"] = field(["Phosphodiesterase"], ["Sildenafil citrate tablets"])   # not in quote -> flagged
    llm = FakeLLM([json.dumps(e)])
    g = build_graph(llm, ledger, {"x": {"indications_and_usage": SRC}})
    assert "summary" not in g.invoke({"set_id": "x", "min_conf": "low", "do_summary": False})
