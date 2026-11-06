"""The Phase 3 LangGraph workflow, one node per step:

    retrieve -> extract -> validate -> route -> (summarize, only if something was queued) -> end

State is a plain dict. The graph never sees the gold values; scoring happens outside it.
"""
import json
import re
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.extract.retrieve import select_text
from src.extract.scoring import FIELDS, ROUTES
from src.extract.validate import route_decisions, validate_extraction

EXTRACT_SYSTEM = f"""You extract structured fields from the text of an FDA prescription drug label.
Use ONLY the label text you are given. For every field give the value, one or more supporting QUOTES, and your confidence.

Rules for quotes: copy them EXACTLY, character for character, from the label text (3 to 300 characters each, no edits,
no ellipses). A quote must contain the words that justify the value.
If the text does not state something, use null / [] / false with no quotes. Do not use outside knowledge.

Fields:
- generic_name: the generic (non-brand) drug name as the label writes it, string.
- active_ingredients: list of the active ingredient names as written in the label (for example "sildenafil citrate").
- is_combination: true if the product has two or more different active ingredients, else false.
- route: how the product is given. Choose exactly one of: {json.dumps(ROUTES)}. null if the text gives no clue.
- has_boxed_warning: true only if the text contains a boxed warning (a prominent "WARNING:" block at the top), else false.
- pharm_class: pharmacologic class names ONLY if the label text itself names a class (for example "Nonsteroidal Anti-inflammatory Drug").
  Otherwise [].
- confidence for each field: "high" (stated plainly), "medium" (needs some interpretation), or "low" (a guess).

Reply with ONLY a JSON object, no other text, of the form:
{{"generic_name": {{"value": ..., "quotes": [...], "confidence": "high"}}, "active_ingredients": {{...}}, "is_combination": {{...}},
 "route": {{...}}, "has_boxed_warning": {{...}}, "pharm_class": {{...}}}}"""

SUMMARY_SYSTEM = """You write a short note (at most three sentences) for a human reviewer. You are given the fields of one drug
label extraction that were flagged, each with its extracted value, the quote offered as evidence, and why it was flagged.
State what needs checking and cite the evidence quote in double quotes. Do not add facts that are not in the input."""


class State(TypedDict, total=False):
    set_id: str
    min_conf: str
    do_summary: bool
    sections: dict
    source_text: str
    segments: list
    raw_reply: str
    extraction: Any
    validation: dict
    decisions: dict
    summary: str
    usage: list
    timings_ms: dict


def parse_json(text):
    """Pull the JSON object out of the model reply (it may be wrapped in code fences). None if not valid."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except ValueError:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except ValueError:
                return None
    return None


def build_graph(llm, ledger, sections_by_id):
    """llm: a LangChain chat model. ledger: UsageLedger (hard cap). sections_by_id: {set_id: {section: text}}."""
    from langchain_core.messages import HumanMessage, SystemMessage

    def call(system, human, tag, state):
        ledger.check(est_input_tokens=3500, est_output_tokens=700)
        t0 = time.perf_counter()
        msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        ms = (time.perf_counter() - t0) * 1000
        u = getattr(msg, "usage_metadata", None) or {}
        cost = ledger.record(u.get("input_tokens", 0), u.get("output_tokens", 0), tag=tag)
        state.setdefault("usage", []).append({"tag": tag, "input_tokens": u.get("input_tokens", 0),
                                              "output_tokens": u.get("output_tokens", 0), "cost_usd": round(cost, 6),
                                              "ms": round(ms, 1)})
        content = msg.content
        if isinstance(content, list):
            content = "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
        return content

    def retrieve(state):
        text, segments = select_text(sections_by_id[state["set_id"]])
        return {"source_text": text, "segments": segments, "usage": []}

    def extract(state):
        reply = call(EXTRACT_SYSTEM, f"Label text:\n\n{state['source_text']}", "extract", state)
        return {"raw_reply": reply, "extraction": parse_json(reply), "usage": state["usage"]}

    def validate(state):
        return {"validation": validate_extraction(state["extraction"], state["source_text"])}

    def route(state):
        return {"decisions": route_decisions(state["validation"], state.get("min_conf", "low"))}

    def after_route(state):
        queued = [f for f, d in state["decisions"].items() if not d["approved"]]
        return "summarize" if queued and state.get("do_summary") else END

    def summarize(state):
        lines = []
        for f in FIELDS:
            d, v = state["decisions"][f], state["validation"][f]
            if not d["approved"]:
                lines.append(f"- {f}: value={json.dumps(v['value'])}; quote={json.dumps(v['quotes'][:2])}; "
                             f"flagged because: {'; '.join(d['queue_reasons'])}")
        return {"summary": call(SUMMARY_SYSTEM, "\n".join(lines), "summarize", state).strip(), "usage": state["usage"]}

    g = StateGraph(State)
    for name, fn in [("retrieve", retrieve), ("extract", extract), ("validate", validate),
                     ("route", route), ("summarize", summarize)]:
        g.add_node(name, fn)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "extract")
    g.add_edge("extract", "validate")
    g.add_edge("validate", "route")
    g.add_conditional_edges("route", after_route, {"summarize": "summarize", END: END})
    g.add_edge("summarize", END)
    return g.compile()
