"""Answer service tests: fake query service + fake LLM, so no network, no AWS, no cost."""
import json

import httpx
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src.generation.usage import BudgetExceeded, UsageLedger, cost_usd
from src.services.answer import create_app

CHUNKS = [{"chunk_id": "aaaaaaaa-1111:0", "text": "Warfarin overdose causes bleeding.", "drug": "Warfarin",
           "route": "ORAL", "section": "overdosage", "set_id": "aaaaaaaa-1111", "start": 0, "end": 10, "score": 0.9}]


def fake_query_service():
    def handler(request):
        return httpx.Response(200, json={"question": "q", "results": CHUNKS, "timings_ms": {"wall": 12.0}})
    return httpx.Client(transport=httpx.MockTransport(handler))


def fake_llm(text):
    def run(prompt_value):
        # prove the prompt carries the context and the rules
        s = prompt_value.to_string()
        assert "aaaaaaaa-1111:0" in s and "Not enough information" in s
        return AIMessage(content=text, usage_metadata={"input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050})
    return RunnableLambda(run)


def make(tmp_path, text, budget=1.0):
    ledger = UsageLedger(tmp_path / "usage.jsonl", budget, 1.0, 5.0, "fake-model")
    return TestClient(create_app(llm=fake_llm(text), ledger=ledger, http_client=fake_query_service())), ledger


def test_answer_cites_retrieved_chunks_and_logs_cost(tmp_path):
    with make(tmp_path, "Overdose causes bleeding [aaaaaaaa-1111:0].")[0] as c:
        body = c.post("/answer", json={"question": "overdose of warfarin?"}).json()
    assert body["citations"] == ["aaaaaaaa-1111:0"] and body["invalid_citations"] == [] and body["answerable"]
    assert body["usage"] == {"input_tokens": 1000, "output_tokens": 50, "cost_usd": 0.00125}
    assert body["end_to_end_ms"] >= 0 and body["search_ms"] == 12.0


def test_invented_citation_is_flagged_and_refusal_is_detected(tmp_path):
    with make(tmp_path, "Made up [bbbbbbbb-2222:9].")[0] as c:
        body = c.post("/answer", json={"question": "x"}).json()
    assert body["invalid_citations"] == ["bbbbbbbb-2222:9"] and body["citations"] == []
    with make(tmp_path, "Not enough information in the retrieved labels.")[0] as c:
        assert c.post("/answer", json={"question": "x"}).json()["answerable"] is False


def test_budget_cap_blocks_calls(tmp_path):
    client, ledger = make(tmp_path, "ok [aaaaaaaa-1111:0]", budget=0.001)
    with client as c:
        assert c.post("/answer", json={"question": "x"}).status_code == 429
    assert not (tmp_path / "usage.jsonl").exists()                 # nothing was spent


def test_cost_formula_and_ledger_total(tmp_path):
    assert cost_usd(1_000_000, 1_000_000, 1.0, 5.0) == 6.0
    ledger = UsageLedger(tmp_path / "u.jsonl", 1.0, 1.0, 5.0, "m")
    ledger.record(1000, 100)
    ledger.record(2000, 0)
    assert abs(ledger.total() - (0.0015 + 0.002)) < 1e-9
    assert json.loads((tmp_path / "u.jsonl").read_text().splitlines()[0])["input_tokens"] == 1000
