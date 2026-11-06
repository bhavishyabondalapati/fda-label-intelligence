"""Answer service: POST /answer {"question": "...", "top_k": 5}

  question -> LangChain: retriever (query service /query) -> Bedrock Claude -> cited answer

Returns the answer, the chunk ids it cites (checked against what was retrieved), token counts,
estimated cost and end-to-end latency (search + generation).
"""
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.generation.usage import BudgetExceeded
from src.services import common

# Bedrock settings. Region/profile are the project rules; model and prices are configurable.
BEDROCK_REGION = os.getenv("AWS_REGION", "us-east-2")
BEDROCK_PROFILE = os.getenv("AWS_PROFILE", "sky")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "")
PRICE_IN = float(os.getenv("BEDROCK_PRICE_IN_PER_M", "0"))      # USD per million input tokens
PRICE_OUT = float(os.getenv("BEDROCK_PRICE_OUT_PER_M", "0"))
BUDGET = float(os.getenv("BEDROCK_BUDGET_USD", "0.90"))          # hard cap for the whole phase (limit is $1)
LEDGER = common.ROOT / "results" / "bedrock_usage.jsonl"


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)


def make_llm():
    from langchain_aws import ChatBedrockConverse
    return ChatBedrockConverse(model=MODEL_ID, region_name=BEDROCK_REGION, credentials_profile_name=BEDROCK_PROFILE,
                               temperature=0, max_tokens=400)


def create_app(llm=None, ledger=None, http_client=None) -> FastAPI:
    from src.generation.rag import QueryServiceRetriever, build_chain, run_answer
    from src.generation.usage import UsageLedger
    state = {"llm": llm, "ledger": ledger, "http": http_client, "chain": None, "retriever": None}

    @asynccontextmanager
    async def lifespan(app):
        if state["http"] is None:
            state["http"] = httpx.Client(timeout=60)
        if state["llm"] is None:
            state["llm"] = make_llm()
        if state["ledger"] is None:
            state["ledger"] = UsageLedger(LEDGER, BUDGET, PRICE_IN, PRICE_OUT, MODEL_ID)
        yield

    app = FastAPI(title="answer-service", lifespan=lifespan)

    def chain_for(top_k):
        retriever = QueryServiceRetriever(url=common.QUERY_URL, top_k=top_k, client=state["http"])
        return retriever, build_chain(retriever, state["llm"])

    @app.get("/health")
    def health():
        return {"status": "ok", "model": MODEL_ID, "spent_usd": round(state["ledger"].total(), 4) if state["ledger"] else None}

    @app.post("/answer")
    def answer(req: AnswerRequest):
        retriever, chain = chain_for(req.top_k)
        try:
            return run_answer(chain, state["ledger"], req.question, retriever)
        except BudgetExceeded as e:
            raise HTTPException(429, str(e))
        except httpx.HTTPError as e:
            raise HTTPException(502, f"query service unavailable: {type(e).__name__}")

    return app


app = create_app()
