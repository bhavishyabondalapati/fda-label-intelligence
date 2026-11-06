"""LangChain RAG chain: query service (retriever) -> prompt -> Bedrock Claude -> cited answer.

    question --> QueryServiceRetriever (HTTP /query) --> format chunks with their ids
             --> prompt --> ChatBedrockConverse (Claude on Bedrock) --> answer text + token usage

The answer must cite chunk ids like [setid:3]. We then CHECK the citations: every cited id must be
one of the chunks that were retrieved (anything else is flagged as an invalid citation).
"""
import re
import time
from typing import Any

import httpx
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnableParallel

NOT_ENOUGH = "Not enough information in the retrieved labels."

SYSTEM = f"""You answer questions about FDA prescription drug labels using ONLY the context chunks below.
Rules:
1. Every factual statement must end with a citation naming the chunk id in square brackets, exactly as
   written in the context, for example [1a2b3c:4]. Cite only ids that appear in the context.
2. If the context does not contain the answer (for example the drug asked about is not in the context,
   or the chunks do not cover the topic), reply with exactly: {NOT_ENOUGH}
3. Do not use outside knowledge. Keep the answer short (at most five sentences)."""

CITE = re.compile(r"\[([0-9a-fA-F\-]{8,}:\d+)\]")


class QueryServiceRetriever(BaseRetriever):
    """LangChain retriever that calls our own query service over HTTP."""
    url: str
    top_k: int = 5
    client: Any = None                     # injectable httpx client (tests / connection reuse)

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(self, query, *, run_manager=None):
        http = self.client or httpx.Client(timeout=60)
        r = http.post(f"{self.url}/query", json={"question": query, "top_k": self.top_k})
        r.raise_for_status()
        body = r.json()
        self._last_timings = body.get("timings_ms", {})
        return [Document(page_content=x["text"], metadata={k: v for k, v in x.items() if k != "text"})
                for x in body["results"]]


def format_docs(docs):
    parts = []
    for d in docs:
        m = d.metadata
        route = f", {m['route'].lower()}" if m.get("route") else ""
        parts.append(f"[{m['chunk_id']}] ({m['drug']}{route}; {m['section'].replace('_', ' ')})\n{d.page_content}")
    return "\n\n".join(parts)


def build_chain(retriever, llm):
    """Returns a LangChain runnable: question -> {"docs", "message"} (message = the LLM's AIMessage)."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("human", "Context:\n{context}\n\nQuestion: {question}")])
    answer = prompt | llm
    return (RunnableParallel(docs=retriever, question=lambda q: q)
            | RunnableLambda(lambda x: {**x, "context": format_docs(x["docs"])})
            | RunnableLambda(lambda x: {"docs": x["docs"], "message": answer.invoke(x)}))


def run_answer(chain, ledger, question, retriever=None):
    """One end-to-end answer with timing, usage logging and citation checking."""
    ledger.check()
    t0 = time.perf_counter()
    out = chain.invoke(question)
    wall = (time.perf_counter() - t0) * 1000
    msg = out["message"]
    text = StrOutputParser().invoke(msg).strip()
    usage = getattr(msg, "usage_metadata", None) or {}
    tin, tout = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    cost = ledger.record(tin, tout, tag="answer")
    retrieved = [d.metadata["chunk_id"] for d in out["docs"]]
    cited = list(dict.fromkeys(CITE.findall(text)))
    return {
        "question": question,
        "answer": text,
        "answerable": not text.startswith("Not enough information"),
        "citations": [c for c in cited if c in retrieved],
        "invalid_citations": [c for c in cited if c not in retrieved],
        "retrieved_chunk_ids": retrieved,
        "usage": {"input_tokens": tin, "output_tokens": tout, "cost_usd": round(cost, 6)},
        "end_to_end_ms": round(wall, 1),
        "search_ms": round(getattr(retriever, "_last_timings", {}).get("wall", 0.0), 1) if retriever else None,
    }
