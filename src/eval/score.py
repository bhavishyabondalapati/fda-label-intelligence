"""Score rankings against the frozen eval-v1 gold questions.

ONE place applies the hit rule, so the local baseline and the Pinecone service are scored
identically. The rule itself lives in src/eval/metrics.py (frozen, see PLAN.md).
"""
import json
from collections import defaultdict
from pathlib import Path

from src.chunk.naive import join_label
from src.eval.gold import drug_key
from src.eval.metrics import chunk_hits_gold, chunk_hits_relevant, mrr, recall_at_k


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


class EvalContext:
    """Per-label facts the hit rule needs: drug key (name + route) and section spans."""

    def __init__(self, labels, label_spans):
        self.labels = labels                                   # set_id -> label meta
        self.label_spans = label_spans                         # set_id -> {section: (start, end)}
        self.key_of = {sid: drug_key(lab) for sid, lab in labels.items()}


def load_context(data_dir: Path) -> EvalContext:
    labels = {lab["set_id"]: lab for lab in read_jsonl(data_dir / "labels.jsonl")}
    sections = defaultdict(list)
    for s in read_jsonl(data_dir / "sections.jsonl"):
        sections[s["set_id"]].append(s)
    spans = {sid: join_label(secs)[1] for sid, secs in sections.items()}
    return EvalContext(labels, spans)


def summarize(flags):
    return {"recall_at_1": round(recall_at_k([f[:1] for f in flags]), 4),
            "recall_at_5": round(recall_at_k([f[:5] for f in flags]), 4),
            "mrr_at_10": round(mrr(flags), 4),
            "n_hits_at_5": int(sum(any(f[:5]) for f in flags))}


def score_rankings(gold, rankings, ctx: EvalContext):
    """gold: list of gold question dicts. rankings: for each question, a list of
    (set_id, start, end) chunk positions in rank order (best first, up to 10).

    Returns relaxed (primary) and strict (cross-check) summaries plus per-question flags.
    """
    rel_flags, strict_flags, per_section = [], [], defaultdict(list)
    for g, ranked in zip(gold, rankings):
        gold_key = ctx.key_of[g["set_id"]]
        gold_span = ctx.label_spans[g["set_id"]][g["section"]]
        r_row, s_row = [], []
        for set_id, start, end in ranked:
            own_span = ctx.label_spans[set_id].get(g["section"])
            r_row.append(chunk_hits_relevant(ctx.key_of[set_id], start, end, gold_key, own_span))
            s_row.append(chunk_hits_gold(set_id, start, end, g["set_id"], gold_span))
        rel_flags.append(r_row)
        strict_flags.append(s_row)
        per_section[g["section"]].append(any(r_row[:5]))
    return {
        "relaxed": summarize(rel_flags),
        "strict": summarize(strict_flags),
        "rel_flags": rel_flags,
        "strict_flags": strict_flags,
        "per_section_recall_at_5": {s: round(sum(v) / len(v), 3) for s, v in sorted(per_section.items())},
        "per_section_n": {s: len(v) for s, v in sorted(per_section.items())},
    }
