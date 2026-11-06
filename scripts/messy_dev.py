"""Messy-query robustness check on DEV labels (never eval-v1).

Builds 20 queries from dev gold items: 8 misspelled drug names, 8 brand-name instead of the
indexed drug name, 4 with no drug name at all (behaviour only: no single gold drug exists).
Scores the scorable 16 with the same hit rule, for: baseline, headers+rerank (no filter), and the
chosen headers+filter+rerank. Writes data/eval/messy_dev_questions.jsonl and results/messy_dev.json
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.argv = sys.argv[:1]
from scripts.ablate import Corpus, search, load_model, embed_texts, load_reranker  # noqa: E402
from src.eval.gold import TEMPLATES, norm_name  # noqa: E402
from src.eval.score import load_context, read_jsonl, score_rankings  # noqa: E402
from src.retrieve.drug_detect import DrugDetector  # noqa: E402
from src.services import common  # noqa: E402

NO_DRUG = ["What is the boxed warning for the blood pressure pill my doctor gave me?",
           "What side effects does that new antibiotic cause?",
           "How should my insulin pen be stored?",
           "What happens if someone takes too much of a painkiller?"]


def misspell(name, rng):
    w = list(name)
    pos = [i for i in range(2, len(w) - 1) if w[i].isalpha()]
    i = rng.choice(pos)
    if rng.random() < .5:
        del w[i]                      # "naproxen" -> "naprxen"
    else:
        w[i], w[i + 1] = w[i + 1], w[i]   # swap two neighbours
    return "".join(w)


def main():
    rng = random.Random(3)
    ctx = load_context(common.DATA_DIR)
    dev = list(read_jsonl(ROOT / "data" / "eval" / "dev_questions.jsonl"))
    rng.shuffle(dev)
    items, used = [], set()
    def pick(pred, n, kind, make):
        for g in dev:
            if len([x for x in items if x["kind"] == kind]) == n:
                break
            lab = ctx.labels[g["set_id"]]
            if g["drug_name"] in used or g["route_in_question"] or not pred(lab, g):
                continue
            used.add(g["drug_name"])
            tpl = rng.choice(TEMPLATES[g["section"]])
            items.append({**g, "kind": kind, "question": tpl.format(d=make(lab, g))})
    pick(lambda l, g: len(g["drug_name"]) >= 7 and " " not in g["drug_name"], 8, "misspelled", lambda l, g: misspell(g["drug_name"], rng))
    # "other name": ask by the generic name when the indexed drug_name is a brand (or the reverse).
    # Relevance is still judged by the indexed drug_name, so this is deliberately strict.
    alt = lambda l: next((x for x in (l.get("generic_name"), l.get("brand_name")) if x and norm_name(x) != norm_name(l["drug_name"])
                          and len(x) < 40 and "," not in x), None)
    pick(lambda l, g: alt(l) is not None, 8, "brand_vs_generic", lambda l, g: alt(l).title())
    for i, q in enumerate(NO_DRUG):
        items.append({"qid": f"n{i}", "question": q, "kind": "no_drug", "set_id": None, "section": None})
    out_q = ROOT / "data" / "eval" / "messy_dev_questions.jsonl"
    out_q.write_text("\n".join(json.dumps(x) for x in items) + "\n")

    detector = DrugDetector(ctx.labels.values())
    model, rer = load_model(), load_reranker()
    corp = {c: Corpus(c) for c in ("naive",)}["naive"]
    qv = embed_texts(model, [x["question"] for x in items], show_progress=False)
    scorable = [i for i, x in enumerate(items) if x["set_id"]]
    cfgs = {"baseline": (0, 0, 0, 0, 0), "headers+rerank (no filter)": (1, 0, 0, 0, 1), "CHOSEN headers+filter+rerank": (1, 0, 1, 0, 1)}
    report = {"n_scored": len(scorable), "configs": {}, "rows": []}
    tops = {}
    for name, cfg in cfgs.items():
        timing = {"dense": 0.0, "bm25": 0.0}
        tops[name] = [search(corp, cfg, items[i]["question"], qv[i], detector, rer, timing) for i in range(len(items))]
        rk = [[(corp.set_ids[j], corp.chunks[j]["start"], corp.chunks[j]["end"]) for j in tops[name][i]] for i in scorable]
        sc = score_rankings([items[i] for i in scorable], rk, ctx)
        by_kind = {}
        for i, f in zip(scorable, sc["rel_flags"]):
            by_kind.setdefault(items[i]["kind"], []).append(any(f[:5]))
        report["configs"][name] = {**sc["relaxed"], "by_kind_recall_at_5": {k: round(sum(v) / len(v), 3) for k, v in by_kind.items()}}
    for i, x in enumerate(items):
        det = detector.detect(x["question"])
        top = corp.chunks[tops["CHOSEN headers+filter+rerank"][i][0]]
        report["rows"].append({"kind": x["kind"], "question": x["question"], "gold_drug": x.get("drug_name"),
                               "detected": det, "chosen_top1": f"{top['drug']} / {top['section']}"})
    (ROOT / "results" / "messy_dev.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["configs"], indent=1))
    for r in report["rows"]:
        print(f"{r['kind']:10s} det={r['detected']!s:28s} gold={r['gold_drug']!s:22s} top1={r['chosen_top1'][:45]} | {r['question'][:60]}")


if __name__ == "__main__":
    main()
