"""Phase 4 scoring of Qwen outputs (PLAN.md, M4). Uses the frozen Phase 3 scoring rules in src/extract/scoring.py.

Output counts as *parsed* when it is a JSON object. Malformed values are read leniently and identically for every
model (see PLAN.md addendum): a one-string list becomes a one-element list, non-booleans in boolean fields are absent.
"""
import random

from src.extract.graph import parse_json
from src.extract.scoring import BOOL_FIELDS, FIELDS, LIST_FIELDS, f1_from_counts, score_label

FIELDS5 = [f for f in FIELDS if f != "pharm_class"]


def parse_reply(reply):
    obj = parse_json(reply or "")
    return obj if isinstance(obj, dict) else None


def values_from(obj):
    """{field: raw value} from a parsed object, with the lenient coercions. None when not parsed."""
    if obj is None:
        return None
    out = {}
    for f in FIELDS:
        e = obj.get(f)
        v = e.get("value") if isinstance(e, dict) else None
        if f in LIST_FIELDS:
            if isinstance(v, str):
                v = [v]
            v = [str(x) for x in v if isinstance(x, (str, int, float))] if isinstance(v, list) else []
        elif f in BOOL_FIELDS:
            v = v if isinstance(v, bool) else None
        elif v is not None and not isinstance(v, str):
            v = str(v)
        out[f] = v
    return out


def counts(values, label, fields):
    """(tp, fp, fn) summed over `fields` for one label; values None = unparsed = every item missed."""
    s = score_label(values, label)
    tp = sum(s[f]["items"][0] for f in fields)
    fp = sum(s[f]["items"][1] for f in fields)
    fn = sum(s[f]["items"][2] for f in fields)
    return tp, fp, fn


def micro_f1(per_label):
    tp, fp, fn = (sum(c[i] for c in per_label) for i in range(3))
    return f1_from_counts(tp, fp, fn)["f1"]


def bootstrap(per_a, per_b, n=2000, seed=42):
    """Paired bootstrap over labels. per_a / per_b: per-label (tp,fp,fn) for base and adapter (same order).
    Returns 95% intervals for the absolute and relative improvement of b over a."""
    rng = random.Random(seed)
    idx = range(len(per_a))
    absd, rel = [], []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        fa, fb = micro_f1([per_a[i] for i in s]), micro_f1([per_b[i] for i in s])
        absd.append(fb - fa)
        if fa > 0:
            rel.append((fb - fa) / fa)
    q = lambda xs: [round(sorted(xs)[int(0.025 * len(xs))], 4), round(sorted(xs)[int(0.975 * len(xs)) - 1], 4)]
    return {"abs_ci95": q(absd), "rel_ci95": q(rel)}
