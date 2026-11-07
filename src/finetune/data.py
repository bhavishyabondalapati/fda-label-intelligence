"""Build QLoRA training examples from openFDA gold + label section text (Phase 4, PLAN.md "M4 pre-registration").

The model sees exactly the Phase 3 prompt. The target is the Phase 3 JSON (value, quotes, confidence per field).
Gold has no quotes, so quotes are BUILT BY CODE: find the gold value in the source text and cut a short window
around it. No language model is involved. `pharm_class` tokens are marked for loss masking when the gold is
unreliable (empty, or the class phrase is not in the text).
"""
import json
import re

from src.extract.graph import EXTRACT_SYSTEM
from src.extract.scoring import FIELDS, ROUTES, name_key, route_key
from src.extract.validate import QUOTE_MAX, QUOTE_MIN, ROUTE_CUES, quote_ok

WINDOW = 60          # characters of context on each side of the match


def user_message(source_text):
    """The Phase 3 human message, unchanged."""
    return f"Label text:\n\n{source_text}"


def squash(text):
    """Collapse all whitespace to single spaces, KEEPING case (quotes are checked case-insensitively)."""
    return " ".join(str(text).split())


def window_around(src, start, end):
    """A quote that contains src[start:end] plus some context, cut at word boundaries, 3 to 300 characters."""
    a, b = max(0, start - WINDOW), min(len(src), end + WINDOW)
    if a > 0:                                     # move forward to the next word start
        nxt = src.find(" ", a)
        a = nxt + 1 if 0 <= nxt < start else a
    if b < len(src):                              # move back to the previous word end
        prv = src.rfind(" ", end, b)
        b = prv if prv > end else b
    q = src[a:b].strip()
    return q if QUOTE_MIN <= len(q) <= QUOTE_MAX else None


def find_quote(src, needle):
    """First occurrence of `needle` (case-insensitive) in the squashed source -> a verified quote, or None."""
    needle = squash(needle)
    if len(needle) < 2:
        return None
    i = src.lower().find(needle.lower())
    if i < 0:
        return None
    q = window_around(src, i, i + len(needle))
    return q if q and quote_ok(q, src.lower()) else None


def find_route_quote(src, route):
    """Quote around the first Phase 3 cue word for this route, or None."""
    low = src.lower()
    for cue in ROUTE_CUES.get(route, []):
        m = re.search(r"(?<![a-z])" + re.escape(cue) + (r"" if len(cue) > 4 else r"(?![a-z])"), low)
        if m:
            q = window_around(src, m.start(), m.end())
            if q and quote_ok(q, low):
                return q
    return None


def clean_class(c):
    return re.sub(r"\s*\[[^\]]*\]\s*$", "", str(c)).strip()


def entry(value, quotes, conf="high"):
    return {"value": value, "quotes": quotes, "confidence": conf}


def build_example(label, sections, source_text):
    """-> dict with system, user, completion, mask_spans (char spans of `completion` to drop from the loss), stats."""
    src = squash(source_text)
    ingredients = [s.lower() for s in label["substance_name"]]
    generic = label["generic_name"].lower()
    stats = {}

    q_generic = find_quote(src, generic)
    q_ing = [q for q in (find_quote(src, i) for i in ingredients) if q]
    is_combo = len({name_key(i) for i in ingredients}) >= 2

    rk = route_key(label["route"])
    route_val = next((r for r in ROUTES if route_key(r) == rk), "OTHER") if rk else None
    q_route = find_route_quote(src, route_val) if route_val else None

    has_box = "boxed_warning" in label["sections"]
    q_box = None
    if has_box and sections.get("boxed_warning"):
        first = squash(sections["boxed_warning"])[:160]
        first = first[:first.rfind(" ")] if len(first) == 160 and " " in first else first
        q_box = first if quote_ok(first, src.lower()) else None

    classes = [clean_class(c) for c in label["pharm_class_epc"]]
    q_cls = [find_quote(src, c) for c in classes]
    cls_quotes = [q for q in q_cls if q]
    unmasked_cls = bool(classes) and all(q_cls)

    entries = {
        "generic_name": entry(generic, [q_generic] if q_generic else []),
        "active_ingredients": entry(ingredients, q_ing),
        "is_combination": entry(is_combo, []),
        "route": entry(route_val, [q_route] if q_route else []),
        "has_boxed_warning": entry(has_box, [q_box] if q_box else []),
        "pharm_class": entry(classes, cls_quotes),
    }
    parts, mask_spans, pos = [], [], 1                     # pos counts characters; the opening brace is 1 char
    for k, f in enumerate(FIELDS):
        key = ("" if k == 0 else ", ") + json.dumps(f) + ": "
        body = json.dumps(entries[f], ensure_ascii=False)
        if f == "pharm_class" and not unmasked_cls:
            cut = body.index(', "confidence"')
            mask_spans.append([pos + len(key), pos + len(key) + cut])
        parts.append(key + body)
        pos += len(key) + len(body)
    completion = "{" + "".join(parts) + "}"

    stats = {"quote_generic": bool(q_generic), "quote_ingredients": len(q_ing) == len(ingredients) and bool(ingredients),
             "quote_route": bool(q_route), "quote_boxed": (not has_box) or bool(q_box),
             "pharm_unmasked": unmasked_cls, "route_other": route_val == "OTHER"}
    return {"system": EXTRACT_SYSTEM, "user": user_message(source_text), "completion": completion,
            "mask_spans": mask_spans, "stats": stats}
