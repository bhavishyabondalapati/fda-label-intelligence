"""validate node: rule checks on one label's extraction. It never sees the gold values.

A field passes its GATES only if all of these hold (see PLAN.md, "M2 pre-registration"):
  1. schema valid
  2. VERBATIM-QUOTE gate (required): at least one 3-300 character quote that appears word for word in the source text
  3. value-support: the value's own words (or a cue word, for route) appear inside the quotes
  4. cross-field consistency
Absence claims (null / empty list / false) have nothing to quote, so they never pass.
Confidence is NOT used here: it is applied afterwards in route, and can only add a requirement.
"""
import re

from src.extract.scoring import FIELDS, ROUTES, name_key, normalize, class_key

CONF_RANK = {"low": 0, "medium": 1, "high": 2}
QUOTE_MIN, QUOTE_MAX = 3, 300

# Words in the quote that support each route value. Frozen before the train run. (OTHER has no cues.)
ROUTE_CUES = {
    "ORAL": ["oral", "orally", "tablet", "capsule", "by mouth", "chewable", "lozenge"],
    "INTRAVENOUS": ["intravenous", "intravenously", "infusion", "iv"],
    "TOPICAL": ["topical", "topically", "cream", "ointment", "gel", "lotion", "shampoo", "foam"],
    "INTRAMUSCULAR": ["intramuscular", "intramuscularly", "im"],
    "OPHTHALMIC": ["ophthalmic", "eye", "ocular", "conjunctiv"],
    "RESPIRATORY (INHALATION)": ["inhal", "aerosol", "nebuliz", "respiratory"],
    "SUBCUTANEOUS": ["subcutaneous", "subcutaneously", "subcut"],
    "EPIDURAL": ["epidural"], "INTRA-ARTICULAR": ["intra articular", "intraarticular", "joint"],
    "RECTAL": ["rectal", "suppositor"], "INFILTRATION": ["infiltration"], "NASAL": ["nasal"],
    "TRANSDERMAL": ["transdermal", "patch"], "SUBLINGUAL": ["sublingual"],
    "AURICULAR (OTIC)": ["otic", "ear", "auricular"], "CUTANEOUS": ["cutaneous", "skin"],
    "SUBMUCOSAL": ["submucosal", "mucosal"], "DENTAL": ["dental", "periodontal"], "VAGINAL": ["vaginal"],
}


def _squash(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def quote_ok(quote, source_squashed):
    """Verbatim: the quote (whitespace-collapsed, case-insensitive) is a substring of the source text."""
    if not isinstance(quote, str):
        return False
    q = _squash(quote)
    return QUOTE_MIN <= len(q) <= QUOTE_MAX and q in source_squashed


def _words(text):
    return set(normalize(text).split())


def _has_cue(cue_list, quote_text):
    padded = f" {normalize(quote_text)} "
    return any(f" {c} " in padded or (len(c) > 4 and c in padded) for c in cue_list)


def parse_field(entry):
    """-> (value, quotes, confidence) or None when the entry is malformed."""
    if not isinstance(entry, dict) or "value" not in entry:
        return None
    quotes = entry.get("quotes") or []
    conf = str(entry.get("confidence", "low")).lower()
    if not isinstance(quotes, list) or conf not in CONF_RANK:
        return None
    return entry["value"], quotes, conf


def validate_extraction(extraction, source_text):
    """extraction: parsed JSON dict (or None if the model's reply was not valid JSON).
    Returns {field: {"gates_pass": bool, "reasons": [str], "confidence": str, "value": ..., "quotes": [...]}}"""
    src = _squash(source_text)
    out = {f: {"gates_pass": False, "reasons": [], "confidence": "low", "value": None, "quotes": []} for f in FIELDS}
    if not isinstance(extraction, dict):
        for f in FIELDS:
            out[f]["reasons"].append("schema: reply was not valid JSON")
        return out

    for f in FIELDS:
        r = out[f]
        parsed = parse_field(extraction.get(f))
        if parsed is None:
            r["reasons"].append("schema: field missing or malformed")
            continue
        value, quotes, conf = parsed
        r["value"], r["quotes"], r["confidence"] = value, quotes, conf
        _check_field(f, value, quotes, src, r)

    _cross_field(out)
    for f in FIELDS:
        out[f]["gates_pass"] = not out[f]["reasons"]
    return out


def _check_field(f, value, quotes, src, r):
    if f == "is_combination":                   # derived field, checked in _cross_field
        if not isinstance(value, bool):
            r["reasons"].append("schema: is_combination must be true or false")
        return
    if f == "route":
        if value is not None and value not in ROUTES:
            r["reasons"].append(f"schema: route {value!r} not in the allowed list")
            return
    elif f in ("active_ingredients", "pharm_class"):
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            r["reasons"].append(f"schema: {f} must be a list of strings")
            return
    elif f == "generic_name":
        if value is not None and not isinstance(value, str):
            r["reasons"].append("schema: generic_name must be a string or null")
            return
    elif f == "has_boxed_warning":
        if not isinstance(value, bool):
            r["reasons"].append("schema: has_boxed_warning must be true or false")
            return

    absent = value in (None, False, "") or value == []
    if absent:
        r["reasons"].append("absence claim: nothing to quote, needs a reviewer")
        return

    good = [q for q in quotes if quote_ok(q, src)]
    if not good:
        r["reasons"].append("quote gate: no supporting quote found word for word in the source text"
                            if quotes else "quote gate: no supporting quote given")
        return
    if len(good) < len(quotes):
        r["reasons"].append("quote gate: at least one given quote is not in the source text")
        return
    qwords = _words(" ".join(good))
    if f == "generic_name":
        if not name_key(value) <= qwords:
            r["reasons"].append("value support: generic name words are not inside the quote")
    elif f == "active_ingredients":
        missing = [v for v in value if not name_key(v) <= qwords]
        if missing:
            r["reasons"].append(f"value support: ingredient not inside the quotes: {missing[:2]}")
    elif f == "pharm_class":
        missing = [v for v in value if not set(class_key(v).split()) <= qwords]
        if missing:
            r["reasons"].append(f"value support: class words not inside the quotes: {missing[:2]}")
    elif f == "route":
        if not _has_cue(ROUTE_CUES.get(value, []), " ".join(good)):
            r["reasons"].append("value support: quote has no wording that points to this route")
    elif f == "has_boxed_warning":
        if "warning" not in qwords:
            r["reasons"].append("value support: quote does not contain 'warning'")


def _cross_field(out):
    ing, comb, gen = out["active_ingredients"], out["is_combination"], out["generic_name"]
    if isinstance(ing["value"], list) and isinstance(comb["value"], bool):
        n = len({name_key(i) for i in ing["value"] if i})
        if comb["value"] != (n >= 2):
            comb["reasons"].append(f"consistency: is_combination={comb['value']} but {n} ingredient(s) listed")
            ing["reasons"].append("consistency: ingredient list disagrees with is_combination")
    if ing["reasons"]:                          # derived field cannot be approved if its source is not
        if not any(r.startswith("consistency") for r in comb["reasons"]):
            comb["reasons"].append("derived from active_ingredients, which was flagged")
    if isinstance(gen["value"], str) and gen["value"] and isinstance(ing["value"], list) and ing["value"]:
        g = name_key(gen["value"])
        u = frozenset().union(*[name_key(i) for i in ing["value"]])
        if not (g <= u or u <= g):
            gen["reasons"].append("consistency: generic name and ingredients name different drugs")
            ing["reasons"].append("consistency: ingredients do not match the generic name")


# AMENDMENT 2026-11-06: these fields are never auto-approved (policy). openFDA's pharmacologic class is missing for
# about 61% of labels even when the label text names a class, so the answer key cannot support auto-approval.
REVIEW_ONLY = frozenset({"pharm_class"})


def route_decisions(validation, min_conf):
    """route node logic. A field is auto-approved iff its gates passed AND its confidence >= min_conf.
    Returns {field: {"approved": bool, "queue_reasons": [...]}}. Confidence can never rescue a failed gate."""
    need = CONF_RANK[min_conf]
    res = {}
    for f, v in validation.items():
        reasons = list(v["reasons"])
        if v["gates_pass"] and CONF_RANK.get(v["confidence"], 0) < need:
            reasons.append(f"confidence {v['confidence']} is below {min_conf}")
        if f in REVIEW_ONLY:
            reasons.append("policy: this field is always reviewed by a person")
        res[f] = {"approved": not reasons, "queue_reasons": reasons}
    return res
