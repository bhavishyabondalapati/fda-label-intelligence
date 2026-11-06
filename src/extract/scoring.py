"""Extraction scoring: what counts as "correct" for each field. FROZEN before the Phase 3 train run.

The rules are written in PLAN.md (section "M2 pre-registration"). This module is the only place that
decides whether a predicted value matches the gold value, so the numbers cannot drift with the outputs.

Gold comes from the openFDA structured data (labels.jsonl). The model never sees it.
"""
import re

FIELDS = ["generic_name", "active_ingredients", "is_combination", "route", "has_boxed_warning", "pharm_class"]
LIST_FIELDS = {"active_ingredients", "pharm_class"}
BOOL_FIELDS = {"is_combination", "has_boxed_warning"}

# Allowed route values (every route seen >= 5 times in extraction_train, plus OTHER). Frozen.
ROUTES = ["ORAL", "INTRAVENOUS", "TOPICAL", "INTRAMUSCULAR", "OPHTHALMIC", "RESPIRATORY (INHALATION)",
          "SUBCUTANEOUS", "EPIDURAL", "INTRA-ARTICULAR", "RECTAL", "INFILTRATION", "NASAL", "TRANSDERMAL",
          "SUBLINGUAL", "AURICULAR (OTIC)", "CUTANEOUS", "SUBMUCOSAL", "DENTAL", "VAGINAL", "OTHER"]

# Salt / hydrate / ester words removed from both sides before comparing drug names. Frozen.
SALT_WORDS = frozenset("""hydrochloride dihydrochloride hcl sodium potassium calcium magnesium chloride sulfate bisulfate
tartrate bitartrate acetate phosphate succinate maleate citrate fumarate hydrobromide bromide monohydrate dihydrate
trihydrate hydrate anhydrous tromethamine hyclate propionate dipropionate mesylate besylate carbonate bicarbonate
oxalate medoxomil mononitrate nitrate fluoride pamoate benzoate lactate gluconate decanoate valerate furoate olamine
etabonate proxetil acetonide dimeglumine tablets tablet usp""".split())

# AMENDMENT 2026-11-06 (after the train run, before the test run): dosage-form words are also ignored in names,
# on BOTH sides. A short fixed list, chosen from the train errors ("hydrocortisone cream" vs "hydrocortisone").
FORM_WORDS = frozenset("""cream gel ointment solution injection ophthalmic oral topical er xr extended release for
suspension capsule capsules lotion spray""".split())
IGNORED_NAME_WORDS = SALT_WORDS | FORM_WORDS


def normalize(s):
    """lowercase; every non-alphanumeric character becomes a space; whitespace collapsed."""
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def name_key(s):
    """Salt- and dosage-form-insensitive key of a drug name: the set of its words minus salt words, form words and 'and'.
    If nothing is left (e.g. 'sodium chloride'), keep the words (minus 'and')."""
    words = [w for w in normalize(s).split() if w != "and"]
    kept = [w for w in words if w not in IGNORED_NAME_WORDS]
    return frozenset(kept or words)


def strict_key(s):
    """Same as normalize but ignores word order and 'and' (no salt stripping)."""
    return frozenset(w for w in normalize(s).split() if w != "and")


def class_key(s):
    """Pharmacologic class without its trailing '[EPC]'-style suffix, normalized."""
    return normalize(re.sub(r"\s*\[[^\]]*\]\s*$", "", str(s)))


def route_key(s):
    return normalize(s) if s and str(s).strip() else None


def gold_fields(label):
    """The six gold values for one label (a labels.jsonl record). Lists are sets of keys."""
    ingredients = {name_key(s) for s in label["substance_name"]}
    return {
        "generic_name": name_key(label["generic_name"]),
        "active_ingredients": ingredients,
        "is_combination": len(ingredients) >= 2,
        "route": route_key(label["route"]),
        "has_boxed_warning": "boxed_warning" in label["sections"],
        "pharm_class": {class_key(c) for c in label["pharm_class_epc"]},
    }


def pred_fields(values):
    """Turn the model's raw values {field: value} into the same key form as gold_fields.
    Missing / null scalars -> None; missing / null lists -> empty set."""
    v = values or {}
    g = v.get("generic_name")
    ing = v.get("active_ingredients") or []
    pc = v.get("pharm_class") or []
    return {
        "generic_name": name_key(g) if g else None,
        "active_ingredients": {name_key(i) for i in ing if i},
        "is_combination": v.get("is_combination") if isinstance(v.get("is_combination"), bool) else None,
        "route": route_key(v.get("route")),
        "has_boxed_warning": v.get("has_boxed_warning") if isinstance(v.get("has_boxed_warning"), bool) else None,
        "pharm_class": {class_key(c) for c in pc if c},
    }


def field_correct(field, pred, gold):
    """pred / gold are one entry each of pred_fields() / gold_fields()."""
    return pred == gold


def field_items(field, pred, gold):
    """(tp, fp, fn) for micro-averaged F1. Lists count each element; scalars count one item."""
    if field in LIST_FIELDS:
        return len(pred & gold), len(pred - gold), len(gold - pred)
    if pred is None and gold is None:
        return 0, 0, 0
    if pred is None:
        return 0, 0, 1
    if gold is None:
        return 0, 1, 0
    return (1, 0, 0) if pred == gold else (0, 1, 1)


def strict_correct(field, raw_pred, label):
    """Secondary: names compared WITHOUT salt stripping (only for the two name fields)."""
    if field == "generic_name":
        return bool(raw_pred) and strict_key(raw_pred) == strict_key(label["generic_name"])
    if field == "active_ingredients":
        return {strict_key(i) for i in (raw_pred or [])} == {strict_key(s) for s in label["substance_name"]}
    raise ValueError(field)


def f1_from_counts(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def score_label(values, label):
    """Per-field correctness and F1 counts for one label."""
    gold, pred = gold_fields(label), pred_fields(values)
    out = {}
    for f in FIELDS:
        out[f] = {"correct": field_correct(f, pred[f], gold[f]), "items": field_items(f, pred[f], gold[f])}
    return out
