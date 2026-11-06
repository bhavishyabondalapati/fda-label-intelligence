"""Parse openFDA drug label records into clean per-section records.

Pure functions only (no network, no files), so they are easy to test.
"""
import hashlib
import re

# The label sections we keep. openFDA stores each as a list of strings under
# a lowercase key. Everything else (package art, unclassified text) is dropped.
SECTIONS = [
    "boxed_warning",
    "indications_and_usage",
    "dosage_and_administration",
    "dosage_forms_and_strengths",
    "contraindications",
    "warnings_and_cautions",
    "warnings",
    "precautions",
    "adverse_reactions",
    "drug_interactions",
    "use_in_specific_populations",
    "pregnancy",
    "pediatric_use",
    "geriatric_use",
    "overdosage",
    "drug_abuse_and_dependence",
    "description",
    "clinical_pharmacology",
    "mechanism_of_action",
    "pharmacokinetics",
    "nonclinical_toxicology",
    "clinical_studies",
    "how_supplied",
    "storage_and_handling",
    "information_for_patients",
]

RX_PRODUCT_TYPE = "HUMAN PRESCRIPTION DRUG"  # exact value used in the openFDA bulk files
MIN_SECTION_CHARS = 100      # a section shorter than this is treated as empty
MIN_SECTIONS_PER_LABEL = 4   # skip stub labels with almost no content

# Split shares (see PLAN.md). Decided by a hash of set_id, so they never change.
RETRIEVAL_EVAL_SHARE = 0.10
EXTRACTION_TEST_SHARE = 0.10


def clean_text(parts) -> str:
    """openFDA gives a list of strings; join them and collapse whitespace."""
    if isinstance(parts, str):
        parts = [parts]
    text = " ".join(p for p in parts if isinstance(p, str))
    return re.sub(r"\s+", " ", text).strip()


def _first(openfda: dict, key: str) -> str:
    vals = openfda.get(key) or []
    return vals[0].strip() if vals else ""


def hash_fraction(seed: int, salt: str, set_id: str) -> float:
    """Deterministic number in [0, 1) for a label. Same inputs -> same output."""
    digest = hashlib.sha256(f"{seed}:{salt}:{set_id}".encode()).hexdigest()
    return int(digest[:12], 16) / 16**12


def sample_key(seed: int, set_id: str) -> float:
    """Sorting key for the fixed-seed sample: keep the labels with the smallest keys."""
    return hash_fraction(seed, "sample", set_id)


def assign_split(seed: int, set_id: str) -> str:
    u = hash_fraction(seed, "split", set_id)
    if u < RETRIEVAL_EVAL_SHARE:
        return "retrieval_eval"
    if u < RETRIEVAL_EVAL_SHARE + EXTRACTION_TEST_SHARE:
        return "extraction_test"
    return "extraction_train"


def is_human_rx(raw: dict) -> bool:
    types = (raw.get("openfda") or {}).get("product_type") or []
    return RX_PRODUCT_TYPE in types


def parse_label(raw: dict, seed: int):
    """Turn one raw openFDA label into (label_meta, [section_records]).

    Returns None if the label is not usable (not Rx, no id or name, too little text).
    """
    if not is_human_rx(raw):
        return None
    set_id = raw.get("set_id")
    openfda = raw.get("openfda") or {}
    brand = _first(openfda, "brand_name")
    generic = _first(openfda, "generic_name")
    drug_name = brand or generic
    if not set_id or not drug_name:
        return None

    sections = []
    for name in SECTIONS:
        if name in raw:
            text = clean_text(raw[name])
            if len(text) >= MIN_SECTION_CHARS:
                sections.append({"set_id": set_id, "section": name, "text": text})
    if len(sections) < MIN_SECTIONS_PER_LABEL:
        return None

    meta = {
        "set_id": set_id,
        "drug_name": drug_name,
        "brand_name": brand,
        "generic_name": generic,
        "manufacturer": _first(openfda, "manufacturer_name"),
        "route": _first(openfda, "route"),
        "substance_name": openfda.get("substance_name") or [],
        "pharm_class_epc": openfda.get("pharm_class_epc") or [],
        "effective_time": raw.get("effective_time", ""),
        "split": assign_split(seed, set_id),
        "sections": [s["section"] for s in sections],
        "n_chars": sum(len(s["text"]) for s in sections),
    }
    return meta, sections
