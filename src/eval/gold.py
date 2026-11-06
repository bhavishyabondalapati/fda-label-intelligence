"""Build the gold retrieval questions from section text. No LLM, fully deterministic.

Each question asks about one section of one label, and records the gold
(set_id, section). A retrieval system "gets it right" if it returns text from
that section of that label (see src/eval/metrics.py).
"""
import math
import random
import re
from collections import Counter

# Question templates per section. Only sections with a clear, distinct meaning
# are used, so a single gold section is a fair answer.
TEMPLATES = {
    "indications_and_usage": [
        "What is {d} indicated for?",
        "What conditions is {d} used to treat?",
    ],
    "contraindications": [
        "What are the contraindications for {d}?",
        "When should {d} not be used?",
    ],
    "dosage_and_administration": [
        "What is the recommended dosage of {d}?",
        "How should {d} be administered?",
    ],
    "adverse_reactions": [
        "What are the adverse reactions reported with {d}?",
        "What side effects does {d} cause?",
    ],
    "drug_interactions": [
        "What drug interactions does {d} have?",
        "Which medications interact with {d}?",
    ],
    "overdosage": [
        "What happens in an overdose of {d}?",
        "How is an overdose of {d} treated?",
    ],
    "description": [
        "What is the chemical description of {d}?",
        "What are the ingredients and structure of {d}?",
    ],
    "how_supplied": [
        "How is {d} supplied?",
        "What strengths and package sizes does {d} come in?",
    ],
    "mechanism_of_action": [
        "What is the mechanism of action of {d}?",
        "How does {d} work in the body?",
    ],
    "boxed_warning": [
        "What is the boxed warning for {d}?",
        "What serious risks does {d} carry a boxed warning for?",
    ],
    "warnings_and_cautions": [
        "What are the warnings and precautions for {d}?",
        "What safety warnings apply to {d}?",
    ],
    "storage_and_handling": [
        "How should {d} be stored?",
        "What are the storage and handling requirements for {d}?",
    ],
}
MIN_SECTION_CHARS = 300  # gold section must have enough text to be answerable


def norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def name_counts(labels):
    """How many labels use each normalized brand/generic name. Used to spot ambiguity."""
    counts = Counter()
    for lab in labels:
        for n in {norm_name(lab["brand_name"]), norm_name(lab["generic_name"])} - {""}:
            counts[n] += 1
    return counts


def build_gold(labels, sections_by_label, n_questions=200, seed=42):
    """labels: list of label meta dicts. sections_by_label: set_id -> {section: text}."""
    rng = random.Random(seed)
    counts = name_counts(labels)
    pair_counts = Counter((norm_name(lab["drug_name"]), norm_name(lab["manufacturer"])) for lab in labels)

    pool = sorted((lab for lab in labels if lab["split"] == "retrieval_eval"),
                  key=lambda lab: lab["set_id"])  # sorted first so shuffle is reproducible
    rng.shuffle(pool)

    quota = math.ceil(n_questions / len(TEMPLATES))
    taken = Counter()
    gold = []
    for lab in pool:
        if len(gold) >= n_questions:
            break
        name = lab["drug_name"]
        disambiguated = False
        if counts[norm_name(name)] > 1:
            # Same name on several labels: add the manufacturer, but only if that is unique.
            if not lab["manufacturer"] or pair_counts[(norm_name(name), norm_name(lab["manufacturer"]))] > 1:
                continue
            name = f"{name} ({lab['manufacturer']})"
            disambiguated = True
        texts = sections_by_label[lab["set_id"]]
        options = [s for s in TEMPLATES if s in texts and len(texts[s]) >= MIN_SECTION_CHARS
                   and taken[s] < quota]
        if not options:
            continue
        section = rng.choice(sorted(options))
        template = rng.choice(TEMPLATES[section])
        taken[section] += 1
        gold.append({
            "qid": f"q{len(gold) + 1:04d}",
            "question": template.format(d=name),
            "set_id": lab["set_id"],
            "section": section,
            "drug_name": lab["drug_name"],
            "disambiguated": disambiguated,
        })
    return gold
