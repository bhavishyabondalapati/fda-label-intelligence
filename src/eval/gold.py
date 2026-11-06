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


def drug_key(label):
    """Normalized drug name + route: what makes two labels 'the same drug' for relevance."""
    return (norm_name(label["drug_name"]), label["route"].strip().lower())


def build_gold(labels, sections_by_label, n_questions=200, seed=42, split="retrieval_eval", exclude_keys=frozenset()):
    """labels: list of label meta dicts. sections_by_label: set_id -> {section: text}.

    Questions name only the drug (no manufacturer): any label with the same drug name and
    route is an equally good source (see PLAN.md amendments). A drug name that appears with
    more than one route in the corpus gets the route added to the question text, e.g.
    "Lidocaine (transdermal)", so the question says which one it means. Such a label with
    no route recorded cannot be asked about unambiguously and is skipped.

    split / exclude_keys are used only for the DEV set (Phase 2): questions come from a different
    split, and any label whose (drug, route) also appears in eval-v1's split is left out, so dev
    and eval-v1 share no drug. The defaults reproduce eval-v1 exactly.
    """
    rng = random.Random(seed)
    routes_by_name = {}
    for lab in labels:
        routes_by_name.setdefault(norm_name(lab["drug_name"]), set()).add(lab["route"].strip().lower())

    pool = sorted((lab for lab in labels if lab["split"] == split and drug_key(lab) not in exclude_keys),
                  key=lambda lab: lab["set_id"])  # sorted first so shuffle is reproducible
    rng.shuffle(pool)

    quota = math.ceil(n_questions / len(TEMPLATES))
    taken = Counter()
    gold = []
    # Pass 1 balances sections with a per-section quota. Pass 2 (no quota) only fills
    # any shortfall, because rare sections (e.g. storage_and_handling) may run out of labels.
    used = set()
    for section_quota in (quota, math.inf):
        for lab in pool:
            if len(gold) >= n_questions:
                break
            multi_route = len(routes_by_name[norm_name(lab["drug_name"])]) > 1
            if lab["set_id"] in used or (multi_route and not lab["route"].strip()):
                continue
            asked_name = f"{lab['drug_name']} ({lab['route'].strip().lower()})" if multi_route else lab["drug_name"]
            texts = sections_by_label[lab["set_id"]]
            options = [s for s in TEMPLATES if s in texts and len(texts[s]) >= MIN_SECTION_CHARS
                       and taken[s] < section_quota]
            if not options:
                continue
            section = rng.choice(sorted(options))
            template = rng.choice(TEMPLATES[section])
            taken[section] += 1
            used.add(lab["set_id"])
            gold.append({
                "qid": f"q{len(gold) + 1:04d}",
                "question": template.format(d=asked_name),
                "set_id": lab["set_id"],
                "section": section,
                "drug_name": lab["drug_name"],
                "route": lab["route"],
                "route_in_question": multi_route,
            })
    return gold
