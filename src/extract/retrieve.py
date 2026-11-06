"""retrieve node: pick the sections to show the model and cut them to about 3K tokens (12,000 characters).

Only label SECTION TEXT is used. The openfda block, labels.jsonl metadata and section names never reach
the model. Sections are joined with a blank line and no headings.
"""
import json
from pathlib import Path

# Which sections, in priority order, with a per-section character cap. Frozen before the train run.
PLAN = [("boxed_warning", 3000), ("description", 4000), ("indications_and_usage", 3000),
        ("how_supplied", 1500), ("clinical_pharmacology", 3000), ("dosage_and_administration", 1500)]
TOTAL_CAP = 12000


def load_sections(path, set_ids):
    """{set_id: {section: text}} for the wanted labels only."""
    want, out = set(set_ids), {}
    with open(Path(path)) as f:
        for line in f:
            d = json.loads(line)
            if d["set_id"] in want:
                out.setdefault(d["set_id"], {})[d["section"]] = d["text"]
    return out


def select_text(sections):
    """Returns (source_text, segments). segments = [(section_name, start, end)] inside source_text."""
    parts, segments, used = [], [], 0
    for name, cap in PLAN:
        text = sections.get(name)
        if not text or used >= TOTAL_CAP:
            continue
        take = text[:min(cap, TOTAL_CAP - used)]
        start = used + (2 * len(parts))          # account for the "\n\n" joiners
        parts.append(take)
        segments.append((name, start, start + len(take)))
        used += len(take)
    return "\n\n".join(parts), segments
