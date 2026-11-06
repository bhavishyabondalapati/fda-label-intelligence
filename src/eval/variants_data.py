"""Load all chunks for a chunker variant, in a fixed order (labels order of sections.jsonl)."""
from collections import OrderedDict

from src.chunk.variants import naive_records, section_chunks
from src.eval.score import read_jsonl


def load_chunks(data_dir, chunker):
    labels = {lab["set_id"]: lab for lab in read_jsonl(data_dir / "labels.jsonl")}
    by_label = OrderedDict()
    for s in read_jsonl(data_dir / "sections.jsonl"):
        by_label.setdefault(s["set_id"], []).append({"section": s["section"], "text": s["text"]})
    fn = {"naive": naive_records, "section": section_chunks}[chunker]
    chunks = []
    for sid, secs in by_label.items():
        lab = labels[sid]
        chunks.extend(fn(sid, lab["drug_name"], lab["route"], secs))
    return chunks
