"""Fails if the frozen eval set or hit rule (git tag eval-v1) is edited. See PLAN.md."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_eval_v1_files_unchanged():
    lock = {}
    for line in (ROOT / "data/eval/eval-v1.lock").read_text().splitlines():
        if line and not line.startswith("#"):
            path, digest = line.split()
            lock[path] = digest
    assert set(lock) == {"data/eval/gold_questions.jsonl", "src/eval/metrics.py"}
    for path, digest in lock.items():
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        assert actual == digest, f"{path} changed after eval-v1 was frozen"
