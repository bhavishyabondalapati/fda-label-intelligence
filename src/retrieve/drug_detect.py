"""Find the drug (and route) a question is about, so retrieval can be restricted to its labels.

Confident = a known drug name from the corpus appears in the question as whole words. We take the
LONGEST such name ("lidocaine hydrochloride" beats "lidocaine"). If nothing matches, return None
and the search stays unfiltered. A route is read from a "(route)" in the question, if it is one of that drug's routes.
"""
import re

from src.eval.gold import norm_name

MIN_NAME_CHARS = 4
# words that appear in question templates; a "drug" equal to one of these is not a real detection
STOP = {"use", "used", "dose", "dosage", "side", "effects", "warning", "warnings", "risk", "risks", "storage",
        "oral", "work", "body", "form", "name", "pain", "water", "air", "base", "plus", "forte", "ultra"}


class DrugDetector:
    def __init__(self, labels):
        """labels: iterable of label dicts (drug_name, route). Names come from the indexed corpus only."""
        self.routes = {}
        for lab in labels:
            self.routes.setdefault(norm_name(lab["drug_name"]), set()).add(lab["route"].strip().lower())
        self.names = {n for n in self.routes if len(n) >= MIN_NAME_CHARS and n not in STOP}
        self.max_words = max((len(n.split()) for n in self.names), default=1)

    def detect(self, question):
        """-> (drug_norm, route or None) or None."""
        groups = [g.strip().lower() for g in re.findall(r"\(([^()]+)\)", question)]   # "(route)" sits right after the name
        words = norm_name(question).split()
        best = None
        for n in range(min(self.max_words, len(words)), 0, -1):          # longest candidate first
            for i in range(len(words) - n + 1):
                cand = " ".join(words[i:i + n])
                if cand in self.names:
                    best = cand
                    break
            if best:
                break
        if not best:
            return None
        route = next((g for g in groups if g in self.routes[best]), None)   # only a real route of this drug counts
        return best, route
