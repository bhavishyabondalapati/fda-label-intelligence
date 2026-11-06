"""Token and cost bookkeeping for Bedrock calls, with a hard spending cap.

Every call appends one line to a JSONL ledger (results/bedrock_usage.jsonl). Before each call we
check the running total against the budget and refuse to call if it would be exceeded.
"""
import json
import time
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


def cost_usd(input_tokens, output_tokens, price_in_per_m, price_out_per_m):
    return input_tokens * price_in_per_m / 1e6 + output_tokens * price_out_per_m / 1e6


class UsageLedger:
    def __init__(self, path, budget_usd, price_in_per_m, price_out_per_m, model_id):
        self.path, self.budget = Path(path), budget_usd
        self.price_in, self.price_out, self.model_id = price_in_per_m, price_out_per_m, model_id

    def total(self):
        if not self.path.exists():
            return 0.0
        return sum(json.loads(line)["cost_usd"] for line in self.path.read_text().splitlines() if line.strip())

    def check(self, est_input_tokens=3000, est_output_tokens=400):
        """Raise before the call if the worst-case next call would pass the cap."""
        worst = cost_usd(est_input_tokens, est_output_tokens, self.price_in, self.price_out)
        if self.total() + worst > self.budget:
            raise BudgetExceeded(f"Bedrock budget ${self.budget:.2f} would be exceeded (spent ${self.total():.4f})")

    def record(self, input_tokens, output_tokens, tag=""):
        c = cost_usd(input_tokens, output_tokens, self.price_in, self.price_out)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": self.model_id, "tag": tag,
                                "input_tokens": input_tokens, "output_tokens": output_tokens,
                                "cost_usd": round(c, 6)}) + "\n")
        return c
