"""Evaluator-visible fail-closed checker for the formal Claim 1 summary."""

from __future__ import annotations

import json
from pathlib import Path


summary = json.loads(
    Path(__file__).with_name("formal_run_summary.json").read_text(encoding="utf-8")
)
exact = summary["exact_generating_function_certificate"]
finite = summary["finite_span_audit"]
assert exact["passed"]
assert all(row["exact_rank"] == row["unknowns"] for row in exact["exact_fraction_checks"])
assert exact["negative_control"]["rejected_as_intended"]
assert finite["passed"]
assert summary["all_checks_passed"]
print("VERIFIED: Claim 1 exact certificate, consequence audit, and control pass")
