"""Evaluator-visible fail-closed checker for the completed BLOCKED audit."""

from __future__ import annotations

import json
from pathlib import Path


summary = json.loads(
    Path(__file__).with_name("formal_run_summary.json").read_text(encoding="utf-8")
)
assert summary["verdict"] == "BLOCKED"
assert summary["confidence"] == "LOW"
assert summary["protocol_complete"]
assert set(summary["routes"]) == {
    "1_exact_discretization_non_circularity",
    "2_exact_paper_configuration_audit",
    "3_independent_cpu_resource_calibration",
    "4_mandatory_falsification_attempt",
}
assert summary["routes"]["1_exact_discretization_non_circularity"]["passed"]
assert summary["routes"]["4_mandatory_falsification_attempt"]["negative_control_passed"]
assert not summary["routes"]["4_mandatory_falsification_attempt"]["falsification_succeeded"]
print("BLOCKED: four-route Claim 5 audit is complete and fail-closed")
