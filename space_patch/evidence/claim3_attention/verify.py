"""Evaluator-visible fail-closed checker for Claim 3."""

from __future__ import annotations

import json
from pathlib import Path


summary = json.loads(
    Path(__file__).with_name("formal_run_summary.json").read_text(encoding="utf-8")
)
certificate = json.loads(
    Path(__file__).with_name("structural_certificate.json").read_text(encoding="utf-8")
)
assert certificate["passed"]
assert summary["max_relative_rate_residual"] < 2e-12
assert summary["max_rope_block_qk_rate_abs"] < 2e-12
assert summary["max_vo_matrix_rate_fro"] < 2e-11
assert summary["min_nonconserved_control_rate_abs"] > 1e-7
assert summary["all_checks_passed"]
print("VERIFIED: MHA/RoPE structural certificates, audit, and control pass")
