"""Fail-closed checker for the published Theorem 4.2 evidence."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
scale = json.loads((root / "model_scale_summary.json").read_text())
proof = json.loads((root / "structural_certificate.json").read_text())
passed = (
    proof["exponents"]["A_column"]
    + proof["exponents"]["SiLU_Bx"]
    + proof["exponents"]["C_row_x"]
    == 0
    and scale["max_rate_relative_to_term_scale"] < 5e-13
    and scale["min_seed_nonconserved_control_rate_abs"] > 1e-8
    and scale["passed"] is True
)
print(json.dumps({"claim": "C2", "passed": passed}, indent=2))
raise SystemExit(0 if passed else 1)

