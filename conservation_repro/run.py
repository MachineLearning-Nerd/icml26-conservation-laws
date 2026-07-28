"""Fixed cumulative entrypoint for every OpenResearch experiment node."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import core
from .claim4_theorem47 import verify as verify_theorem47
from .claim2_swiglu import verify as verify_swiglu
from .claim3_attention import verify as verify_attention
from .claim1_theorem41 import verify as verify_theorem41
from .claim5_figure2 import verify as verify_figure2


def main() -> int:
    started = time.perf_counter()
    report: dict[str, object] = {
        "schema_version": 1,
        "node_role": "cumulative_all_claims_with_claim5_four_route_audit",
        "paper": "arXiv:2606.17816",
        "compute": {
            "estimate_cores": 1,
            "runtime_class": "uncertain_model_scale",
            "selected_backend": "hf",
            "selected_flavor": "cpu-upgrade",
            "logical_cpus_visible": os.cpu_count(),
            "effective_thread_limit": 1,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "claims": {},
    }

    c1_rows = []
    for seed in range(6):
        rates = core.gelu_silu_candidate_drift(seed)
        c1_rows.append({"seed": seed, **rates})
    c1_ok = all(min(row["gelu"], row["silu"]) > 1e-3 for row in c1_rows)
    report["claims"]["C1"] = {
        "status": "TOY",
        "test": "one ReLU-style candidate is not conserved; not a completeness proof",
        "rows": c1_rows,
        "check_passed": c1_ok,
    }

    c2_rows = [
        {"seed": seed, "orthogonality_residual": core.swiglu_residual(seed)}
        for seed in range(6)
    ]
    c2_ok = max(row["orthogonality_residual"] for row in c2_rows) < 1e-10
    report["claims"]["C2"] = {
        "status": "TOY",
        "rows": c2_rows,
        "check_passed": c2_ok,
    }

    c3_rows = [
        {"seed": seed, **core.attention_residual(seed)} for seed in range(5)
    ]
    c3_drift = core.attention_gd_drift(seed=1)
    c3_ok = (
        max(max(row["qk"], row["vo"]) for row in c3_rows) < 1e-10
        and c3_drift["qk"] < 0.05
        and c3_drift["vo"] < 0.05
    )
    report["claims"]["C3"] = {
        "status": "TOY",
        "scope": "standard MHA only; RoPE absent",
        "rows": c3_rows,
        "gd_drift_40_steps": c3_drift,
        "check_passed": c3_ok,
    }

    c4_rows = [
        {"seed": seed, "max_residual": core.moe_swiglu_residual(seed)}
        for seed in range(5)
    ]
    c4_ok = max(row["max_residual"] for row in c4_rows) < 1e-10
    report["claims"]["C4"] = {
        "status": "TOY",
        "scope": "dense softmax expert invariants only; sparse and sigmoid absent",
        "rows": c4_rows,
        "check_passed": c4_ok,
    }

    c5_rows = []
    for seed in range(4):
        taus, drifts, slope = core.tau2_scaling(seed)
        c5_rows.append(
            {"seed": seed, "taus": taus, "drifts": drifts, "slope": slope}
        )
    c5_ok = all(1.95 < row["slope"] < 2.05 for row in c5_rows)
    report["claims"]["C5"] = {
        "status": "TOY",
        "scope": "single small SwiGLU step; no paper dataset/model training",
        "rows": c5_rows,
        "check_passed": c5_ok,
    }

    theorem47 = verify_theorem47()
    negative_control = subprocess.run(
        [
            sys.executable,
            "-m",
            "conservation_repro.claim4_theorem47",
            "--assert-theorem",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    theorem47["negative_control"] = {
        "command": (
            "uv run --frozen --no-dev python -m "
            "conservation_repro.claim4_theorem47 --assert-theorem"
        ),
        "expected_exit_code": 1,
        "actual_exit_code": negative_control.returncode,
        "rejected_as_intended": negative_control.returncode == 1,
        "stdout_tail": negative_control.stdout.strip().splitlines()[-1],
    }
    theorem47["check_passed"] = bool(theorem47["all_checks_passed"]) and bool(
        theorem47["negative_control"]["rejected_as_intended"]
    )
    report["claims"]["C4_theorem_4_7_exact"] = theorem47

    swiglu = verify_swiglu()
    swiglu["check_passed"] = bool(swiglu["all_checks_passed"])
    report["claims"]["C2_theorem_4_2_exact"] = swiglu

    attention = verify_attention()
    attention["check_passed"] = bool(attention["all_checks_passed"])
    report["claims"]["C3_theorems_4_3_4_4_exact"] = attention

    theorem41 = verify_theorem41()
    theorem41["check_passed"] = bool(theorem41["all_checks_passed"])
    report["claims"]["C1_theorem_4_1_exact"] = theorem41

    figure2 = verify_figure2()
    figure2["check_passed"] = bool(figure2["all_checks_passed"])
    report["claims"]["C5_figure2_exact_scope"] = figure2

    all_checks = all(
        bool(claim["check_passed"]) for claim in report["claims"].values()
    )
    report["baseline_regression_passed"] = all_checks
    report["runtime_seconds"] = time.perf_counter() - started

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "baseline.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"=== EVAL.md: {report['node_role']} ===")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("=== END EVAL.md ===")
    return 0 if all_checks else 1


if __name__ == "__main__":
    raise SystemExit(main())
