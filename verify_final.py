"""Verify the standardized repository dossier and published branch surface."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_BRANCHES = {
    "audit/claim1-gelu-silu",
    "audit/claim2-swiglu-symmetry",
    "audit/claim3-attention-rope",
    "audit/claim4-normalized-sigmoid",
    "audit/claim5-figure2-availability",
    "historical/judged-baseline",
    "main",
    "release/evaluator-candidate",
    "release/space-root-package",
}
EXPECTED_COMMITS = 31
CANONICAL_IDENTITY = "MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>"
EXPECTED_STATUSES = {
    "C1": "VERIFIED_SCOPED",
    "C2": "VERIFIED_SCOPED",
    "C3": "VERIFIED_SCOPED",
    "C4": "FALSIFIED_SCOPED",
    "C5": "BLOCKED",
}
EXPECTED_OVERALL = "PARTIAL_C1_C2_C3_VERIFIED_C4_FALSIFIED_C5_BLOCKED"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"verification failed: {message}")


def published_branches() -> set[str]:
    try:
        lines = git("ls-remote", "--heads", "origin").splitlines()
    except subprocess.CalledProcessError:
        lines = []
    if lines:
        return {line.split("\t", 1)[1].removeprefix("refs/heads/") for line in lines}
    return set(git("for-each-ref", "--format=%(refname:short)", "refs/heads").splitlines())


def main() -> None:
    claims = json.loads((ROOT / "claims.json").read_text(encoding="utf-8"))
    verdicts = json.loads((ROOT / "reproduction_verdicts.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "EVIDENCE_MANIFEST.json").read_text(encoding="utf-8"))
    state = json.loads((ROOT / "AUTONOMOUS_STATE.json").read_text(encoding="utf-8"))
    logbook = json.loads((ROOT / "space_patch/logbook.json").read_text(encoding="utf-8"))

    branches = published_branches()
    require(branches == EXPECTED_BRANCHES, "published branches")
    require(not any(branch.startswith("orx/") for branch in branches), "legacy orx branch")
    require(int(git("rev-list", "--all", "--count")) == EXPECTED_COMMITS, "reachable commit count")
    identities = git("log", "--all", "--format=%an <%ae>%n%cn <%ce>").splitlines()
    require(identities and all(identity == CANONICAL_IDENTITY for identity in identities), "canonical identity")

    require(claims["overall_status"] == EXPECTED_OVERALL, "claims overall status")
    require(verdicts["overall_verdict"] == EXPECTED_OVERALL, "verdict overall status")
    require(verdicts["claim_statuses"] == EXPECTED_STATUSES, "verdict statuses")
    require(state["repository"]["expected_reachable_commits"] == EXPECTED_COMMITS, "state commit count")
    require(state["repository"]["canonical_email"] == "MachineLearning-Nerd@users.noreply.github.com", "state identity")
    require(all((ROOT / path).exists() for path in manifest["required_paths"]), "manifest paths")

    c1 = json.loads((ROOT / ".openresearch/artifacts/claim1_theorem41/raw/formal_run_summary.json").read_text())
    c2 = json.loads((ROOT / ".openresearch/artifacts/claim2_theorem42/raw/model_scale_summary.json").read_text())
    c3 = json.loads((ROOT / ".openresearch/artifacts/claim3_attention/raw/formal_run_summary.json").read_text())
    c4 = json.loads((ROOT / ".openresearch/artifacts/claim4_theorem47/raw/exact_certificate.json").read_text())
    c5 = json.loads((ROOT / ".openresearch/artifacts/claim5_figure2/raw/formal_run_summary.json").read_text())
    require(c1["verdict"] == "VERIFIED", "C1 evidence")
    require(c2["passed"] is True, "C2 evidence")
    require(c3["verdict"] == "VERIFIED", "C3 evidence")
    require(c4["verdict"] == "FALSIFIED" and c4["theorem_4_7_contradicted"] is True, "C4 evidence")
    require(c5["verdict"] == "BLOCKED" and c5["protocol_complete"] is True, "C5 evidence")

    routes: dict[str, str] = {}

    def collect(node: dict[str, object]) -> None:
        routes[str(node["slug"])] = str(node["file"])
        for child in node.get("children", []):
            collect(child)

    collect(logbook["root"])
    required_routes = {
        "index": "pages/index.md",
        "current-claim1": "pages/current-claim1/page.md",
        "current-claim2": "pages/current-claim2/page.md",
        "current-claim3": "pages/current-claim3/page.md",
        "current-claim4": "pages/current-claim4/page.md",
        "current-claim5": "pages/current-claim5/page.md",
        "verify": "pages/verify/page.md",
        "overview": "pages/overview/page.md",
    }
    for slug, path in required_routes.items():
        require(routes.get(slug) == path, f"logbook route {slug}")
        if slug.startswith("current") or slug == "index":
            require((ROOT / "space_patch" / path).is_file(), f"current page {slug}")
        else:
            require(f"space_patch/{path}" in manifest["historical_supplied_paths"], f"historical page declaration {slug}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    require("arXiv:2606.17816" in readme, "paper citation")
    require("Thank you" in readme, "thank-you note")
    require("forecast" in readme.lower(), "score forecast boundary")
    require("no current score" in readme.lower(), "current score boundary")
    require("STATUS.md" in readme and "CLAIM_EVIDENCE.md" in readme, "dossier links")
    branch_audit = (ROOT / "branch-audit.md").read_text(encoding="utf-8")
    require(CANONICAL_IDENTITY in branch_audit, "branch identity documentation")
    require("Result: PASS" in (ROOT / "space_patch/release/protected-subset-check.txt").read_text(), "release protected subset")

    print(
        "FINAL_AUDIT=VERIFIED "
        f"branches={len(EXPECTED_BRANCHES)} commits={EXPECTED_COMMITS} "
        "claims=C1:C2:C3_verified_scoped,C4_falsified_scoped,C5_blocked "
        "historical_score=5/10 current_score_claim=false publication_allowed=false"
    )


if __name__ == "__main__":
    main()
