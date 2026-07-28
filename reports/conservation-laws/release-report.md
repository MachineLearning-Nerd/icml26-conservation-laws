# Release report

- Previous live judged score: `5/10`
- Conservative projected score range after the proposed change: `7–9/10`
- Best-supported possible new score: `9/10` — forecast only, not a judge result

## Claim forecast

| Claim | Current points | Possible points | Confidence | Evidence status | Basis and remaining risk |
| --- | ---: | ---: | --- | --- | --- |
| C1 | 1 | 2 | HIGH | VERIFIED | Arbitrary-width symbolic reconstruction, exact Fraction ranks, rank-dropping control, and independent full-span audits. Reviewer may demand a proof assistant. |
| C2 | 1 | 2 | HIGH | VERIFIED | Exact iff gradient-condition reconstruction, model symmetry, and paper-width audit. Numerical layers are repeated audits, not training. |
| C3 | 1 | 2 | MEDIUM | VERIFIED | Standard MHA and RoPE displayed invariants have exact group-action certificates and strong controls. The broader no-additional-laws classification is not separately formalized. |
| C4 | 1 | 2 | HIGH | FALSIFIED | Dense/sparse softmax laws are addressed; normalized-sigmoid row-sum law has an exact assumption-satisfying `9/250` counterexample with an 80-digit checker. |
| C5 | 1 | 2 | LOW | BLOCKED | Three distinct verification routes plus mandatory falsification route completed. Exact named-dataset trajectories remain unavailable under CPU-only authorization. |

Current live total: **5/10**. Conservative projected total: **7–9/10**.
Best-supported possible total: **9/10**. Claims 1–4 changed materially since
the previous judge result. Claim 5 remains BLOCKED because the exact
120+-configuration H100 campaign, training implementation, and raw
trajectories are unavailable.

The publication action is: upload the exact text allowlist to the existing
`DineshAI/ay4Q69fAJL` Space through the Hugging Face API, preserve the judged
file set, verify the returned revision, mirror the same reader-facing text to
GitHub `main`, and mark the paper awaiting judge.

## Evaluator-visible matrix

| Claim | Canonical page | Code visible | Data inline | Raw link | Checker | Control | Exact claim tested | Reviewer verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | `pages/current-claim1/page.md` | yes | yes | yes | yes | repeated-pole rank loss | yes | VERIFIED |
| C2 | `pages/current-claim2/page.md` | yes | yes | yes | yes | elementwise non-law | yes | VERIFIED |
| C3 | `pages/current-claim3/page.md` | yes | yes | yes | yes | unmatched RoPE block | yes | VERIFIED |
| C4 | `pages/current-claim4/page.md` | yes | yes | yes | yes | softmax and false-assertion controls | yes | FALSIFIED |
| C5 | `pages/current-claim5/page.md` | yes | yes | yes | yes | slope-one and softmax-zero controls | yes | BLOCKED |

## Experiment tree and winning lineage

1. Historical judged baseline reconstruction — local run `b041710f…`.
2. Exact normalized-sigmoid counterexample — local run `cc479d85…`.
3. SwiGLU proof and model-width audit — HF run `03fc4630…`.
4. MHA and RoPE proof and audit — HF run `dd30dd24…`.
5. GELU and SiLU universal theorem calibration — HF run `dc3ad414…`.
6. Figure 2 four-route audit — HF run `0ec4b425…`.
7. Evaluator-visible cumulative release candidate — final formal run recorded
   in the release manifest.

The winning branch is
`orx/evaluator-visible-cumulative-release-candidate`; its final Git SHA is
recorded in the manifest after the release-candidate commit.

## Runtime and compute

- Local baseline: 5 s wall time.
- Local exact counterexample: 10 s wall time.
- HF `cpu-upgrade` formal jobs: four completed 21 s jobs before the final
  candidate; 64 logical CPUs were visible.
- No GPU was used.
- Hugging Face cost is not exposed by `orx`; no monetary value is invented.

The final run records the pre-run estimate, selected `cpu-upgrade` flavor,
actual affinity/visible CPU counts, and verifier runtime in its raw output.

## Command ledger

Startup and source audit:

```bash
orx skill
orx skill orx-experiment-tree
orx skill orx-evidence
orx skill orx-git
orx skill orx-compute
orx projects --json
orx project view f9b76bd4-4164-41e8-89c4-19861c96ef75
orx runs f9b76bd4-4164-41e8-89c4-19861c96ef75
git status --short
git branch -a
df -h .
orx paper 2606.17816 --full
orx lit "conservation laws gradient flows neural networks Marcotte Proposition 5.1" --limit 8
```

Fixed command and formal launches:

```bash
uv run --frozen --no-dev python -m conservation_repro.run
orx exp run 1f3b4b80-4c0a-4b95-bed7-d65b1acc200e --backend local
orx exp run 790334a8-838f-42f3-9e39-d2e545bdc5ac --backend hf --flavor cpu-upgrade --image ghcr.io/astral-sh/uv:python3.12-bookworm-slim
orx exp run fcb64e20-15e3-45ff-aa43-cca27656a470 --backend hf --flavor cpu-upgrade --image ghcr.io/astral-sh/uv:python3.12-bookworm-slim
orx exp run 1f9b3892-2c2d-44cd-a4ef-743a9e03d980 --backend hf --flavor cpu-upgrade --image ghcr.io/astral-sh/uv:python3.12-bookworm-slim
orx exp run 5aab0d7a-78ea-4849-ad01-afd914cc3094 --backend hf --flavor cpu-upgrade --image ghcr.io/astral-sh/uv:python3.12-bookworm-slim
orx exp wait <experiment-id> --timeout 480
orx logs <run-id> --bytes 200000
```

Release validation:

```bash
uv run --frozen --no-dev python -m py_compile conservation_repro/*.py
uv run --frozen --no-dev python -m json.tool space_patch/logbook.json
marimo check notebooks/conservation_laws_tutorial.py
git diff --check
git ls-remote origin
```

All experiment nodes were created with `orx create-experiment ... --parent
<parent-id>`, checked out with `git fetch origin && git checkout <branch>`,
committed, and pushed before their runs.

## Evidence paths

- Internal contracts and raw summaries: `.openresearch/artifacts/`
- Candidate evaluator patch: `space_patch/`
- Illustrated report: `reports/conservation-laws/report.md`
- Tutorial notebook: `notebooks/conservation_laws_tutorial.py`
- Project dashboard mirror:
  `project/conservation-laws-reproduction/report.md` in the OpenResearch files
  directory.

The protected old/new subset check, exact text upload allowlist, SHA-256
manifest, secret scan, blind-review file-open trace, final Hugging Face
revision, and final GitHub remote SHA are recorded in the sealed release
manifest.
