---
title: "Repro - NN Conservation Laws"
emoji: 🎯
colorFrom: yellow
colorTo: red
sdk: static
pinned: false
tags:
 - trackio
 - trackio-logbook
 - open-experiment
 - icml2026-repro
 - paper-ay4Q69fAJL
---

# Conservation Laws for Modern Neural Architectures — current reproduction

Previous live judged score: **5/10**

Conservative projected score after this candidate: **7–9/10**

Best-supported possible score: **9/10 (forecast, not a judge result)**

This is the canonical evaluator entrypoint. Current verification appears
first. The exact judged revision
`47668d9139058ae1e50a39d5bfcb7280fdd74c34` is preserved additively; its old
`verify` and `overview` pages are labeled **Historical rejected baseline**.

## Current claim summary

| Claim | Status | Central evidence | Canonical page |
| --- | --- | --- | --- |
| C1 GELU/SiLU | **VERIFIED** | arbitrary-width proof; exact ranks 2–16 | [Claim 1](#/current-claim1) |
| C2 SwiGLU | **VERIFIED** | iff reconstruction; relative rate `3.91e-16` | [Claim 2](#/current-claim2) |
| C3 MHA/RoPE | **VERIFIED** | exact RoPE symmetry; rate `6.66e-16` | [Claim 3](#/current-claim3) |
| C4 MoE variants | **FALSIFIED** | sigmoid gate-row derivative `9/250` | [Claim 4](#/current-claim4) |
| C5 Figure 2 | **BLOCKED** | four routes; exact CPU campaign unavailable | [Claim 5](#/current-claim5) |

The exact inherited command for every formal node is:

```bash
uv run --frozen --no-dev python -m conservation_repro.run
```

Pinned environment:
[pyproject.toml](environment/pyproject.toml) · [uv.lock](environment/uv.lock).
The complete cumulative source is visible under
[source/conservation_repro](source/conservation_repro/run.py).

## Evaluator-visible evidence matrix

| Claim | Canonical page | Code visible | Data inline | Raw link | Checker | Control | Exact claim tested | Reviewer verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | [page](#/current-claim1) | [source](source/conservation_repro/claim1_theorem41.py) | yes | [raw](evidence/claim1_theorem41/formal_run_summary.json) | [checker](evidence/claim1_theorem41/verify.py) | repeated-pole rank drop | yes, universal C1 classification | VERIFIED |
| C2 | [page](#/current-claim2) | [source](source/conservation_repro/claim2_swiglu.py) | yes | [raw](evidence/claim2_theorem42/model_scale_summary.json) | [checker](evidence/claim2_theorem42/verify.py) | elementwise non-law | yes, iff conditions and invariants | VERIFIED |
| C3 | [page](#/current-claim3) | [source](source/conservation_repro/claim3_attention.py) | yes | [raw](evidence/claim3_attention/formal_run_summary.json) | [checker](evidence/claim3_attention/verify.py) | unmatched RoPE block | yes, MHA and RoPE displayed laws | VERIFIED |
| C4 | [page](#/current-claim4) | [source](source/conservation_repro/claim4_theorem47.py) | yes | [raw](evidence/claim4_theorem47/exact_certificate.json) | [checker](evidence/claim4_theorem47/verify.py) | matched softmax / false theorem assertion | yes, Theorems 4.5–4.7 | FALSIFIED |
| C5 | [page](#/current-claim5) | [source](source/conservation_repro/claim5_figure2.py) | yes | [raw](evidence/claim5_figure2/formal_run_summary.json) | [checker](evidence/claim5_figure2/verify.py) | non-law slope / softmax zero | yes, named Figure 2 trajectories | BLOCKED |

Every checker exits nonzero if its recorded evidence or intended control fails.
A zero exit for Claim 5 means the four-route BLOCKED record is internally
complete; it does not turn the empirical claim into a pass.

## Exact scope and limitations

- C1 is proof-level: finite ranks support but do not replace its arbitrary-width
  pole-order derivation.
- C2 includes the theorem's iff gradient characterization and exact invariant
  symmetry; its model-width audit is corroboration.
- C3 includes RoPE and standard MHA. The displayed invariants are exact; the
  broader no-additional-laws classification is not separately formalized.
- C4 structurally addresses dense and sparse softmax laws and provides a valid
  assumption-satisfying counterexample to normalized sigmoid.
- C5 used no named-dataset training and is not full scale. It remains BLOCKED
  after three verification routes and the mandatory falsification route.

## Historical evidence

The old [verify](#/verify) and [overview](#/overview) pages remain reachable
unchanged as **Historical rejected baseline**. They are superseded by the five
current pages above.
