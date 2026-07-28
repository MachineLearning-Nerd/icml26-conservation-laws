# Conservation Laws for Modern Neural Architectures — reproduction

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-ay4Q69fAJL-conservation-laws-for-modern-neural-architectures/blob/main/notebooks/conservation_laws_tutorial.py)

This repository reproduces the five judged claims of
[arXiv:2606.17816](https://arxiv.org/abs/2606.17816) under CPU-only compute.
The previous live score is **5/10**. The current evidence forecast is
**7–9/10**, with **9/10 the best-supported possible score**—not a judge result.

The headline result is an assumption-satisfying exact counterexample to the
normalized-sigmoid part of Theorem 4.7:

| Paper prediction | Observed result | Assessment |
| --- | --- | --- |
| `d/dt sum_i W_i = 0` | `d/dt(W_1,1+W_2,1)=9/250` | **FALSIFIED** |

Claims 1–3 are VERIFIED with arbitrary-width symbolic or exact symmetry
certificates plus independent numerical audits. Claim 4 is FALSIFIED while
dense and sparse softmax laws are separately addressed. Claim 5 is BLOCKED
after four routes: the exact 120+-configuration H100 campaign and its raw
trajectories are unavailable under CPU-only authorization. No toy result is
called full-scale.

Read the [illustrated technical report](reports/conservation-laws/report.md)
or the [self-contained tutorial notebook](notebooks/conservation_laws_tutorial.py).
The notebook embeds the formal results; expensive experiments are not required
to see them.

## Reproduce

The environment is Python 3.12 and NumPy 2.3.2, pinned by `uv.lock`. Every
formal node inherits exactly:

```bash
uv run --frozen --no-dev python -m conservation_repro.run
```

The command regenerates `outputs/baseline.json` and exits nonzero when a
certificate, independent checker, or intended control fails. Uncertain or
multi-core work used Hugging Face `cpu-upgrade`; only short one-core checks
ran locally.

## Experiment log

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
| --- | --- | --- | --- | --- |
| `main` | Publication surface | Not run as an experiment (publication surface) | README, report, notebook, published evidence | local text checks |
| [Historical judged baseline](https://github.com/MachineLearning-Nerd/icml26-repro-ay4Q69fAJL-conservation-laws-for-modern-neural-architectures/tree/orx/historical-judged-baseline-reconstruction) | Reconstruct and freeze judged toy baseline | `uv run --frozen --no-dev python -m conservation_repro.run` | TOY baseline reproduced | local CPU, 5 s |
| [Exact normalized-sigmoid counterexample](https://github.com/MachineLearning-Nerd/icml26-repro-ay4Q69fAJL-conservation-laws-for-modern-neural-architectures/tree/orx/exact-normalized-sigmoid-counterexample) | Falsify Theorem 4.7 gate-row law | `uv run --frozen --no-dev python -m conservation_repro.run` | C4 FALSIFIED; exact `9/250` | local CPU, 10 s |
| [SwiGLU proof and audit](https://github.com/MachineLearning-Nerd/icml26-repro-ay4Q69fAJL-conservation-laws-for-modern-neural-architectures/tree/orx/swiglu-symmetry-proof-and-model-scale-audit) | Theorem 4.2 exact and paper-width audit | `uv run --frozen --no-dev python -m conservation_repro.run` | C2 VERIFIED; relative residual `3.91e-16` | HF cpu-upgrade, 21 s |
| [MHA and RoPE proof](https://github.com/MachineLearning-Nerd/icml26-repro-ay4Q69fAJL-conservation-laws-for-modern-neural-architectures/tree/orx/mha-and-rope-symmetry-proof-at-model-scale) | Add missing Theorem 4.4 RoPE evidence | `uv run --frozen --no-dev python -m conservation_repro.run` | C3 VERIFIED; RoPE rate `6.66e-16` | HF cpu-upgrade, 21 s |
| [GELU / SiLU completeness](https://github.com/MachineLearning-Nerd/icml26-repro-ay4Q69fAJL-conservation-laws-for-modern-neural-architectures/tree/orx/gelu-and-silu-universal-theorem-calibration) | Replace candidate drift with proof-level evidence | `uv run --frozen --no-dev python -m conservation_repro.run` | C1 VERIFIED; exact ranks 2–16 | HF cpu-upgrade, 21 s |
| [Figure 2 four-route audit](https://github.com/MachineLearning-Nerd/icml26-repro-ay4Q69fAJL-conservation-laws-for-modern-neural-architectures/tree/orx/figure-2-non-circular-four-route-audit) | Non-circularity, scope, CPU calibration, falsification route | `uv run --frozen --no-dev python -m conservation_repro.run` | C5 BLOCKED; exact paper-scale run unavailable | HF cpu-upgrade, 21 s |
| [Cumulative release candidate](https://github.com/MachineLearning-Nerd/icml26-repro-ay4Q69fAJL-conservation-laws-for-modern-neural-architectures/tree/orx/evaluator-visible-cumulative-release-candidate) | Full regression and evaluator-visible package | `uv run --frozen --no-dev python -m conservation_repro.run` | All cumulative checks passed | HF cpu-upgrade, 16 s |
| [Runnable Space package](https://github.com/MachineLearning-Nerd/icml26-repro-ay4Q69fAJL-conservation-laws-for-modern-neural-architectures/tree/orx/runnable-space-root-package) | Make the exact fixed command runnable from Space root | `uv run --frozen --no-dev python -m conservation_repro.run` | All cumulative checks passed; published as HF `6219f9c…` | HF cpu-upgrade, 26 s |

## Scope and provenance

The paper used PyTorch 2.9.1/CUDA 12.8 on an 80GB H100 for Figure 2; this
campaign did not use a GPU. Numerical audits match paper widths where useful
but are supporting evidence, not replacements for universal proofs or named
dataset training. Source retrieval, judged-Space manifests, claim contracts,
raw summaries, limitations, and fail-closed checkers are under
`.openresearch/artifacts/`.

Published evaluator artifact:
[Hugging Face revision `6219f9cc8d0132ab7407b2d31712ec4180c13c67`](https://huggingface.co/spaces/DineshAI/ay4Q69fAJL/commit/6219f9cc8d0132ab7407b2d31712ec4180c13c67).
The exact uploaded text mirror and release manifests are under `space_patch/`.
