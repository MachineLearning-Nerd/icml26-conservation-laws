# Branch audit

This repository was migrated from opaque OpenResearch-generated branch names to descriptive branches. The clean branches preserve the historical, theorem, counterexample, availability, and release snapshots.

## Mapping

| Former branch | Clean branch | Purpose |
| --- | --- | --- |
| orx/historical-judged-baseline-reconstruction | historical/judged-baseline | Preserve the original 5/10 judged toy baseline and source provenance. |
| orx/exact-normalized-sigmoid-counterexample | audit/claim4-normalized-sigmoid | Add the exact Theorem 4.7 normalized-sigmoid counterexample. |
| orx/swiglu-symmetry-proof-and-model-scale-audit | audit/claim2-swiglu-symmetry | Reconstruct the SwiGLU theorem and model-scale audit. |
| orx/mha-and-rope-symmetry-proof-at-model-scale | audit/claim3-attention-rope | Add MHA and RoPE group-action certificates. |
| orx/gelu-and-silu-universal-theorem-calibration | audit/claim1-gelu-silu | Calibrate the arbitrary-width GELU/SiLU completeness theorem. |
| orx/figure-2-non-circular-four-route-audit | audit/claim5-figure2-availability | Complete the four-route audit of the paper-scale Figure 2 claim. |
| orx/evaluator-visible-cumulative-release-candidate | release/evaluator-candidate | Assemble the cumulative evaluator-visible release. |
| orx/runnable-space-root-package | release/space-root-package | Make the fixed verifier runnable from the Space root. |
| main | main | Cumulative publication surface. |

## Migration guarantees

- Every live branch contains the current README and this branch audit.
- All reachable commits are attributed to MachineLearning-Nerd <37579156+MachineLearning-Nerd@users.noreply.github.com>.
- Former orx/* remote branches are deleted after their clean replacements are published.
- Active README and report links use the renamed repository and clean branch names.
- The DineshAI Space identifier and historical judged artifact are retained as provenance, not as GitHub ownership or branch names.

## Verification checklist

~~~
git show-ref --verify refs/heads/<branch>
git show <branch>:README.md >/dev/null
git show <branch>:branch-audit.md >/dev/null
git log <branch> --format='%an <%ae>' | sort -u
~~~
