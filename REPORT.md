# Audit report

## Result boundary

The previous live judged result is **5/10**. This repository records no new
judge score. The current evidence package is:

| Claim group | Result | Evidence boundary |
| --- | --- | --- |
| C1 | VERIFIED_SCOPED | Arbitrary-width pole certificate plus exact and finite audits |
| C2 | VERIFIED_SCOPED | Inverse-scaling symmetry, iff reconstruction, and model-width residuals |
| C3 | VERIFIED_SCOPED | MHA/RoPE group actions and paper-width audit |
| C4 | FALSIFIED_SCOPED | Exact normalized-sigmoid gate-row-sum counterexample |
| C5 | BLOCKED | Four-route audit; named training artifacts and H100 campaign unavailable |

The existing evaluator-visible space_patch candidate records a conservative
forecast of 7–9/10, with 9/10 as the best-supported possible score. That
forecast is clearly separated from the historical 5/10 judge result and is
not a current evaluation.

## Claim production path

Each claim is produced from a committed source anchor, a claim contract or
formal summary, a verifier, and a control:

- C1 uses the arbitrary-width rational/pole derivation, exact Fraction ranks,
  and a repeated-gamma rank-loss control.
- C2 uses the inverse-scaling group action, the gradient characterization,
  paper-width float64 audits, and an elementwise non-law control.
- C3 uses exact Q/K, V/O, and RoPE symmetries plus a non-conserved block
  control.
- C4 uses exact rational differentiation, an independent high-precision
  checker, and a matched softmax control.
- C5 uses discretization, availability, resource, and falsification routes;
  it stays BLOCKED because a proxy cannot answer the exact empirical claim.

## Release provenance

The evaluator-visible space_patch candidate retains the judged historical
pages and reports Result: PASS for the protected-subset check. The root
README and branch audit are mirrored in this GitHub repository, while the
historical Space remains an external provenance record.

## Limitations

The theorem-level universal statements are not imported into a proof
assistant. Finite numerical audits support the structural derivations but do
not replace them. Claim 4's counterexample addresses the normalized-sigmoid
portion of Theorem 4.7 only. Claim 5 needs the exact authors' training code,
raw trajectories, or an authorized paper-scale accelerator campaign.
