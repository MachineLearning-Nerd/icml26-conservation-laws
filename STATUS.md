# Reproduction status

## Overall verdict

**PARTIAL_C1_C2_C3_VERIFIED_C4_FALSIFIED_C5_BLOCKED_HISTORICAL_SCORE_5_OF_10_NO_CURRENT_SCORE**

This repository is an independent, claim-by-claim audit of
[*Conservation Laws for Modern Neural Architectures*](https://arxiv.org/abs/2606.17816).
It is not the authors' official implementation.

- Historical live judge result: **5/10** for the evaluator-visible Space.
- The current evidence package does not claim a new judge score.
- publication_allowed=false; no author endorsement is claimed.
- Claims 1–3 pass their explicit theorem and audit scopes.
- Claim 4 is falsified by an exact assumption-matched counterexample to the
  normalized-sigmoid statement.
- Claim 5 remains blocked because the exact paper-scale training campaign and
  raw trajectories are unavailable under the CPU-only contract.

| Claim | Status | How the result is produced | Boundary |
| --- | --- | --- | --- |
| C1 GELU/SiLU completeness | VERIFIED_SCOPED | Arbitrary-width pole/rational certificate, exact ranks, and sampled full-span audits | Supporting computation does not replace a formal proof assistant |
| C2 SwiGLU invariants | VERIFIED_SCOPED | Inverse-scaling symmetry, iff reconstruction, model-width residual audit, and non-law control | Numerical layers corroborate the structural derivation |
| C3 MHA/RoPE invariants | VERIFIED_SCOPED | Exact group-action certificates and paper-width RoPE audit with a non-conserved control | Broader completeness classification is not separately formalized |
| C4 normalized-sigmoid MoE | FALSIFIED_SCOPED | Exact d/dt gate-row-sum counterexample with a matched softmax control | Falsifies the normalized-sigmoid part, not softmax or expert-level laws |
| C5 Figure 2 scaling | BLOCKED | Four-route availability, resource, discretization, and falsification audit | Named datasets, architectures, raw trajectories, and H100 campaign remain unavailable |

See [CLAIM_EVIDENCE.md](CLAIM_EVIDENCE.md) for the production path of each
claim, [SOURCE_AUDIT.md](SOURCE_AUDIT.md) for paper provenance, and
[ENVIRONMENT.md](ENVIRONMENT.md) for the locked runtime.
