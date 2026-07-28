# Source audit — Theorem 4.7

The source is arXiv:2606.17816v1, retrieved from ar5iv on
2026-07-28T12:12:40Z with SHA-256
`31078a87b1a9b38b5d13b0bafe40512ac6cccb04bd6f35b8de05ed2b8b804557`.

The normalized sigmoid gate immediately before Theorem 4.7 is

`g_i(x;W) = sigmoid(W_i x) / sum_p sigmoid(W_p x)`.

The theorem at `#S4.Thmtheorem7` says every conservation law for both
sigmoid-gated dense MoE and SMoE coincides with that of standard
softmax-gated MoE and satisfies exactly the Theorem 4.5 constraints.
Theorem 4.5 at `#S4.Thmtheorem5` includes the gating invariant
`sum_i W_i`.

The paper's global assumptions from Sections 2–3 are Euclidean gradient flow,
`C1` conservation laws holding for every dataset and initialization, a `C2`
loss, and `V_ell=R^d_out`. The counterexample uses squared loss, so these loss
conditions hold. It targets dense MoE, so the sparse theorem's `k>1`
restriction is irrelevant.

