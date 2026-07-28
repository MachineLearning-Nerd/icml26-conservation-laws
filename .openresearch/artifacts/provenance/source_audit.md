# Paper source audit

- Paper: *Conservation Laws for Modern Neural Architectures*, arXiv:2606.17816v1.
- Retrieved URL: `https://ar5iv.labs.arxiv.org/html/2606.17816`
- Retrieval time: `2026-07-28T12:12:40Z`
- HTTP User-Agent: `OpenResearch-Reproduction/1.0 (+https://github.com/MachineLearning-Nerd/icml26-repro-ay4Q69fAJL-conservation-laws-for-modern-neural-architectures)`
- Retrieved HTML SHA-256: `31078a87b1a9b38b5d13b0bafe40512ac6cccb04bd6f35b8de05ed2b8b804557`

Primary anchors:

- Theorem 4.1: `#S4.Thmtheorem1`; FFN definitions: `#S4.SS1`.
- Theorem 4.2: `#S4.Thmtheorem2`.
- Theorem 4.3: `#S4.Thmtheorem3`; MHA definition: `#S4.E5`.
- Theorem 4.4: `#S4.Thmtheorem4`; RoPE definition: `#S4.E7`.
- Theorem 4.5: `#S4.Thmtheorem5`; dense MoE definition: `#S4.E8`.
- Theorem 4.6: `#S4.Thmtheorem6`; explicit precondition `k > 1`.
- Theorem 4.7: `#S4.Thmtheorem7`; normalized sigmoid gate: `#S4.Ex35`.
- Empirical protocol: `#S6`; bound: `#S6.E9`; metric: `#S6.E10`.
- Figure 2: `#S5.F2`; Appendix D model/data/runtime configuration: `#A4`.

Global assumptions imported from Sections 2–3:

- Euclidean gradient flow.
- Conservation laws are `C1` functions on the full parameter space and hold
  for every initialization and every dataset.
- The loss is `C2` in its prediction argument.
- `V_l(z)` is independent of `z`, and the paper thereafter assumes
  `V_l = R^{d_out}`.

Exact quantifier notes:

- Theorem 4.1 says **all** conservation laws for the stated GELU or SiLU FFN
  are constant. Finite trajectories cannot verify this universal completeness
  statement.
- Theorem 4.2 quantifies over every conservation law and every hidden index.
- Theorems 4.3–4.4 quantify over every conservation law and every head (and
  every RoPE frequency block for Theorem 4.4).
- Theorem 4.6 is restricted to `k > 1`.
- Theorem 4.7 says **every** conservation law for normalized-sigmoid dense and
  sparse MoE coincides with the softmax case and satisfies exactly the
  Theorem 4.5 constraints.
- Section 6 uses 10 independent seeds per configuration and the named
  Qwen-3/ViT models and four named datasets. The claimed empirical scaling is
  `O(tau^2 k)` under the stated bounded-Hessian/bounded-gradient assumptions.

