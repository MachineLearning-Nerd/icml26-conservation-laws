# Source audit

## Paper

- Title: *Conservation Laws for Modern Neural Architectures*
- Authors: Viet-Hoang Tran, Vinh Khanh Bui, Tan Lai Ngoc, Nam Nguyen, Tuan
  Dam, and Tan M. Nguyen
- Source: [arXiv:2606.17816](https://arxiv.org/abs/2606.17816)
- Audited version: arXiv:2606.17816v1
- Retrieved source: https://ar5iv.labs.arxiv.org/html/2606.17816
- Retrieved UTC: 2026-07-28T12:12:40Z
- Retrieved HTML SHA-256:
  31078a87b1a9b38b5d13b0bafe40512ac6cccb04bd6f35b8de05ed2b8b804557

This GitHub repository is an independent reproduction and evidence audit. It
does not claim to be maintained by, endorsed by, or identical to the authors'
implementation.

## Paper anchors used

- Theorem 4.1: #S4.Thmtheorem1; FFN definitions: #S4.SS1
- Theorem 4.2: #S4.Thmtheorem2
- Theorem 4.3: #S4.Thmtheorem3; MHA definition: #S4.E5
- Theorem 4.4: #S4.Thmtheorem4; RoPE definition: #S4.E7
- Theorem 4.5: #S4.Thmtheorem5; dense MoE definition: #S4.E8
- Theorem 4.6: #S4.Thmtheorem6; explicit precondition k > 1
- Theorem 4.7: #S4.Thmtheorem7; normalized sigmoid gate: #S4.Ex35
- Empirical protocol: #S6; bound: #S6.E9; metric: #S6.E10
- Figure 2: #S5.F2; Appendix D model/data/runtime configuration: #A4

## Quantifier and assumption notes

The audit preserves the paper's strongest boundaries rather than silently
replacing them with proxy experiments:

- Theorems quantify over every C1 law, dataset, and initialization.
- The dynamics are Euclidean gradient flow.
- The loss is C2 in its prediction argument and the output-gradient span is
  the full output space.
- Theorem 4.6 requires k > 1.
- Figure 2 requires the named models, datasets, schedules, three learning
  rates, ten seeds per configuration, and the paper's block metric.

## Historical evaluation provenance

- Evaluator Space: DineshAI/ay4Q69fAJL
- Judged revision: 47668d9139058ae1e50a39d5bfcb7280fdd74c34
- Published evaluator revision: 6219f9cc8d0132ab7407b2d31712ec4180c13c67
- Historical live score: **5/10**

The historical Space artifact is preserved as provenance. It is not treated
as a current score claim, and this repository does not claim a new evaluation
without an external judge result.
