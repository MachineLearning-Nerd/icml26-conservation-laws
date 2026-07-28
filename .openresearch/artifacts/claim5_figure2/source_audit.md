# Source audit — Figure 2 and Section 6

The source document and hash are recorded in the project provenance audit.
Relevant anchors are Figure 2 `#S5.F2`, Equation 9 `#S6.E9`, the block metric
`#S6.E10`, and configurations `#A4`.

The paper reports three learning rates and ten independent seeds per
configuration. It uses Qwen-3-style 12-layer language models on PTB and
WikiText-103 and ViTs on CIFAR-10 and ImageNet-1K. Appendix D states PyTorch
2.9.1/CUDA 12.8, a single 80GB H100, 12 data workers, and up to four hours per
individual configuration.

Equation 9 is an upper bound in expectation, not an assertion that a fitted
log-log slope must equal two. A single quadratic-invariant Euler step has
exactly quadratic error once its linear term cancels, so that check alone is
circular evidence for the architecture-level empirical claim.
