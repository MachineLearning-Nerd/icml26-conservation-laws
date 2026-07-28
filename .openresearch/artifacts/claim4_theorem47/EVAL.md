# EVAL — Claim 4 / Theorem 4.7

Verdict: **FALSIFIED**

At an explicit normalized-sigmoid dense-MoE parameter point satisfying the
paper's stated model and loss assumptions, the claimed Theorem 4.5 gating
invariant has exact gradient-flow derivative

`d/dt (W[1,1] + W[2,1]) = 9/250`.

The matched softmax control is exactly zero. The independent 80-digit central
difference recovers the corresponding common-shift loss derivative `-9/250`
with absolute error `2.28e-53`. The fail-closed theorem assertion exits 1.

This falsifies the normalized-sigmoid portion of Theorem 4.7. It does not by
itself falsify Theorems 4.5 or 4.6, the per-expert SwiGLU invariants, or the
sparse normalized-sigmoid statement.

