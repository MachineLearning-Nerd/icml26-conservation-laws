# Source audit — Theorems 4.3 and 4.4

Theorem 4.3 at `#S4.Thmtheorem3` uses the MHA map in Equation 5 and identifies
the per-head matrix invariants `Q^TQ-K^TK` and `V^TV-O^TO`. Theorem 4.4 at
`#S4.Thmtheorem4` uses the RoPE map in Equation 7, requires even head dimension
`d_h=2m`, and identifies per-frequency-block Frobenius differences plus the
same value/output matrix invariant.

The verifier implements exactly the paper's relative rotation
`x_p Q R_{p-q} K^T x_q^T` through the algebraically equivalent factorization
`(x_p Q R_p)(x_q K R_q)^T`.

