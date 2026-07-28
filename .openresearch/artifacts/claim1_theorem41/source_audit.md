# Source audit — Theorem 4.1

Source: ar5iv HTML for arXiv:2606.17816, retrieved 2026-07-28 with an
explicit browser User-Agent. Whole-document SHA-256:
`31078a87b1a9b38b5d13b0bafe40512ac6cccb04bd6f35b8de05ed2b8b804557`.

Anchors: theorem `#S4.Thmtheorem1`; GELU proof `#A1.SS1`; SiLU proof
`#A1.SS2`. The paper defines a conservation law as a C1 function constant
along every Euclidean gradient-flow solution for every dataset and
initialization. It assumes a C2 loss and full loss-gradient span
`V_l=R^d_out`.

The exact quantified conclusion is a completeness result: all such laws are
constant. Showing that one ReLU-style candidate drifts is insufficient.

The appendix first presents scalar output and then extends the gradient
conditions componentwise. The independent reconstruction audits the scalar
functional family at `x=t e_j`, the dense generic parameter set, continuity,
and connectedness explicitly.
