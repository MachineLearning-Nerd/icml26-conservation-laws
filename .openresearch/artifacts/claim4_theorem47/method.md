# Method

Use `d=3`, two experts, and one hidden unit. Let `x=e1`, `y=0`. Choose explicit
SwiGLU parameters so the two expert outputs at `x` are `E1(x)=0` and
`E2(x)=e1`:

- Expert 1 has `A1=0`.
- Expert 2 has `A2=(4/(3 ln 3),0,0)^T`,
  `B2=(ln 3,0,0)`, and `C2=(1,0,0)`.

Because `SiLU(ln 3)=3 ln(3)/4`, Expert 2 is exactly `e1`. Set gating rows
`W1=(0,0,0)` and `W2=(ln 3,0,0)`. The sigmoid values are exactly `1/2` and
`3/4`, so normalized gates are `2/5` and `3/5`.

For `L=0.5||MoE(x)||^2`, rational differentiation gives
`dL/dz1=-9/125`, `dL/dz2=9/250`, and hence
`dL/dc=-9/250` for the common shift `z_i -> z_i+c`. Since
`h(W)=W11+W21`, Euclidean gradient flow gives `dh/dt=9/250`, not zero.

The independent checker uses an 80-digit central difference on the scalar
loss along the common-shift direction. It does not reuse the rational
derivative. A matched softmax control gives exactly zero because softmax is
common-shift invariant. A fail-closed negative-control invocation asserts the
paper's claimed conservation and must exit with code 1.

