# Method

For each hidden index, apply the exact one-parameter transformation

`A[:,i] -> exp(t) A[:,i]`, `C[i,:] -> exp(-t) C[i,:]`.

The corresponding SwiGLU model term has exponents `+1 + 0 -1 = 0`, with
`SiLU(B[i,:]x)` treated as an arbitrary unchanged factor. The model and every
loss composed with it are invariant. Differentiating at `t=0` gives
`<grad_A L,A>-<grad_C L,C>=0`. Under Euclidean gradient flow this is exactly
half the negative derivative of the norm-difference invariant.

The independent numerical route instantiates 12 sequential blocks at the
paper's CIFAR-10 ViT widths (`d=256`, `d1=1024`) for 10 seeds and directly
forms analytic gradients. The control tracks the paper Appendix E
elementwise difference, which lacks the required column/row aggregation.

