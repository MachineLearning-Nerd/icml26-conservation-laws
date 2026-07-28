# Method

For standard attention, apply `Q->QE`, `K->K E^{-T}`. The product reduces
exactly as `Q E E^{-1} K^T=QK^T`. The identical construction for `V,O`
leaves `VO^T` fixed. Differentiating these functional symmetries and
substituting Euclidean gradient flow yields the full matrix invariants.

For RoPE frequency block `j`, scale `Q^(j)` by `exp(t)` and `K^(j)` by
`exp(-t)`. Their contribution through every fixed `R_{p-q}^{(j)}` is
unchanged, yielding the block Frobenius invariant.

The independent numerical implementation performs explicit RoPE attention
and analytic backpropagation at `d=192`, head dimension 64, 3 heads, 12
layers, sequence length 32, and 10 seeds. The intended control is the
nonconserved column-sign combination specified in paper Appendix E.

