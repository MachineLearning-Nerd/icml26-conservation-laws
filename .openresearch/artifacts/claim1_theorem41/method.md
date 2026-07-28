# Method

The verification uses three non-circular components.

1. Reconstruct the arbitrary-width proof from activation Taylor coefficients.
   Both activations have a linear term plus a nonzero coefficient at every
   positive even power. Orthogonality at `x=t e_j` therefore yields
   `sum_i (u_i+m w_i) gamma_i^m=0` for every positive integer `m`, where
   `gamma_i=B_ij^2`.
2. Sum those identities as a formal power series. The resulting rational
   function has distinct double and simple poles. Each double-pole coefficient
   forces `w_i=0`; each remaining simple-pole coefficient forces `u_i=0`.
   This is an arbitrary-width argument. Exact Fraction arithmetic checks its
   finite numerator maps for widths 1–8.
3. Independently form sampled parameter Jacobians for generic scalar-output
   GELU and SiLU networks and check full column rank. This finite calculation
   is corroboration only.

The negative control repeats a squared incoming weight, so two rational poles
coincide. Its exact rank must drop.
