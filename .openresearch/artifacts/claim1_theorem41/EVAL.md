# C1 evaluator record

Current verdict: **VERIFIED**, conditional on the formal cumulative run
passing.

The old six-seed candidate-drift result remains historical toy evidence. The
current verifier is `conservation_repro.claim1_theorem41`, invoked by the
fixed cumulative command:

`uv run --frozen --no-dev python -m conservation_repro.run`

The verifier fails closed when the exact rational map loses rank, the
repeated-pole control does not lose rank, or either activation's finite
generic Jacobian lacks full column rank.
