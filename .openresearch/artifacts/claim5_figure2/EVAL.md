# C5 evaluator record

Verdict: **BLOCKED** after three materially different verification routes and
the mandatory fourth falsification route.

The fixed command is:

`uv run --frozen --no-dev python -m conservation_repro.run`

The command emits all four route records. It exits nonzero if a route is
missing, the algebraic calibration fails, or the negative control no longer
behaves as intended. A zero exit indicates that the BLOCKED conclusion is
reproducibly supported; it does not convert the claim into PASS.
