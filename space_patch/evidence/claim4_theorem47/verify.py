"""Standalone exact verifier for arXiv:2606.17816 Theorem 4.7."""
from decimal import Decimal, localcontext
from fractions import Fraction
import json
import sys

s1, s2 = Fraction(1, 2), Fraction(3, 4)
ds1, ds2 = s1 * (1 - s1), s2 * (1 - s2)
den = s1 + s2
g2 = s2 / den
d1 = g2 * (-(s2 * ds1) / den**2)
d2 = g2 * (ds2 * (den - s2) / den**2)
dh_dt = -(d1 + d2)

p1, p2 = Fraction(1, 4), Fraction(3, 4)
softmax_shift = p2 * (-p1 * p2) + p2 * (p2 * (1 - p2))

with localcontext() as ctx:
    ctx.prec = 80
    one, ln3, step = Decimal(1), Decimal(3).ln(), Decimal("1e-25")

    def sigmoid(x):
        return one / (one + (-x).exp())

    def loss(c):
        a, b = sigmoid(c), sigmoid(ln3 + c)
        out = b / (a + b)
        return out * out / Decimal(2)

    independent = (loss(step) - loss(-step)) / (2 * step)
    independent_error = abs(independent - Decimal(-9) / Decimal(250))

passed = (
    dh_dt == Fraction(9, 250)
    and softmax_shift == 0
    and independent_error < Decimal("1e-45")
)
result = {
    "exact_dh_dt": f"{dh_dt.numerator}/{dh_dt.denominator}",
    "softmax_shift_derivative": str(softmax_shift),
    "independent_dL_dc": str(independent),
    "independent_error": str(independent_error),
    "verdict": "FALSIFIED" if passed else "FAIL",
}
print(json.dumps(result, indent=2))
if "--assert-theorem" in sys.argv:
    raise SystemExit(0 if dh_dt == 0 else 1)
raise SystemExit(0 if passed else 1)

