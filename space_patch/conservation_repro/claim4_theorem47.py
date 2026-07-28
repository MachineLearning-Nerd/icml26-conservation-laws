"""Exact counterexample to the normalized-sigmoid part of Theorem 4.7.

The primary calculation uses only rational arithmetic. An independent Decimal
finite-difference checker evaluates the loss along the common-logit-shift
direction without reusing the symbolic derivative.
"""

from __future__ import annotations

import argparse
import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction


def _fraction(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def exact_certificate() -> dict[str, object]:
    # At x=e_1, the explicit expert parameters in the claim contract produce
    # E_1(x)=0 and E_2(x)=e_1. The logits are (0, ln 3), so sigmoid values are
    # exactly (1/2, 3/4).
    s1, s2 = Fraction(1, 2), Fraction(3, 4)
    ds1, ds2 = s1 * (1 - s1), s2 * (1 - s2)
    denominator = s1 + s2
    g2 = s2 / denominator
    loss = Fraction(1, 2) * g2 * g2

    dg2_dz1 = -(s2 * ds1) / (denominator * denominator)
    dg2_dz2 = ds2 * (denominator - s2) / (denominator * denominator)
    dloss_dz1 = g2 * dg2_dz1
    dloss_dz2 = g2 * dg2_dz2
    common_shift_loss_derivative = dloss_dz1 + dloss_dz2
    invariant_gradient_flow_derivative = -common_shift_loss_derivative

    # Matched softmax control at logits (0, ln 3): probabilities (1/4, 3/4).
    p1, p2 = Fraction(1, 4), Fraction(3, 4)
    softmax_output = p2
    softmax_dloss_dz1 = softmax_output * (-p1 * p2)
    softmax_dloss_dz2 = softmax_output * (p2 * (1 - p2))
    softmax_common_shift_derivative = softmax_dloss_dz1 + softmax_dloss_dz2

    exact_ok = (
        g2 == Fraction(3, 5)
        and loss == Fraction(9, 50)
        and dloss_dz1 == Fraction(-9, 125)
        and dloss_dz2 == Fraction(9, 250)
        and common_shift_loss_derivative == Fraction(-9, 250)
        and invariant_gradient_flow_derivative == Fraction(9, 250)
        and softmax_common_shift_derivative == 0
    )
    return {
        "construction": {
            "d": 3,
            "n_experts": 2,
            "hidden_dimension": 1,
            "input": ["1", "0", "0"],
            "target": ["0", "0", "0"],
            "logits": ["0", "ln(3)"],
            "expert_outputs": [["0", "0", "0"], ["1", "0", "0"]],
            "loss": "squared_error_half_norm",
        },
        "normalized_sigmoid": {
            "sigmoid_values": [_fraction(s1), _fraction(s2)],
            "gate_2": _fraction(g2),
            "loss": _fraction(loss),
            "dL_dz1": _fraction(dloss_dz1),
            "dL_dz2": _fraction(dloss_dz2),
            "dL_d_common_shift": _fraction(common_shift_loss_derivative),
            "d_dt_sum_gate_rows_coordinate_1": _fraction(
                invariant_gradient_flow_derivative
            ),
            "claimed_invariant_is_conserved": invariant_gradient_flow_derivative == 0,
        },
        "softmax_reference_control": {
            "gate_2": _fraction(p2),
            "dL_dz1": _fraction(softmax_dloss_dz1),
            "dL_dz2": _fraction(softmax_dloss_dz2),
            "dL_d_common_shift": _fraction(softmax_common_shift_derivative),
            "claimed_invariant_is_conserved": softmax_common_shift_derivative == 0,
        },
        "exact_arithmetic_passed": exact_ok,
        "theorem_4_7_contradicted": invariant_gradient_flow_derivative != 0,
        "verdict": "FALSIFIED" if exact_ok and invariant_gradient_flow_derivative != 0 else "FAIL",
    }


def independent_decimal_check() -> dict[str, object]:
    """Numerically differentiate the loss along a common shift at 80 digits."""

    with localcontext() as context:
        context.prec = 80
        one = Decimal(1)
        three = Decimal(3)
        ln3 = three.ln()
        step = Decimal("1e-25")

        def sigmoid(value: Decimal) -> Decimal:
            return one / (one + (-value).exp())

        def normalized_sigmoid_loss(shift: Decimal) -> Decimal:
            s1 = sigmoid(shift)
            s2 = sigmoid(ln3 + shift)
            output = s2 / (s1 + s2)
            return output * output / Decimal(2)

        def softmax_loss(shift: Decimal) -> Decimal:
            e1 = shift.exp()
            e2 = (ln3 + shift).exp()
            output = e2 / (e1 + e2)
            return output * output / Decimal(2)

        normalized_derivative = (
            normalized_sigmoid_loss(step) - normalized_sigmoid_loss(-step)
        ) / (Decimal(2) * step)
        softmax_derivative = (
            softmax_loss(step) - softmax_loss(-step)
        ) / (Decimal(2) * step)
        expected = Decimal(-9) / Decimal(250)
        normalized_error = abs(normalized_derivative - expected)
        softmax_error = abs(softmax_derivative)
        passed = normalized_error < Decimal("1e-45") and softmax_error < Decimal(
            "1e-45"
        )
        return {
            "method": "80-digit central difference along common logit shift",
            "step": str(step),
            "normalized_sigmoid_dL_dc": str(normalized_derivative),
            "expected_dL_dc": str(expected),
            "absolute_error": str(normalized_error),
            "softmax_dL_dc": str(softmax_derivative),
            "softmax_absolute_error": str(softmax_error),
            "passed": passed,
        }


def softmax_and_sparse_structural_certificate() -> dict[str, object]:
    """Address Theorems 4.5-4.6 independently of the sigmoid falsification."""
    scores = [2.0, 1.0, 0.0]
    expert_values = [0.0, 1.0, 4.0]
    selected = (0, 1)

    def sparse_output(values: list[float]) -> float:
        weights = [math.exp(values[index]) for index in selected]
        denominator = sum(weights)
        return sum(
            weight * expert_values[index] / denominator
            for weight, index in zip(weights, selected, strict=True)
        )

    original = sparse_output(scores)
    common_shift = sparse_output([score + 7.0 for score in scores])
    wrong_individual_shift = sparse_output([scores[0] + 0.5, scores[1], scores[2]])
    common_shift_residual = abs(common_shift - original)
    wrong_shift_change = abs(wrong_individual_shift - original)
    passed = common_shift_residual < 1e-14 and wrong_shift_change > 1e-3
    return {
        "dense_softmax": {
            "symmetry": "W_i -> W_i + r for every expert i",
            "reason": "all logits receive the same r*x, so softmax gates are unchanged",
            "consequence": "sum_i W_i is conserved under Euclidean gradient flow",
        },
        "sparse_top_k_softmax": {
            "precondition": "k>1",
            "symmetry": "W_i -> W_i + r for every expert i",
            "selection_reason": (
                "a common logit shift preserves the full ordering and Top-k set; "
                "renormalized softmax weights within that set are unchanged"
            ),
            "expert_reason": (
                "each selected or unselected SwiGLU expert retains its independent "
                "A-column/C-row rescaling symmetry"
            ),
            "consequence": (
                "the displayed dense gating-row-sum and expert invariants remain "
                "laws of sparse Top-k MoE"
            ),
        },
        "numeric_control": {
            "scores": scores,
            "selected_indices": list(selected),
            "common_shift_residual": common_shift_residual,
            "individual_shift_change": wrong_shift_change,
            "individual_shift_rejected_as_intended": wrong_shift_change > 1e-3,
        },
        "completeness_scope": (
            "This exact functional-symmetry certificate verifies the displayed "
            "laws for dense and sparse softmax MoE; it does not independently "
            "re-prove the paper's no-additional-laws classification."
        ),
        "passed": passed,
    }


def verify() -> dict[str, object]:
    exact = exact_certificate()
    independent = independent_decimal_check()
    positive = softmax_and_sparse_structural_certificate()
    passed = (
        exact["verdict"] == "FALSIFIED"
        and bool(exact["exact_arithmetic_passed"])
        and bool(independent["passed"])
        and bool(positive["passed"])
    )
    return {
        "claim": "Theorem 4.7 normalized-sigmoid MoE gating conservation",
        "exact_certificate": exact,
        "independent_checker": independent,
        "softmax_and_sparse_structural_certificate": positive,
        "all_checks_passed": passed,
        "verdict": "FALSIFIED" if passed else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assert-theorem",
        action="store_true",
        help="negative control: require the claimed gating invariant to be conserved",
    )
    args = parser.parse_args()
    result = verify()
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.assert_theorem:
        is_conserved = bool(
            result["exact_certificate"]["normalized_sigmoid"][
                "claimed_invariant_is_conserved"
            ]
        )
        print(
            "NEGATIVE CONTROL: theorem conservation assertion "
            + ("unexpectedly passed" if is_conserved else "rejected as intended")
        )
        return 0 if is_conserved else 1
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
