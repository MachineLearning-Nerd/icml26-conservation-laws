"""Small NumPy reference implementations used by the frozen historical baseline.

These checks reconstruct the evidence visible in the judged Hugging Face Space.
They are intentionally small and are not presented as proof-level or full-scale
evidence.
"""

from __future__ import annotations

import os
import math

# The local baseline is authorized for one CPU core. Set limits before NumPy is
# imported so BLAS cannot silently turn these tiny checks into a multithreaded job.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-x))


def silu(x: Array) -> Array:
    return x * sigmoid(x)


def silu_prime(x: Array) -> Array:
    s = sigmoid(x)
    return s + x * s * (1.0 - s)


def gelu(x: Array) -> Array:
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))


def gelu_prime(x: Array) -> Array:
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))
    pdf = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
    return cdf + x * pdf


def _ffn_candidate_rate(seed: int, activation: str) -> float:
    rng = np.random.default_rng(seed)
    d, hidden = 8, 12
    a = rng.normal(scale=0.25, size=(d, hidden))
    b = rng.normal(scale=0.25, size=(hidden, d))
    x = rng.normal(size=d)
    target = rng.normal(size=d)
    z = b @ x
    if activation == "gelu":
        phi, dphi = gelu(z), gelu_prime(z)
    else:
        phi, dphi = silu(z), silu_prime(z)
    residual = a @ phi - target
    grad_a = np.outer(residual, phi)
    grad_b = np.outer((a.T @ residual) * dphi, x)
    rates = -2.0 * (
        np.sum(a * grad_a, axis=0) - np.sum(b * grad_b, axis=1)
    )
    return float(np.max(np.abs(rates)))


def gelu_silu_candidate_drift(seed: int) -> dict[str, float]:
    """Return rates for a ReLU-style candidate which is not conserved."""

    return {
        "gelu": _ffn_candidate_rate(seed, "gelu"),
        "silu": _ffn_candidate_rate(seed + 1000, "silu"),
    }


def _swiglu_gradients(
    a: Array, b: Array, c: Array, x: Array, target: Array, output_scale: float = 1.0
) -> tuple[Array, Array, Array]:
    z = b @ x
    gate = silu(z)
    value = c @ x
    hidden = gate * value
    output = a @ hidden
    residual = output_scale * (output - target)
    grad_a = np.outer(residual, hidden)
    grad_hidden = a.T @ residual
    grad_b = np.outer(grad_hidden * value * silu_prime(z), x)
    grad_c = np.outer(grad_hidden * gate, x)
    return grad_a, grad_b, grad_c


def swiglu_residual(seed: int) -> float:
    rng = np.random.default_rng(seed)
    d, hidden = 8, 12
    a = rng.normal(scale=0.2, size=(d, hidden))
    b = rng.normal(scale=0.2, size=(hidden, d))
    c = rng.normal(scale=0.2, size=(hidden, d))
    x, target = rng.normal(size=d), rng.normal(size=d)
    grad_a, _, grad_c = _swiglu_gradients(a, b, c, x, target)
    rates = 2.0 * (
        np.sum(a * grad_a, axis=0) - np.sum(c * grad_c, axis=1)
    )
    return float(np.max(np.abs(rates)))


def _softmax_rows(scores: Array) -> Array:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def _attention_gradients(
    x: Array,
    target: Array,
    q: Array,
    k: Array,
    v: Array,
    o: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    scores = (x @ q) @ (x @ k).T
    probs = _softmax_rows(scores)
    xv = x @ v
    values = xv @ o.T
    output = probs @ values
    residual = output - target

    grad_values = probs.T @ residual
    grad_probs = residual @ values.T
    grad_scores = probs * (
        grad_probs - np.sum(grad_probs * probs, axis=1, keepdims=True)
    )
    grad_q = x.T @ grad_scores @ x @ k
    grad_k = x.T @ grad_scores.T @ x @ q
    grad_v = x.T @ grad_values @ o
    grad_o = grad_values.T @ xv
    return output, grad_q, grad_k, grad_v, grad_o


def attention_residual(seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    length, d, head = 5, 8, 4
    x, target = rng.normal(size=(length, d)), rng.normal(size=(length, d))
    q = rng.normal(scale=0.15, size=(d, head))
    k = rng.normal(scale=0.15, size=(d, head))
    v = rng.normal(scale=0.15, size=(d, head))
    o = rng.normal(scale=0.15, size=(d, head))
    _, gq, gk, gv, go = _attention_gradients(x, target, q, k, v, o)
    qk = q.T @ gq + gq.T @ q - k.T @ gk - gk.T @ k
    vo = v.T @ gv + gv.T @ v - o.T @ go - go.T @ o
    return {
        "qk": float(np.linalg.norm(qk)),
        "vo": float(np.linalg.norm(vo)),
    }


def attention_gd_drift(seed: int, steps: int = 40, tau: float = 0.02) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    length, d, head = 5, 8, 4
    x, target = rng.normal(size=(length, d)), rng.normal(size=(length, d))
    q = rng.normal(scale=0.15, size=(d, head))
    k = rng.normal(scale=0.15, size=(d, head))
    v = rng.normal(scale=0.15, size=(d, head))
    o = rng.normal(scale=0.15, size=(d, head))
    qk0 = q.T @ q - k.T @ k
    vo0 = v.T @ v - o.T @ o
    for _ in range(steps):
        _, gq, gk, gv, go = _attention_gradients(x, target, q, k, v, o)
        q -= tau * gq
        k -= tau * gk
        v -= tau * gv
        o -= tau * go
    return {
        "qk": float(np.linalg.norm(q.T @ q - k.T @ k - qk0)),
        "vo": float(np.linalg.norm(v.T @ v - o.T @ o - vo0)),
    }


def moe_swiglu_residual(seed: int) -> float:
    rng = np.random.default_rng(seed)
    d, hidden, experts = 8, 12, 4
    x, target = rng.normal(size=d), rng.normal(size=d)
    w = rng.normal(scale=0.2, size=(experts, d))
    gates = np.exp(w @ x - np.max(w @ x))
    gates /= np.sum(gates)
    params: list[tuple[Array, Array, Array]] = []
    outputs: list[Array] = []
    for _ in range(experts):
        a = rng.normal(scale=0.2, size=(d, hidden))
        b = rng.normal(scale=0.2, size=(hidden, d))
        c = rng.normal(scale=0.2, size=(hidden, d))
        params.append((a, b, c))
        outputs.append(a @ (silu(b @ x) * (c @ x)))
    output = sum(g * y for g, y in zip(gates, outputs, strict=True))
    residual = output - target
    max_rate = 0.0
    for gate, (a, b, c) in zip(gates, params, strict=True):
        z = b @ x
        hidden = silu(z) * (c @ x)
        grad_a = np.outer(gate * residual, hidden)
        grad_c = np.outer((a.T @ (gate * residual)) * silu(z), x)
        rates = 2.0 * (
            np.sum(a * grad_a, axis=0) - np.sum(c * grad_c, axis=1)
        )
        max_rate = max(max_rate, float(np.max(np.abs(rates))))
    return max_rate


def tau2_scaling(seed: int) -> tuple[list[float], list[float], float]:
    rng = np.random.default_rng(seed)
    d, hidden = 8, 12
    a = rng.normal(scale=0.2, size=(d, hidden))
    b = rng.normal(scale=0.2, size=(hidden, d))
    c = rng.normal(scale=0.2, size=(hidden, d))
    x, target = rng.normal(size=d), rng.normal(size=d)
    grad_a, _, grad_c = _swiglu_gradients(a, b, c, x, target)
    invariant0 = np.sum(a * a, axis=0) - np.sum(c * c, axis=1)
    taus = np.array([1e-4, 2e-4, 5e-4, 1e-3], dtype=np.float64)
    drifts = []
    for tau in taus:
        a1 = a - tau * grad_a
        c1 = c - tau * grad_c
        invariant1 = np.sum(a1 * a1, axis=0) - np.sum(c1 * c1, axis=1)
        drifts.append(float(np.linalg.norm(invariant1 - invariant0)))
    slope = float(np.polyfit(np.log(taus), np.log(np.array(drifts)), 1)[0])
    return taus.tolist(), drifts, slope
