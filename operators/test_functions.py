from __future__ import annotations

import cmath
import math
from typing import Sequence

from helper_funcs import _as_float_seq, _dot
from .custom_types import Observable


def coordinate(i: int) -> Observable[tuple[float, ...]]:
    """f(x)=x_i"""

    def f(x: tuple[float, ...]) -> float:
        return float(x[i])

    return f


def linear(v: Sequence[float]) -> Observable[tuple[float, ...]]:
    """f(x)=<v,x>"""
    v_ = [float(vi) for vi in v]

    def f(x: tuple[float, ...]) -> float:
        return _dot(v_, _as_float_seq(x))

    return f


def squared_norm(p: float = 2.0) -> Observable[tuple[float, ...]]:
    """Energy/growth function: f(x)=||x||_p^2 (default p=2)."""
    if p <= 0:
        raise ValueError("p must be > 0")

    def f(x: tuple[float, ...]) -> float:
        xs = _as_float_seq(x)
        if p == 2.0:
            return float(sum(xi * xi for xi in xs))
        # ||x||_p = (Σ|xi|^p)^{1/p}, then squared
        norm_p = sum(abs(xi) ** p for xi in xs) ** (1.0 / p)
        return float(norm_p * norm_p)

    return f


def monomial(powers: Sequence[int]) -> Observable[tuple[float, ...]]:
    """Moment monomial: f(x)=∏_i x_i^{powers[i]} (powers are nonnegative ints)."""
    pw = list(powers)
    if any(k < 0 for k in pw):
        raise ValueError("powers must be nonnegative")

    def f(x: tuple[float, ...]) -> float:
        xs = _as_float_seq(x)
        if len(xs) != len(pw):
            raise ValueError("dimension mismatch: x vs powers")
        out = 1.0
        for xi, ki in zip(xs, pw):
            out *= xi**ki
        return float(out)

    return f


def sin_frequency(xi: Sequence[float]) -> Observable[tuple[float, ...]]:
    """Oscillator: f(x)=sin(<xi,x>). Useful for Fourier / spectral heuristics."""
    xi_ = [float(a) for a in xi]

    def f(x: tuple[float, ...]) -> float:
        return math.sin(_dot(xi_, _as_float_seq(x)))

    return f


def complex_exponential(xi: Sequence[float]) -> Observable[tuple[float, ...]]:
    """Complex exponential: f(x)=exp(i <xi,x>). (Returns complex.)"""
    xi_ = [float(a) for a in xi]

    def f(x: tuple[float, ...]) -> complex:
        return cmath.exp(1j * _dot(xi_, _as_float_seq(x)))

    return f


def payoff_call(strike: float, idx: int = 0) -> Observable[tuple[float, ...]]:
    """European call payoff: f(x)=max(x[idx]-K, 0)."""
    K = float(strike)

    def f(x: tuple[float, ...]) -> float:
        return max(float(x[idx]) - K, 0.0)

    return f


def payoff_put(strike: float, idx: int = 0) -> Observable[tuple[float, ...]]:
    """European put payoff: f(x)=max(K-x[idx], 0)."""
    K = float(strike)

    def f(x: tuple[float, ...]) -> float:
        return max(K - float(x[idx]), 0.0)

    return f