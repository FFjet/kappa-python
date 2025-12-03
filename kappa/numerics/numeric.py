from __future__ import annotations

import math
from typing import Callable, Tuple

import numpy as np
from scipy import integrate

from .. import constants

# coefficients for Born-Mayer potential, Kustova-Nagnibeda (5.100)
Born_Mayer_coeff_array: np.ndarray = np.array(
    [
        [-267.0, 201.570, 174.672, 54.305],
        [26700.0, -19226.5, -27693.8, -10860.9],
        [-8.9e5, 6.3201e5, 1.0227e6, 5.4304e5],
        [-33.0838, 20.0862, 72.1059, 68.5001],
        [101.571, -56.4472, -286.393, -315.4531],
        [-87.7036, 46.3130, 277.146, 363.1807],
    ],
    dtype=float,
)


def compute_factorial_table(size: int = 70) -> np.ndarray:
    """Pre-compute a small factorial table mirroring the Armadillo fixed vector."""
    return np.asarray([math.factorial(i) for i in range(size)], dtype=float)


factorial_table: np.ndarray = compute_factorial_table()


def factorial(n: int) -> float:
    """Calculate n! using the Python stdlib for clarity."""
    if n < 0:
        raise ValueError("Factorial is only defined for non-negative integers")
    return float(math.factorial(n))


def fact_div_fact(start: int, end: int) -> float:
    """Compute end! / start! for integer bounds."""
    if end < start:
        return 1.0
    prod = 1.0
    for i in range(start + 1, end + 1):
        prod *= float(i)
    return prod


def convert_cm_to_joule(x: float) -> float:
    """Convert a wavenumber value in cm^-1 to Joules."""
    return constants.K_CONST_H * constants.K_CONST_C * 100.0 * x


def integrate_interval(
    func: Callable[[float], float],
    a: float,
    b: float,
    *,
    return_error: bool = False,
    quad_limit: int | None = None,
) -> float | Tuple[float, float]:
    """
    Integrate a scalar function on the finite interval [a, b] using scipy.quad.

    Setting return_error to True returns a (result, estimated_error) tuple,
    mimicking the optional error estimate in the C++ API.
    """
    if b <= a:
        raise ValueError("Upper integration limit must exceed the lower limit")
    result, err = integrate.quad(func, a, b, limit=quad_limit or 200)
    return (result, err) if return_error else result


def integrate_semi_inf(
    func: Callable[[float], float],
    a: float = 0.0,
    subdivisions: int = constants.K_CONST_SUBDIVISIONS,
    *,
    return_error: bool = False,
    quad_limit: int | None = None,
) -> float | Tuple[float, float]:
    """
    Integrate a scalar function on [a, +infinity) using the same change of variables
    as the C++ implementation, dispatched to scipy.quad on sub-intervals of [0, 1].
    """
    if subdivisions <= 0:
        raise ValueError("Subdivisions must be positive")

    step = 1.0 / float(subdivisions)
    total = 0.0
    total_err = 0.0

    def transformed(t: float) -> float:
        return func(a + (1.0 - t) / t) / (t * t)

    for idx in range(subdivisions):
        left = idx * step
        right = (idx + 1) * step
        res, err = integrate.quad(transformed, left, right, limit=quad_limit or 200)
        total += res
        total_err += err

    return (total, total_err) if return_error else total


def find_max_value(func: Callable[[int], float], max_value: float, start: int = 0, *, limit: int | None = None) -> int:
    """
    Find integer i such that f(i) < max_value and f(i+1) > max_value.
    Returns -1 if nothing is found up to the provided limit.
    """
    upper_bound = limit if limit is not None else (2**31 - 2)
    for i in range(start, upper_bound):
        if func(i) < max_value < func(i + 1):
            return i
    return -1
