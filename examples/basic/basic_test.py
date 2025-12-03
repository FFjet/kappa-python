"""Basic smoke tests for the Python kappa port."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from kappa import numerics  # noqa: E402


def main() -> None:
    # YAML parsing check
    node = yaml.safe_load("{pi: 3.14159, integers: [0, 1]}")
    print("pi from YAML:", node["pi"])

    # Linear algebra / numeric helpers
    v1 = [1.0] * 5
    v2 = [4.0, 1.0, 1.0, 1.0, 1.0]
    dot = sum(a * b for a, b in zip(v1, v2))
    print("(v1, v2) dot product:", dot)
    print("5! =", numerics.factorial_table[5], "69! =", numerics.factorial_table[69])

    test_f = lambda i: 0.5 + i  # noqa: E731
    test_f_integration = lambda x: x  # noqa: E731
    print("int(f)_0^1 =", numerics.integrate_interval(test_f_integration, 0.0, 1.0))
    print("Max i =", numerics.find_max_value(test_f, 7.0))


if __name__ == "__main__":
    main()
