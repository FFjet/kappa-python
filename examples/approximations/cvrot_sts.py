"""Rotational specific heat capacity demo (STS approximation)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from kappa import constants  # noqa: E402
from kappa.approximations import Approximation  # noqa: E402
from kappa.particles import Molecule  # noqa: E402


def main() -> None:
    data_dir = Path(os.environ.get("KAPPA_DATA_DIRECTORY", ROOT.parent / "data"))
    particles_yaml = data_dir / "particles.yaml"

    n2 = Molecule("N2", anharmonic_spectrum=False, rigid_rotator=True, filename=particles_yaml)
    approx = Approximation()
    temps = np.arange(300.0, 3000.1, 300.0)
    eq_rot = constants.K_CONST_K / n2.mass  # equipartition check

    print(f"Data source: {particles_yaml}")
    print(f"{'T [K]':>12} {'cv_rot (J/kg/K)':>22} {'k/m':>14}")
    for T in temps:
        cv_rot = approx.c_rot(T, n2, 0, 0)
        print(f"{T:12.1f} {cv_rot:22.6e} {eq_rot:14.6e}")


if __name__ == "__main__":
    main()
