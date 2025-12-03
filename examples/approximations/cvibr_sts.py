"""Specific vibrational heat capacity in the STS approximation."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from kappa.approximations import Approximation  # noqa: E402
from kappa.particles import Molecule  # noqa: E402


def main() -> None:
    data_dir = Path(os.environ.get("KAPPA_DATA_DIRECTORY", ROOT.parent / "data"))
    particles_yaml = data_dir / "particles.yaml"

    # Harmonic spectrum, rigid rotator to mirror the C++ example
    n2 = Molecule("N2", anharmonic_spectrum=False, rigid_rotator=True, filename=particles_yaml)

    approx = Approximation()
    temps = np.arange(300.0, 3000.1, 300.0)

    print(f"Data source: {particles_yaml}")
    print(f"{'T [K]':>12} {'cv_vibr (J/kg/K)':>22}")
    for T in temps:
        cv_vibr = approx.c_vibr_approx(T, n2)
        print(f"{T:12.1f} {cv_vibr:22.6e}")


if __name__ == "__main__":
    main()
