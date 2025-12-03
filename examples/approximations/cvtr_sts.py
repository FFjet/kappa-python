"""Translational specific heat capacity demo (STS approximation)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from kappa import constants  # noqa: E402
from kappa.approximations import Approximation  # noqa: E402
from kappa.particles import Atom, Molecule  # noqa: E402


def main() -> None:
    data_dir = Path(os.environ.get("KAPPA_DATA_DIRECTORY", ROOT.parent / "data"))
    particles_yaml = data_dir / "particles.yaml"

    n2 = Molecule("N2", anharmonic_spectrum=False, rigid_rotator=True, filename=particles_yaml)
    n_atom = Atom("N", filename=particles_yaml)
    approx = Approximation()
    temps = np.arange(300.0, 3000.1, 300.0)

    print(f"Data source: {particles_yaml}")
    header = f"{'T [K]':>12} {'cv_tr(N2)':>16} {'cv_tr(N)':>16} {'1.5k/m(N2)':>16} {'1.5k/m(N)':>16}"
    print(header)
    for T in temps:
        cv_tr_n2 = approx.c_tr(T, n2)
        cv_tr_n = approx.c_tr(T, n_atom)
        classical_n2 = 1.5 * constants.K_CONST_K / n2.mass
        classical_n = 1.5 * constants.K_CONST_K / n_atom.mass
        print(f"{T:12.1f} {cv_tr_n2:16.6e} {cv_tr_n:16.6e} {classical_n2:16.6e} {classical_n:16.6e}")


if __name__ == "__main__":
    main()
