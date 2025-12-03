"""Multi-temperature style vibrational heat capacity sweeps."""
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

    n2_h = Molecule("N2", anharmonic_spectrum=False, rigid_rotator=True, filename=particles_yaml)
    n2_anh = Molecule("N2", anharmonic_spectrum=True, rigid_rotator=True, filename=particles_yaml)
    o2_h = Molecule("O2", anharmonic_spectrum=False, rigid_rotator=True, filename=particles_yaml)
    o2_anh = Molecule("O2", anharmonic_spectrum=True, rigid_rotator=True, filename=particles_yaml)

    approx = Approximation()

    temps = np.arange(500.0, 20000.1, 2500.0)
    print(f"Data source: {particles_yaml}")
    print("Vibrational heat capacity sweep (harmonic vs anharmonic)")
    for name, harm, anh in [
        ("N2", n2_h, n2_anh),
        ("O2", o2_h, o2_anh),
    ]:
        print(f"\n{name}:")
        print(f"{'T [K]':>12} {'cv_vibr harmonic':>22} {'cv_vibr anharm':>22}")
        for T in temps:
            cv_h = approx.c_vibr_approx(T, harm)
            cv_a = approx.c_vibr_approx(T, anh)
            print(f"{T:12.1f} {cv_h:22.6e} {cv_a:22.6e}")


if __name__ == "__main__":
    main()
