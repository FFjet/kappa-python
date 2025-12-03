"""Thermo-diffusion coefficients for N2/N (C++ mixture-sts-thermo-diffusion.cpp)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from kappa import constants  # noqa: E402
from kappa.mixtures import Mixture  # noqa: E402
from kappa.particles import Atom, Molecule  # noqa: E402


def main() -> None:
    data_dir = Path(os.environ.get("KAPPA_DATA_DIRECTORY", ROOT.parent / "data"))
    particles_yaml = data_dir / "particles.yaml"
    interactions_yaml = data_dir / "interaction.yaml"

    mixture = Mixture(
        molecules=[Molecule("N2", anharmonic_spectrum=True, rigid_rotator=False, filename=particles_yaml)],
        atoms=[Atom("N", filename=particles_yaml)],
        particles_filename=particles_yaml,
        interactions_filename=interactions_yaml,
        anharmonic=True,
        rigid_rotators=False,
    )

    temps = [2500.0, 5000.0, 20000.0, 50000.0] + list(np.arange(500.0, 500.0 + 100 * 500.0, 500.0))
    x_atom = 0.20

    print(f"{'T [K]':>12s} {'thermo-diffusion':>20s}")
    for T in temps:
        total_n = 101325.0 / (constants.K_CONST_K * T)
        n_vl = [mixture.Boltzmann_distribution(T, (1 - x_atom) * total_n, mixture.molecules[0], 0)]
        n_atom = [x_atom * total_n]
        mixture.compute_transport_coefficients(T, n_vl_molecule=n_vl, n_atom=n_atom)
        thd = mixture.get_thermodiffusion()
        for val in thd:
            print(f"{T:12.1f} {val:20.10e}")


if __name__ == "__main__":
    main()
