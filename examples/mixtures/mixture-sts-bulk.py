"""Bulk viscosity sweep for N2/N (C++ mixture-sts-bulk.cpp)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from kappa import constants  # noqa: E402
from kappa.mixtures import Mixture  # noqa: E402
from kappa.models import ModelsOmega  # noqa: E402
from kappa.particles import Atom, Molecule  # noqa: E402


def main() -> None:
    data_dir = Path(os.environ.get("KAPPA_DATA_DIRECTORY", ROOT.parent / "data"))
    particles_yaml = data_dir / "particles.yaml"
    interactions_yaml = data_dir / "interaction.yaml"

    # N2 + N, non-rigid rotator, RS model, perturbation 0
    mixture = Mixture(
        molecules=[Molecule("N2", anharmonic_spectrum=True, rigid_rotator=False, filename=particles_yaml)],
        atoms=[Atom("N", filename=particles_yaml)],
        particles_filename=particles_yaml,
        interactions_filename=interactions_yaml,
        anharmonic=True,
        rigid_rotators=False,
    )

    x_N2 = 0.90
    x_N = 0.10

    temps = np.linspace(500.0, 500.0 + 194 * 100.0, 195)  # 500 to 40000 step 100
    print(f"{'T [K]':>10s} {'bulk viscosity (RS)':>22s}")
    for T in temps:
        total_n = 101325.0 / (constants.K_CONST_K * T)
        n_vl = [mixture.Boltzmann_distribution(T, x_N2 * total_n, mixture.molecules[0], 0)]
        n_atom = [x_N * total_n]
        mixture.compute_transport_coefficients(T, n_vl_molecule=n_vl, n_atom=n_atom, model=ModelsOmega.RS, perturbation=0.0)
        print(f"{T:10.1f} {mixture.get_bulk_viscosity():22.8e}")


if __name__ == "__main__":
    main()
