"""Shear, bulk viscosity and thermal conductivity for a pure N2 mixture (STS, non-rigid) (C++ mixture-sts-shear-bulk-thermal.cpp)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

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

    mixture = Mixture(
        molecules=[Molecule("N2", anharmonic_spectrum=True, rigid_rotator=False, filename=particles_yaml)],
        atoms=[Atom("N", filename=particles_yaml)],
        particles_filename=particles_yaml,
        interactions_filename=interactions_yaml,
        anharmonic=True,
        rigid_rotators=False,
    )

    T_vals = [500.0, 1000.0, 5000.0, 10000.0, 15000.0, 20000.0, 25000.0, 30000.0, 35000.0, 40000.0]
    x_N = 0.0  # atomic nitrogen fraction

    print(f"Data source: {particles_yaml}")
    print("Species:", mixture.get_names())
    print(f"{'temperature':>15s} {'shear viscosity':>20s} {'bulk viscosity':>20s} {'thermal conductivity':>24s}")

    for T in T_vals:
        total_n = 101325.0 / (constants.K_CONST_K * T)
        n_vl = [mixture.Boltzmann_distribution(T, (1.0 - x_N) * total_n, mixture.molecules[0], 0)]
        n_atom = [x_N * total_n]
        mixture.compute_transport_coefficients(T, n_vl_molecule=n_vl, n_atom=n_atom, model=ModelsOmega.ESA)
        shear = mixture.get_shear_viscosity()
        bulk = mixture.get_bulk_viscosity()
        th_cond = mixture.get_thermal_conductivity()
        print(f"{T:15.0f} {shear:20.8e} {bulk:20.8e} {th_cond:24.6f}")


if __name__ == "__main__":
    main()
