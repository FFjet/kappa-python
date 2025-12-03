"""Shear viscosity, bulk viscosity, and thermal conductivity for N2/N (STS, non-rigid) at 500 K (C++ mixture-sts-shear.cpp)."""
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

    T = 500.0
    pressure = 101325.0
    total_n = pressure / (constants.K_CONST_K * T)
    x_N2 = 0.90
    x_N = 0.10

    n_vl = [mixture.Boltzmann_distribution(T, x_N2 * total_n, mixture.molecules[0], 0)]
    n_atom = [x_N * total_n]

    mixture.compute_transport_coefficients(T, n_vl_molecule=n_vl, n_atom=n_atom, model=ModelsOmega.RS, perturbation=0.0)

    print(f"Data source: {particles_yaml}")
    print(f"Temperature: {T} K")
    print("Species:", mixture.get_names())
    print(f"Total number density: {total_n:.3e} 1/m^3")
    print(f"Shear viscosity (RS): {mixture.get_shear_viscosity():.6e} Pa*s")
    print(f"Bulk viscosity (RS): {mixture.get_bulk_viscosity():.6e} Pa*s")
    print(f"Thermal conductivity (RS): {mixture.get_thermal_conductivity():.6e} W/m/K")


if __name__ == "__main__":
    main()
