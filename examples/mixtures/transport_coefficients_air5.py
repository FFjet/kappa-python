"""Compute transport coefficients for an air-like mixture (STS-style)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from kappa.exceptions import DataNotFoundException  # noqa: E402
from kappa.mixtures import Mixture  # noqa: E402
from kappa.models import ModelsOmega  # noqa: E402
from kappa.particles import Atom, Molecule  # noqa: E402
from kappa import constants  # noqa: E402


def ground_state_population(molecule: Molecule, number_density: float) -> np.ndarray:
    """Place the entire molecular number density in the ground vibrational level."""
    levels = molecule.num_vibr_levels[0] if molecule.num_vibr_levels else 1
    arr = np.zeros(levels)
    arr[0] = number_density
    return arr


def main() -> None:
    data_dir = Path(os.environ.get("KAPPA_DATA_DIRECTORY", ROOT.parent / "data"))
    particles_yaml = data_dir / "particles.yaml"
    interactions_yaml = data_dir / "interaction.yaml"

    # Load species; skip NO if it is not present in the database
    molecules: List[Molecule] = []
    for name in ("N2", "O2", "NO"):
        molecules.append(Molecule(name, anharmonic_spectrum=True, rigid_rotator=False, filename=particles_yaml))
    atoms = [
        Atom("N", filename=particles_yaml),
        Atom("O", filename=particles_yaml),
    ]

    mixture = Mixture(
        molecules=molecules,
        atoms=atoms,
        interactions_filename=interactions_yaml,
        particles_filename=particles_yaml,
        anharmonic=True,
        rigid_rotators=False,
    )

    # Composition fractions (mole fractions)
    x_N2 = 0.9
    x_O2 = 0.0
    x_NO = 0.0
    x_N = 0.10
    x_O = 0.0

    temperature = 500.0  # K
    total_n = 101325.0 / (constants.K_CONST_K * temperature)

    n_vl = []
    for mol in mixture.molecules:
        if mol.name == "N2":
            n_i = (1.0 - x_N2) * total_n
        elif mol.name == "O2":
            n_i = (1.0 - x_O2) * total_n
        elif mol.name == "NO":
            n_i = (1.0 - x_NO) * total_n
        else:
            n_i = 0.0
        n_vl.append(mixture.Boltzmann_distribution(temperature, n_i, mol, 0))
    n_atom = np.array([x_N * total_n, x_O * total_n])

    # Compute transport coefficients (RS model)
    mixture.compute_transport_coefficients(
        temperature,
        n_vl_molecule=n_vl,
        n_atom=n_atom,
        model=ModelsOmega.RS,
        perturbation=0.0,
    )

    print(f"Data source: {particles_yaml}")
    print("Species:", mixture.get_names())
    print(f"Temperature: {temperature} K")
    print(f"Total number density: {total_n:.3e} 1/m^3")
    print(f"Thermal conductivity (RS): {mixture.get_thermal_conductivity():.6e} W/m/K")
    print(f"Shear viscosity (RS): {mixture.get_shear_viscosity():.6e} Pa*s")
    print(f"Bulk viscosity (RS): {mixture.get_bulk_viscosity():.6e} Pa*s")


if __name__ == "__main__":
    main()
