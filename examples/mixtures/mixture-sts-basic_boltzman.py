"""Compute Boltzmann populations and mass fractions (C++ mixture-sts-basic_boltzman.cpp)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from kappa import constants  # noqa: E402
from kappa.mixtures import Mixture  # noqa: E402
from kappa.particles import Atom, Molecule  # noqa: E402


def read_mix_species(mix_path: Path) -> List[str]:
    species: List[str] = []
    with mix_path.open() as f:
        first = None
        for line in f:
            line = line.strip()
            if not line or line.startswith("/"):
                continue
            if first is None:
                first = int(line)
                continue
            species.append(line)
            if len(species) >= first:
                break
    return species


def main() -> None:
    data_dir = Path(os.environ.get("KAPPA_DATA_DIRECTORY", ROOT.parent / "data"))
    mix_name = "N2"
    particles_yaml = data_dir / "particles.yaml"
    interactions_yaml = data_dir / "interaction.yaml"
    mix_file = data_dir / "mixtures" / f"{mix_name}.mix"

    names = read_mix_species(mix_file)
    molecules: List[Molecule] = []
    atoms: List[Atom] = []
    for name in names:
        if name == "e-":
            continue
        if any(ch.isdigit() for ch in name) or len(name) > 1:
            molecules.append(Molecule(name, anharmonic_spectrum=True, rigid_rotator=False, filename=particles_yaml))
        else:
            atoms.append(Atom(name, filename=particles_yaml))

    mixture = Mixture(
        molecules=molecules,
        atoms=atoms,
        particles_filename=particles_yaml,
        interactions_filename=interactions_yaml,
        anharmonic=True,
        rigid_rotators=False,
    )

    print("Mixture species:", mixture.get_names())
    print("Number of species:", mixture.get_n_particles())

    T0 = 1833.0
    p0 = 2908.8
    total_n = p0 / (constants.K_CONST_K * T0)
    x_N = 0.0

    n_vl = [mixture.Boltzmann_distribution(T0, (1.0 - x_N) * total_n, mol, 0) for mol in mixture.molecules]
    n_atom = np.array([x_N * total_n] * len(atoms))

    for vec in n_vl:
        print(vec)

    # flatten mol populations into a single vector (mirror C++)
    flat = np.concatenate(n_vl)
    mol_mass = mixture.molecules[0].mass if mixture.molecules else 0.0
    # total mass density
    rho = mixture.compute_density(n_vl, n_atom, 0.0)
    print(f" density = {rho} at temperature = {T0} and pressure = {p0}")
    # mass fractions
    mass_frac = (flat * mol_mass) / rho if rho else flat * 0.0
    for val in mass_frac:
        print(" rho_ci = ", val * rho)


if __name__ == "__main__":
    main()
