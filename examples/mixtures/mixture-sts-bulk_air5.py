"""Bulk viscosity for the air5 mixture (port of mixture-sts-bulk_air5.cpp)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from kappa import constants  # noqa: E402
from kappa.exceptions import DataNotFoundException  # noqa: E402
from kappa.mixtures import Mixture  # noqa: E402
from kappa.models import ModelsOmega  # noqa: E402
from kappa.particles import Atom, Molecule  # noqa: E402


def read_species(mix_path: Path) -> List[str]:
    """Read species list from a .mix file (skip commented lines)."""
    species: List[str] = []
    n_species: int | None = None
    with mix_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("/"):
                continue
            if n_species is None:
                n_species = int(line)
                continue
            species.append(line)
            if len(species) == n_species:
                break
    if n_species is None or len(species) != n_species:
        raise RuntimeError(f"Could not parse species list from {mix_path}")
    return species


def boltzmann_for_all(mixture: Mixture, temperature: float, number_density: float) -> list[np.ndarray]:
    """Populate each molecule in its ground level with the provided total number density."""
    return [
        mixture.Boltzmann_distribution(temperature, number_density, mol, 0)
        for mol in mixture.molecules
    ]


def main() -> None:
    mixture_name = "air5"
    data_dir = Path(os.environ.get("KAPPA_DATA_DIRECTORY", ROOT.parent / "data"))
    particles_yaml = data_dir / "particles.yaml"
    interactions_yaml = data_dir / "interaction.yaml"
    mix_path = data_dir / "mixtures" / f"{mixture_name}.mix"

    species = read_species(mix_path)

    molecules: List[Molecule] = []
    atoms: List[Atom] = []
    for name in species:
        try:
            molecules.append(Molecule(name, anharmonic_spectrum=True, rigid_rotator=False, filename=particles_yaml))
        except DataNotFoundException:
            atoms.append(Atom(name, filename=particles_yaml))

    mixture = Mixture(
        molecules=molecules,
        atoms=atoms,
        interactions_filename=interactions_yaml,
        particles_filename=particles_yaml,
        anharmonic=True,
        rigid_rotators=False,
    )

    # Temperatures (500–40000 K, step 500)
    temperatures = [500.0 + 500.0 * i for i in range(80)]
    pressure = 101325.0
    x_atom = 0.25  # 25% atomic fraction, matches the C++ example
    n_atom_entries = len(atoms) if atoms else 1

    out_dir = Path.cwd() / "TRANSPORT_COEFFICIENTS" / "bulk_viscosity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{mixture_name}_xat25.txt"

    header = f"{'Temperature [K]':>20s}{'zeta':>20s}{'zeta / n_tot':>20s}"
    print(f"Mixture: {mixture_name}")
    print("Species:", mixture.get_names())
    print(header)

    with out_file.open("w") as fh:
        fh.write(header + "\n")
        for T in temperatures:
            total_n = pressure / (constants.K_CONST_K * T)
            mol_ndens = boltzmann_for_all(mixture, T, (1.0 - 2.0 * x_atom) * total_n)
            atom_ndens = np.full(n_atom_entries, x_atom * total_n)

            mixture.compute_transport_coefficients(
                T,
                n_vl_molecule=mol_ndens,
                n_atom=atom_ndens,
                model=ModelsOmega.ESA,
                perturbation=1e-9,
            )

            bulk = mixture.get_bulk_viscosity()
            line = f"{T:20.0f}{bulk:20.8e}{bulk / total_n:20.8e}"
            print(line)
            fh.write(line + "\n")

    print(f"Wrote results to {out_file}")


if __name__ == "__main__":
    main()
