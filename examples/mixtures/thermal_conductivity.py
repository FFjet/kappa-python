"""Thermal conductivity for N2/N mixture (port of thermal_conductivity.cpp)."""
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

    molecule = Molecule("N2", anharmonic_spectrum=True, rigid_rotator=False, filename=particles_yaml)
    atom = Atom("N", filename=particles_yaml)
    mixture = Mixture(
        molecules=[molecule],
        atoms=[atom],
        interactions_filename=interactions_yaml,
        particles_filename=particles_yaml,
        anharmonic=True,
        rigid_rotators=False,
    )

    temperature = 500.0
    tmax = 1000.0
    pressure = 101325.0
    x_atom = 0.10

    out_dir = Path.cwd() / "TRANSPORT_COEFFICIENTS" / "thermal_conductivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{molecule.name}_{atom.name}.txt"

    with out_file.open("w") as fh:
        fh.write(f"{'Temperature [K]':>20s}{'k':>20s}\n")

    while temperature < tmax:
        total_n = pressure / (constants.K_CONST_K * temperature)
        atom_ndens = np.array([x_atom * total_n])
        mol_ndens = [mixture.Boltzmann_distribution(temperature, (1.0 - x_atom) * total_n, molecule, 0)]

        mixture.compute_transport_coefficients(
            temperature,
            n_vl_molecule=mol_ndens,
            n_atom=atom_ndens,
            model=ModelsOmega.RS,
            perturbation=0.0,
        )

        th_c = mixture.get_thermal_conductivity()
        print(f"{temperature:20.1f} {th_c:20.8e}")
        print(" ThermoDiffusion")
        print(mixture.get_thermodiffusion())
        print("mixture.get_diffusion()")
        print(mixture.get_diffusion())

        with out_file.open("a") as fh:
            fh.write(f"{temperature:20.1f}{th_c:20.8e}\n")

        temperature += 500.0

    print(f"Wrote thermal conductivity to {out_file}")


if __name__ == "__main__":
    main()
