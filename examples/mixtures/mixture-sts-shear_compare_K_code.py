"""Shear viscosity check against K code (port of mixture-sts-shear_compare_K_code.cpp)."""
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

    # Mixture: N2 + N
    mixture = Mixture(
        molecules=[Molecule("N2", anharmonic_spectrum=True, rigid_rotator=False, filename=particles_yaml)],
        atoms=[Atom("N", filename=particles_yaml)],
        particles_filename=particles_yaml,
        interactions_filename=interactions_yaml,
        anharmonic=True,
        rigid_rotators=False,
    )

    temperature = 9971.9
    pressure = 27358.0

    x_mass = np.zeros(mixture.num_molecules + mixture.num_atoms)
    x_mass[0] = 9.991800e-01  # N2
    x_mass[1] = 1.252500e-03  # N
    x_molar = mixture.convert_mass_frac_to_molar(x_mass)

    x_N2 = x_molar[0]
    x_N = x_molar[1]

    total_n = pressure / (constants.K_CONST_K * temperature)
    mol_ndens = [
        mixture.Boltzmann_distribution(temperature, x_N2 * total_n, mixture.molecules[0], 0),
    ]
    atom_ndens = np.array([x_N * total_n])

    mixture.compute_transport_coefficients(
        temperature,
        n_vl_molecule=mol_ndens,
        n_atom=atom_ndens,
        model=ModelsOmega.RS,
    )

    out_dir = Path.cwd() / "TRANSPORT_COEFFICIENTS" / "shear_viscosity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{mixture.molecules[0].name}_xat_{x_N2}.txt"

    header = f"{'Temperature [K]':>20s}{'Eta':>20s}"
    print(header)
    line = f"{temperature:20.1f}{mixture.get_shear_viscosity():20.8e}"
    print(line)

    with out_file.open("w") as fh:
        fh.write(header + "\n")
        fh.write(line + "\n")

    print(f" bulk viscosity = {mixture.get_bulk_viscosity():.8e}")
    print(f" thermal conductivity = {mixture.get_thermal_conductivity():.8e}")
    print(" thermal diffusion =", mixture.get_thermodiffusion())
    print(" n_molecule[0] =", mol_ndens[0])
    print(f"Wrote shear viscosity to {out_file}")


if __name__ == "__main__":
    main()
