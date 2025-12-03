"""Diffusion and thermo-diffusion coefficients for a simple mixture."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from kappa.mixtures import Mixture  # noqa: E402
from kappa.models import ModelsOmega  # noqa: E402


def main() -> None:
    data_dir = Path(os.environ.get("KAPPA_DATA_DIRECTORY", ROOT.parent / "data"))
    particles_yaml = data_dir / "particles.yaml"
    interactions_yaml = data_dir / "interaction.yaml"

    mixture = Mixture(
        particle_names="N2, O2",
        particles_filename=particles_yaml,
        interactions_filename=interactions_yaml,
        anharmonic=True,
        rigid_rotators=True,
    )

    temperature = 6000.0
    total_n = 2.0e19
    mole_frac = np.array([0.5, 0.5])
    n_molecules = mole_frac * total_n
    n_vl = []
    for mol, n_i in zip(mixture.molecules, n_molecules):
        lvl = mol.num_vibr_levels[0] if mol.num_vibr_levels else 1
        n_vl.append(np.full(lvl, n_i / lvl))
    n_atom = np.zeros(mixture.num_atoms)

    # Cache state and precompute supporting arrays
    mixture.this_n_vl_mol = n_vl
    mixture.this_n_atom = n_atom
    mixture.this_n_molecules = n_molecules
    mixture.this_total_n = float(total_n)
    mixture.this_total_dens = mixture.compute_density(n_vl, n_atom)
    mixture.compute_c_rot(temperature)
    mixture.compute_rot_rel_times(temperature, total_n, model=ModelsOmega.RS)
    mixture.compute_omega11(temperature, model=ModelsOmega.RS)
    mixture.compute_omega12(temperature, model=ModelsOmega.RS)
    mixture.compute_omega22(temperature, model=ModelsOmega.RS)
    mixture.compute_thermal_conductivity_LHS(temperature, model=ModelsOmega.RS)

    # Thermodiffusion + diffusion + binary diffusion
    mixture.thermodiffusion(temperature, model=ModelsOmega.RS)
    mixture.diffusion(temperature)

    print(f"Data source: {particles_yaml}")
    print(f"Temperature: {temperature} K")
    print("Species:", mixture.get_names())
    print("Thermo-diffusion vector:", mixture.get_thermodiffusion())
    print("Full diffusion matrix shape:", mixture.get_diffusion().shape)
    print("Lite diffusion matrix shape:", mixture.get_lite_diffusion().shape)
    print("Binary diffusion coeffs:", mixture.get_binary_diffusion())


if __name__ == "__main__":
    main()
