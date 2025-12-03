"""Thermal diffusion coefficients (port of thermodiffusion.cpp)."""
from __future__ import annotations

import math
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


def compute_d0(mass: float) -> float:
    """Replicate the d0 calculation from the C++ example."""
    vibr_l = 0.0
    re = 1.097e-10
    be = 2.25e-10
    omega_e = 235860.0  # 1/m
    mu = mass  # kg/mol
    l_alpha = math.sqrt(16.863 / (omega_e * mu))
    beta = 2.6986e10
    return re + be + (9.0 / 2.0) * beta * l_alpha * l_alpha * math.exp(2.0 * math.sqrt(beta * l_alpha) * (vibr_l - 1.0))


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

    temperatures = [500.0]
    pressure = 101325.0
    x_atom = 0.10
    d0 = compute_d0(molecule.mass)

    out_dir = Path.cwd()
    out_file = out_dir / f"thd_{molecule.name}_{atom.name}_xat{int(x_atom * 100)}.txt"
    out_file_sts = out_dir / f"thd_sts_{molecule.name}_{atom.name}_xat{int(x_atom * 100)}.txt"

    header = f"{'Temperature [K]':>20s}{'Lambda':>25s}"
    with out_file.open("w") as fh, out_file_sts.open("w") as fh2:
        fh.write(header + "\n")
        fh2.write(header + "\n")

    for T in temperatures:
        tot_ndens = pressure / (constants.K_CONST_K * T)
        mol_ndens = [mixture.Boltzmann_distribution(T, (1.0 - x_atom) * tot_ndens, molecule, 0)]
        atom_ndens = np.array([x_atom * tot_ndens])

        mixture.compute_transport_coefficients(
            T,
            n_vl_molecule=mol_ndens,
            n_atom=atom_ndens,
            model=ModelsOmega.RS,
            perturbation=0.0,
        )

        d0_scale = (3.0 / (8.0 * tot_ndens * d0 * d0)) * math.sqrt((constants.K_CONST_K * T) / (constants.K_CONST_PI * molecule.mass))
        thd = mixture.get_thermodiffusion()

        print(f"{'Temperature [K]':>20s}")
        print(f"{T:20.1f}")
        print("Thermo-diffusion coefficients (scaled by D0):")
        print(thd / d0_scale)

        with out_file.open("a") as fh, out_file_sts.open("a") as fh2:
            for coeff in thd:
                fh.write(f"{T:20.1f}{coeff / d0_scale:25.18e}\n")
                fh2.write(f"{T:20.1f}{coeff / d0_scale:25.18e}\n")

    print(f"Wrote thermo-diffusion to {out_file} and {out_file_sts}")


if __name__ == "__main__":
    main()
