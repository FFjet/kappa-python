"""Multi-component diffusion coefficients (port of TestDiffusion.cpp)."""
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
    pressure = 101325.0
    x_atom = 0.10  # 10% atomic fraction
    total_n = pressure / (constants.K_CONST_K * temperature)

    mol_ndens = [mixture.Boltzmann_distribution(temperature, (1.0 - x_atom) * total_n, molecule, 0)]
    atom_ndens = np.array([x_atom * total_n])

    mixture.compute_transport_coefficients(
        temperature,
        n_vl_molecule=mol_ndens,
        n_atom=atom_ndens,
        model=ModelsOmega.RS,
        perturbation=0.0,
    )

    diffusion = mixture.get_diffusion()
    print(f"{'Temperature [K]':>20s}")
    print(f"{temperature:20.1f}")
    print("mixture.get_diffusion()")
    print(diffusion)

    out_path = Path.cwd() / f"diff_{molecule.name}_{atom.name}_xat{int(x_atom * 100)}.txt"
    with out_path.open("w") as fh:
        fh.write(f"{'Temperature [K]':>20s}\n")
        fh.write(f"{temperature:20.1f}\n")
        np.savetxt(fh, diffusion.reshape(1, -1))
    print(f"Wrote diffusion coefficients to {out_path}")


if __name__ == "__main__":
    main()
