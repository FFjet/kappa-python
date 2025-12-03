"""Mixture demo covering thermodynamic and transport helper calls (C++ mixture-sts-basic.cpp)."""
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
        particle_names="N2, O2, NO, N, O",
        particles_filename=particles_yaml,
        interactions_filename=interactions_yaml,
        anharmonic=True,
        rigid_rotators=True,
    )

    # Build uniform vibrational populations for each molecule
    n_vl = []
    for mol in mixture.molecules:
        lvl = mol.num_vibr_levels[0] if mol.num_vibr_levels else 1
        n_vl.append(np.ones(lvl))
    n_atom = np.zeros(mixture.num_atoms)

    temperature = 8000.0
    density = mixture.compute_density(n_vl, n_atom)
    total_n = mixture.compute_n(n_vl, n_atom)
    pressure = mixture.compute_pressure(temperature, n_vl, n_atom)
    c_tr = mixture.c_tr(n_vl, n_atom)
    c_rot = mixture.c_rot(temperature, n_vl, n_atom, 0.0)

    print("Mixture species:", mixture.get_names())
    print("Total number density:", total_n)
    print("Mass density:", density)
    print("Pressure at", temperature, "K:", pressure)
    print("Translational heat capacity (per mass):", c_tr)
    print("Rotational heat capacity (per mass):", c_rot)
    print("Number of interactions:", len(mixture.interactions))


if __name__ == "__main__":
    main()
