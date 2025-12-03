"""Ionized mixture creation demo with electron support (C++ ionized_mixture_creation.cpp)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from kappa.mixtures import Mixture  # noqa: E402


def main() -> None:
    data_dir = Path(os.environ.get("KAPPA_DATA_DIRECTORY", ROOT.parent / "data"))
    particles_yaml = data_dir / "particles.yaml"
    interactions_yaml = data_dir / "interaction.yaml"

    names = "N+, O+, e-"
    mixture = Mixture(
        particle_names=names,
        particles_filename=particles_yaml,
        interactions_filename=interactions_yaml,
        anharmonic=True,
        rigid_rotators=True,
    )

    print("Ionized mixture species:", mixture.get_names())
    print("Number of particles:", mixture.get_n_particles())
    print("Is ionized:", mixture.is_ionized)
    print("Contains electron mass:", mixture.electron.mass if mixture.electron else None)


if __name__ == "__main__":
    main()
