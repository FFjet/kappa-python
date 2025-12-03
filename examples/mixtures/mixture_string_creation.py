"""Create mixtures from a comma-separated species string."""
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

    names = "N2, O2, N, O"
    mixture = Mixture(
        particle_names=names,
        particles_filename=particles_yaml,
        interactions_filename=interactions_yaml,
        anharmonic=True,
        rigid_rotators=True,
    )

    print("Input string:", names)
    print("Mixture species parsed:", mixture.get_names())
    print("Number of particles:", mixture.get_n_particles())
    print("Vibrational level counts per molecule:", mixture.get_n_vibr_levels_array())


if __name__ == "__main__":
    main()
