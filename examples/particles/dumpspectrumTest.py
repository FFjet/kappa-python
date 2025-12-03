"""Dump vibrational spectra (port of dumpspectrumTest.cpp)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from kappa import constants  # noqa: E402
from kappa.approximations import Approximation  # noqa: E402
from kappa.particles import Molecule  # noqa: E402


def write_spectrum(filename: Path, energies: list[float]) -> None:
    with filename.open("w") as fh:
        fh.write(",".join(f"{e:.12e}" for e in energies))


def main() -> None:
    data_dir = Path(os.environ.get("KAPPA_DATA_DIRECTORY", ROOT.parent / "data"))
    particle_source = data_dir / "particles.yaml"
    interaction_source = data_dir / "interaction.yaml"
    output_dir = Path.cwd()

    print("KAPPA_DATA_DIRECTORY is:", data_dir)
    print("Current directory is:", output_dir)
    print("Particle directory is:", particle_source)
    print("Interaction directory is:", interaction_source)

    n2 = Molecule("N2", True, False, particle_source)
    no = Molecule("NO", True, True, particle_source)
    o2 = Molecule("O2", True, True, particle_source)

    appr = Approximation()

    n2_energies = []
    for i in range(n2.num_vibr_levels[0]):
        e = n2.vibr_energy[0][i]
        n2_energies.append(e / constants.K_CONST_EV)
        # Use electronic level 0 as in the C++ example; i is the vibrational index.
        print(i, e - n2.vibr_energy[0][0], appr.avg_rot_energy(300.0, n2, i, 0))
    write_spectrum(output_dir / "N2_spectrum.csv", n2_energies)

    no_energies = [no.vibr_energy[0][i] / constants.K_CONST_EV for i in range(no.num_vibr_levels[0])]
    write_spectrum(output_dir / "NO_spectrum.csv", no_energies)

    o2_energies = [o2.vibr_energy[0][i] / constants.K_CONST_EV for i in range(o2.num_vibr_levels[0])]
    write_spectrum(output_dir / "O2_spectrum.csv", o2_energies)


if __name__ == "__main__":
    main()
