"""Particle, atom, and molecule loading demo mirroring the C++ example."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from kappa import exceptions  # noqa: E402
from kappa.particles import Atom, Molecule, Particle  # noqa: E402


def main() -> None:
    print("Start Test for Particle classes")
    data_dir = Path(os.environ.get("KAPPA_DATA_DIRECTORY", ROOT.parent / "data"))
    particle_source = data_dir / "particles.yaml"
    interaction_source = data_dir / "interaction.yaml"
    print("particle_source:", particle_source)
    print("interaction_source:", interaction_source)

    try:
        Particle("Ar", "articles.yaml")
    except exceptions.UnopenedFileException as exc:
        print("Caught expected missing file error:", exc)

    try:
        Particle("B", particle_source)
    except exceptions.DataNotFoundException as exc:
        print("Caught expected missing particle error:", exc)

    # Load a handful of particles
    names = ["Ar", "C", "e-", "N", "N+", "O", "O+", "O-"]
    particles = [Particle(name, particle_source) for name in names]
    print("Loaded particles:", [p.name for p in particles])

    atoms = [Atom(name, particle_source) for name in names if not name.endswith("-")]
    print("Loaded atoms:", [a.name for a in atoms])

    molecules = [
        Molecule("C2", True, False, particle_source),
        Molecule("CO", True, False, particle_source),
        Molecule("N2", True, False, particle_source),
    ]
    co = molecules[1]
    n2 = molecules[2]
    print("Number of electron levels in CO:", co.num_electron_levels)
    print("Vibrational spectrum anharmonic for CO:", co.anharmonic_spectrum)
    print("Rigid rotator model used for CO:", co.rigid_rotator)
    print("N2 vibrational levels (ground):", n2.num_vibr_levels[0])

    try:
        Molecule("CO", True, False, "articles.yaml")
    except exceptions.UnopenedFileException as exc:
        print("Caught expected missing file error for molecule:", exc)

    molecules_harm = [
        Molecule("N2", False, False, particle_source),
    ]
    print("N2 vibrational levels (harmonic approx):", molecules_harm[0].num_vibr_levels[0])

    wrong_atom = Atom("N2", particle_source)
    print("Loaded N2 as atom:", wrong_atom.name)


if __name__ == "__main__":
    main()
