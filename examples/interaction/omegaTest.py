"""Omega integrals for several models (port of omegaTest.cpp)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from kappa.approximations import Approximation  # noqa: E402
from kappa.interactions import Interaction  # noqa: E402
from kappa.models import ModelsOmega  # noqa: E402
from kappa.particles import Atom, Molecule  # noqa: E402


def write_block(interaction: Interaction, appr: Approximation, models: list[ModelsOmega], t_vals: list[float], out_path: Path, l: int, r: int) -> None:
    with out_path.open("w") as fh:
        fh.write(f"{'Temperature [K]':>20s}{'Collision diameter':>20s}{'Collision mass':>20s}")
        for model in models:
            fh.write(f"{model.name:>20s}")
        fh.write("\n")
        for T in t_vals:
            fh.write(f"{T:20.0f}{interaction.collision_diameter:20.6e}{interaction.collision_mass:20.6e}")
            for model in models:
                try:
                    val = appr.omega_integral(T, interaction, l, r, model, True)
                    fh.write(f"{val:20.8e}")
                except Exception:
                    fh.write(f"{'':>20s}")
            fh.write("\n")


def main() -> None:
    data_dir = Path(os.environ.get("KAPPA_DATA_DIRECTORY", ROOT.parent / "data"))
    particles_yaml = data_dir / "particles.yaml"
    interactions_yaml = data_dir / "interaction.yaml"

    appr = Approximation()
    n2 = Molecule("N2", True, True, particles_yaml)
    n = Atom("N", particles_yaml)

    molecules = [n2]
    atoms = [n]
    t_vals = [500.0]
    models = [ModelsOmega.RS, ModelsOmega.VSS, ModelsOmega.BORNMAYER, ModelsOmega.LENNARD_JONES, ModelsOmega.ESA]
    l = r = 2

    out_dir = Path.cwd()

    for i, mol1 in enumerate(molecules):
        for mol2 in molecules[i:]:
            inter = Interaction(mol1, mol2, interactions_yaml)
            out_path = out_dir / f"{mol1.name}_{mol2.name}_{l}_{r}.txt"
            write_block(inter, appr, models, t_vals, out_path, l, r)

    for mol in molecules:
        for at in atoms:
            inter = Interaction(mol, at, interactions_yaml)
            out_path = out_dir / f"{mol.name}_{at.name}_{l}_{r}.txt"
            write_block(inter, appr, models, t_vals, out_path, l, r)

    for i, at1 in enumerate(atoms):
        for at2 in atoms[i:]:
            inter = Interaction(at1, at2, interactions_yaml)
            out_path = out_dir / f"{at1.name}_{at2.name}_{l}_{r}.txt"
            write_block(inter, appr, models, t_vals, out_path, l, r)


if __name__ == "__main__":
    main()
