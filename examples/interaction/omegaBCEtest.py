"""Omega* coefficients A/B/C/E (port of omegaBCEtest.cpp)."""
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


def write_coeffs(label: str, interaction: Interaction, appr: Approximation, models: list[ModelsOmega], t_vals: list[float], out_path: Path) -> None:
    with out_path.open("w") as fh:
        fh.write("T;")
        for model in models:
            fh.write(f"{model.name} A*;{model.name} B*;{model.name} C*;{model.name} E*;")
        fh.write("A* (F-K approx); B* (Wright); C* (Wright); E* (F-K approx)\n")
        for T in t_vals:
            fh.write(f"{T};")
            for model in models:
                try:
                    o11 = appr.omega_integral(T, interaction, 1, 1, model, False)
                    o22 = appr.omega_integral(T, interaction, 2, 2, model, False)
                    a_val = o22 / o11
                    b_val = (5 * appr.omega_integral(T, interaction, 1, 2, model, False) - 4 * appr.omega_integral(T, interaction, 1, 3, model, False)) / o11
                    c_val = appr.omega_integral(T, interaction, 1, 2, model, False) / o11
                    e_val = appr.omega_integral(T, interaction, 2, 3, model, False) / o22
                    fh.write(f"{a_val};{b_val};{c_val};{e_val};")
                except Exception:
                    fh.write(";;;;")
            fh.write("1.12;1.15;0.92;0.96\n")


def main() -> None:
    data_dir = Path(os.environ.get("KAPPA_DATA_DIRECTORY", ROOT.parent / "data"))
    particles_yaml = data_dir / "particles.yaml"
    interactions_yaml = data_dir / "interaction.yaml"

    appr = Approximation()
    n2 = Molecule("N2", True, True, particles_yaml)
    n = Atom("N", particles_yaml)

    molecules = [n2]
    atoms = [n]
    models = [ModelsOmega.VSS, ModelsOmega.BORNMAYER, ModelsOmega.LENNARD_JONES, ModelsOmega.ESA]
    t_vals = [500.0, 1000.0, 2000.0, 5000.0, 10000.0, 15000.0, 20000.0, 25000.0, 30000.0, 40000.0]

    out_dir = Path.cwd()

    for i, mol1 in enumerate(molecules):
        for mol2 in molecules[i:]:
            inter = Interaction(mol1, mol2, interactions_yaml)
            write_coeffs(f"{mol1.name}_{mol2.name}", inter, appr, models, t_vals, out_dir / f"{mol1.name}_{mol2.name}.txt")

    for mol in molecules:
        for at in atoms:
            inter = Interaction(mol, at, interactions_yaml)
            write_coeffs(f"{mol.name}_{at.name}", inter, appr, models, t_vals, out_dir / f"{mol.name}_{at.name}.txt")

    for i, at1 in enumerate(atoms):
        for at2 in atoms[i:]:
            inter = Interaction(at1, at2, interactions_yaml)
            write_coeffs(f"{at1.name}_{at2.name}", inter, appr, models, t_vals, out_dir / f"{at1.name}_{at2.name}.txt")

    # quick sanity check
    inter = Interaction(n2, n, interactions_yaml)
    print("LC-test:", appr.omega_integral(1000.0, inter, 1, 1, ModelsOmega.VSS, False))


if __name__ == "__main__":
    main()
