"""Dissociation rate coefficients (port of k_dissTest.cpp)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from kappa.approximations import Approximation  # noqa: E402
from kappa.interactions import Interaction  # noqa: E402
from kappa.models import ModelsKDiss  # noqa: E402
from kappa.particles import Atom, Molecule  # noqa: E402


def calculate(molecule: Molecule, atom: Atom, interactions_yaml: Path, out_dir: Path) -> None:
    inter = Interaction(molecule, atom, interactions_yaml)
    appr = Approximation()

    t_vals = [500.0, 1000.0, 2000.0, 5000.0, 10000.0, 15000.0, 20000.0, 25000.0, 30000.0, 40000.0]
    i_vals = [0, 5, 10, 20] if molecule.name == "O2" else [0, 5, 10, 20, 30]

    models = [
        ModelsKDiss.RS_THRESH_CMASS_VIBR,
        ModelsKDiss.RS_THRESH_VIBR,
        ModelsKDiss.RS_THRESH_CMASS_VIBR,
        ModelsKDiss.RS_THRESH_CMASS,
        ModelsKDiss.RS_THRESH,
        ModelsKDiss.VSS_THRESH_CMASS_VIBR,
        ModelsKDiss.VSS_THRESH_VIBR,
        ModelsKDiss.VSS_THRESH_CMASS,
        ModelsKDiss.VSS_THRESH,
        ModelsKDiss.ARRH_SCANLON,
        ModelsKDiss.ARRH_PARK,
        ModelsKDiss.TM_D6K_ARRH_SCANLON,
        ModelsKDiss.TM_3T_ARRH_SCANLON,
        ModelsKDiss.TM_INF_ARRH_SCANLON,
        ModelsKDiss.TM_D6K_ARRH_PARK,
        ModelsKDiss.TM_3T_ARRH_PARK,
        ModelsKDiss.TM_INF_ARRH_PARK,
        ModelsKDiss.PHYS4ENTRY,
        ModelsKDiss.ILT,
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{molecule.name}_{atom.name}.txt"
    with out_file.open("w") as fh:
        fh.write(f"{'T [K]':>15s}{'Vibr. l.':>20s}{'k_diss':>20s}\n")
        for T in t_vals:
            for i in i_vals:
                fh.write(f"{T:15.0f}{i:20d}")
                for model in models:
                    val = appr.k_diss(T, molecule, inter, i, 0, model)
                    fh.write(f"{val:20.8e}")
                fh.write("\n")
    print(f"Wrote k_diss to {out_file}")


def main() -> None:
    data_dir = Path(os.environ.get("KAPPA_DATA_DIRECTORY", ROOT.parent / "data"))
    particles_yaml = data_dir / "particles.yaml"
    interactions_yaml = data_dir / "interaction.yaml"

    n2 = Molecule("N2", False, True, particles_yaml)
    o2 = Molecule("O2", False, True, particles_yaml)
    n = Atom("N", particles_yaml)
    o = Atom("O", particles_yaml)

    out_dir = Path.cwd() / "DISS-REC"
    calculate(n2, n, interactions_yaml, out_dir)
    calculate(o2, o, interactions_yaml, out_dir)


if __name__ == "__main__":
    main()
