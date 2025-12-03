"""Dissociation cross sections (port of cs_dissTest.cpp)."""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from kappa import constants  # noqa: E402
from kappa.approximations import Approximation  # noqa: E402
from kappa.interactions import Interaction  # noqa: E402
from kappa.models import ModelsCsDiss  # noqa: E402
from kappa.particles import Atom, Molecule  # noqa: E402


def calculate(molecule: Molecule, atom: Atom, interactions_yaml: Path, output_dir: Path) -> None:
    inter = Interaction(molecule, atom, interactions_yaml)
    appr = Approximation()

    t_vals = [
        1000.0,
        2000.0,
        4000.0,
        5000.0,
        10000.0,
        15000.0,
        20000.0,
        25000.0,
        30000.0,
        45000.0,
        50000.0,
        55000.0,
        60000.0,
        65000.0,
        70000.0,
        75000.0,
        80000.0,
        85000.0,
        90000.0,
        95000.0,
        100000.0,
        105000.0,
        110000.0,
        115000.0,
        120000.0,
        125000.0,
        130000.0,
        135000.0,
        140000.0,
        145000.0,
        150000.0,
        155000.0,
        160000.0,
        165000.0,
        170000.0,
        175000.0,
        180000.0,
        185000.0,
        190000.0,
        195000.0,
        200000.0,
    ]
    i_vals = [0, 5, 10, 20] if molecule.name == "O2" else [0, 5, 10, 20, 30]
    cs_models = [
        ModelsCsDiss.RS_THRESH,
        ModelsCsDiss.RS_THRESH_CMASS,
        ModelsCsDiss.RS_THRESH_CMASS_VIBR,
        ModelsCsDiss.VSS_THRESH,
        ModelsCsDiss.VSS_THRESH_CMASS_VIBR,
        ModelsCsDiss.ILT,
    ]
    cs_names = [
        "RS",
        "RS_cmass",
        "RS_cmass_vibr",
        "VSS",
        "VSS_cmass_vibr",
        "ILT",
    ]

    out_file = output_dir / f"{molecule.name}_{atom.name}.txt"
    with out_file.open("w") as fh:
        fh.write("t;")
        for i in i_vals:
            energy_ev = molecule.vibr_energy[0][i] / constants.K_CONST_EV
            for name in cs_names:
                fh.write(f"{name},i={i}({energy_ev:.6g} eV);")
            fh.write(";")
        fh.write("\n")

        for T in t_vals:
            rel_vel = math.sqrt(2.0 * constants.K_CONST_K * T / inter.collision_mass)
            fh.write(f"{T};")
            for i in i_vals:
                for model in cs_models:
                    val = appr.crosssection_diss(rel_vel, molecule, inter, i, 0, model)
                    fh.write(f"{val:.6g};")
                fh.write(";")
            fh.write("\n")

    print(f"Wrote cross sections to {out_file}")


def main() -> None:
    data_dir = Path(os.environ.get("KAPPA_DATA_DIRECTORY", ROOT.parent / "data"))
    particles_yaml = data_dir / "particles.yaml"
    output_dir = Path.cwd()

    print("KAPPA_DATA_DIRECTORY is:", data_dir)
    print("Output directory:", output_dir)

    n2 = Molecule("N2", True, True, particles_yaml)
    o2 = Molecule("O2", True, True, particles_yaml)
    n = Atom("N", particles_yaml)
    o = Atom("O", particles_yaml)

    interactions_yaml = data_dir / "interaction.yaml"
    output_dir = Path.cwd()

    calculate(n2, n, interactions_yaml, output_dir)
    calculate(o2, o, interactions_yaml, output_dir)


if __name__ == "__main__":
    main()
