"""ILT vs STS dissociation rates (port of k_diss_ILTtest.cpp)."""
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
from kappa.models import ModelsKDiss  # noqa: E402
from kappa.particles import Atom, Molecule  # noqa: E402


def calculate(molecule: Molecule, atom: Atom, interactions_yaml: Path, out_dir: Path) -> None:
    inter = Interaction(molecule, atom, interactions_yaml)
    appr = Approximation()

    t_vals = [500.0, 1000.0, 2000.0, 5000.0, 10000.0, 15000.0, 20000.0, 25000.0, 30000.0, 40000.0]
    i_vals = [0, 5, 10, 20] if molecule.name == "O2" else [0, 5, 10, 20, 30]

    out_file = out_dir / f"{molecule.name}_{atom.name}_k_ILT_test.txt"
    with out_file.open("w") as fh:
        fh.write(f"{'T [K]':>20s}{'vibr. level':>20s}{'vibr. energy (eV)':>20s}{'ILT':>20s}{'STS':>20s}\n")
        for T in t_vals:
            for i in i_vals:
                ilt = appr.k_diss(T, molecule, inter, i, 0, ModelsKDiss.ILT)
                if molecule.name == "O2":
                    c1 = 0.3867 * i**3 - 2.7425 * i * i - 1901.9 * i + 61696.0
                    if i <= 31:
                        c2 = 1.63e-9 * i**3 - 1.25e-7 * i * i + 3.24e-6 * i + 7.09e-5
                    elif i <= 37:
                        c2 = -6.67e-6 * i * i + 4.65e-4 * i - 7.91e-3
                    else:
                        c2 = 7.83e-7 * i**4 - 1.31e-4 * i**3 + 8.24e-3 * i * i - 0.23 * i + 2.4049
                    sts = (T ** -0.1) * 1.53e-10 * c2 * math.exp(-c1 / T)
                else:
                    if i <= 8:
                        c1 = 1.786e-18
                    elif i <= 34:
                        c1 = 1.71e-18
                    elif i <= 52:
                        c1 = 1.68e-18
                    else:
                        c1 = 1.66e-18
                    c2 = 4e-19 * i**4 + 5.24e-19 * i**3 - 7.41e-17 * i * i + 6.42e-15 * i + 7.3e-14
                    sts = 7.16e-2 * (T ** -0.25) * c2 * math.exp((molecule.vibr_energy[0][i] - c1) / (constants.K_CONST_K * T))

                fh.write(
                    f"{T:20.0f}"
                    f"{i:20d}"
                    f"{molecule.vibr_energy[0][i] / constants.K_CONST_EV:20.6f}"
                    f"{ilt:20.8e}"
                    f"{sts:20.8e}\n"
                )
    print(f"Wrote ILT comparison to {out_file}")


def main() -> None:
    data_dir = Path(os.environ.get("KAPPA_DATA_DIRECTORY", ROOT.parent / "data"))
    particles_yaml = data_dir / "particles.yaml"
    interactions_yaml = data_dir / "interaction.yaml"
    out_dir = Path.cwd()

    n2 = Molecule("N2", True, True, particles_yaml)
    o2 = Molecule("O2", True, True, particles_yaml)
    n = Atom("N", particles_yaml)
    o = Atom("O", particles_yaml)

    calculate(n2, n, interactions_yaml, out_dir)
    calculate(o2, o, interactions_yaml, out_dir)


if __name__ == "__main__":
    main()
