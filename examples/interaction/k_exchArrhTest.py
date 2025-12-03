"""Arrhenius exchange rates (port of k_exchArrhTest.cpp)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from kappa.approximations import Approximation  # noqa: E402
from kappa.interactions import Interaction  # noqa: E402
from kappa.models import ModelsKExch  # noqa: E402
from kappa.particles import Atom, Molecule  # noqa: E402


def main() -> None:
    data_dir = Path(os.environ.get("KAPPA_DATA_DIRECTORY", ROOT.parent / "data"))
    particles_yaml = data_dir / "particles.yaml"
    interactions_yaml = data_dir / "interaction.yaml"
    out_dir = Path.cwd() / "EXCH"
    out_dir.mkdir(parents=True, exist_ok=True)

    molecule = Molecule("N2", False, True, particles_yaml)
    atom = Atom("O", particles_yaml)
    inter = Interaction(molecule, atom, interactions_yaml)
    appr = Approximation()

    t_vals = [500.0, 1000.0, 2000.0, 5000.0, 10000.0, 15000.0, 20000.0, 25000.0, 30000.0, 40000.0]
    i_vals = [0, 10, 15, 20, 25]

    out_file = out_dir / f"{molecule.name}_{atom.name}.txt"
    with out_file.open("w") as fh:
        for T in t_vals:
            for i in i_vals:
                k_scanlon = appr.k_exch(T, molecule, atom, inter, i, 0, 1, ModelsKExch.ARRH_SCANLON)
                k_park = appr.k_exch(T, molecule, atom, inter, i, 0, 1, ModelsKExch.ARRH_PARK)
                fh.write(f"{T:15.0f}{i:20d}{k_scanlon:20.8e}{k_park:20.8e}\n")
    print(f"Wrote exchange rates to {out_file}")


if __name__ == "__main__":
    main()
