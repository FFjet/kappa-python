"""Omega integrals including charged/neutral/electron cases (port of vss_and_neutral_e.cpp)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from kappa.approximations import Approximation  # noqa: E402
from kappa.interactions import Interaction  # noqa: E402
from kappa.models import ModelsOmega  # noqa: E402
from kappa.particles import Atom, Molecule, Particle  # noqa: E402


def main() -> None:
    data_dir = Path(os.environ.get("KAPPA_DATA_DIRECTORY", ROOT.parent / "data"))
    particles_yaml = data_dir / "particles.yaml"
    interactions_yaml = data_dir / "interaction.yaml"

    appr = Approximation()
    n2 = Molecule("N2", True, True, particles_yaml)
    e = Particle("e-", particles_yaml)
    np = Atom("N+", particles_yaml)
    n = Atom("N", particles_yaml)

    inter_n2_np = Interaction(n2, np, interactions_yaml)
    inter_ee = Interaction(e, e, interactions_yaml)
    inter_ne = Interaction(n2, e, interactions_yaml)

    l_arr = [1, 1, 2, 3]
    r_arr = [1, 4, 2, 3]

    print("collision-reduced mass of two electrons:", inter_ee.collision_mass, "e- mass:", e.mass)
    print("Ne _Om22_0:", inter_ne.data.get("_Om22_0"))

    for l, r in zip(l_arr, r_arr):
        print(f"{n2.name} + {np.name}")
        for model in (ModelsOmega.ESA, ModelsOmega.RS, ModelsOmega.VSS):
            try:
                print(model.name, appr.omega_integral(2000.0, inter_n2_np, l, r, model), appr.omega_integral(10000.0, inter_n2_np, l, r, model))
            except Exception as exc:
                print(model.name, "error:", exc)
        print()

        print(f"{n.name} + {e.name}")
        for model in (ModelsOmega.ESA, ModelsOmega.RS):
            try:
                print(model.name, appr.omega_integral(2000.0, inter_ne, l, r, model), appr.omega_integral(10000.0, inter_ne, l, r, model))
            except Exception as exc:
                print(model.name, "error:", exc)
        print()


if __name__ == "__main__":
    main()
