"""Omega integral comparison across models (port of vssTest.cpp)."""
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


def main() -> None:
    data_dir = Path(os.environ.get("KAPPA_DATA_DIRECTORY", ROOT.parent / "data"))
    particles_yaml = data_dir / "particles.yaml"
    interactions_yaml = data_dir / "interaction.yaml"

    appr = Approximation()
    n2 = Molecule("N2", True, True, particles_yaml)
    n = Atom("N", particles_yaml)

    inter_n2_n2 = Interaction(n2, n2, interactions_yaml)
    inter_n2_n = Interaction(n2, n, interactions_yaml)

    for l, r in [(1, 1), (2, 3)]:
        for T in (2000.0, 10000.0):
            print(f"N2+N2 l={l} r={r} T={T}")
            print(
                "ESA:",
                appr.omega_integral(T, inter_n2_n2, l, r, ModelsOmega.ESA),
                appr.omega_integral(10000.0, inter_n2_n2, l, r, ModelsOmega.ESA) if T != 10000.0 else "",
            )
            print(
                "RS:",
                appr.omega_integral(T, inter_n2_n2, l, r, ModelsOmega.RS),
                appr.omega_integral(10000.0, inter_n2_n2, l, r, ModelsOmega.RS) if T != 10000.0 else "",
            )
            try:
                print(
                    "VSS:",
                    appr.omega_integral(T, inter_n2_n2, l, r, ModelsOmega.VSS),
                    appr.omega_integral(10000.0, inter_n2_n2, l, r, ModelsOmega.VSS) if T != 10000.0 else "",
                )
            except Exception as exc:
                print("VSS error:", exc)
            print()

            print(f"N2+N l={l} r={r} T={T}")
            print(
                "ESA:",
                appr.omega_integral(T, inter_n2_n, l, r, ModelsOmega.ESA),
                appr.omega_integral(10000.0, inter_n2_n, l, r, ModelsOmega.ESA) if T != 10000.0 else "",
            )
            print(
                "RS:",
                appr.omega_integral(T, inter_n2_n, l, r, ModelsOmega.RS),
                appr.omega_integral(10000.0, inter_n2_n, l, r, ModelsOmega.RS) if T != 10000.0 else "",
            )
            try:
                print(
                    "VSS:",
                    appr.omega_integral(T, inter_n2_n, l, r, ModelsOmega.VSS),
                    appr.omega_integral(10000.0, inter_n2_n, l, r, ModelsOmega.VSS) if T != 10000.0 else "",
                )
            except Exception as exc:
                print("VSS error:", exc)
            print()


if __name__ == "__main__":
    main()
