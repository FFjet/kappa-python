"""Interaction demo covering omega integrals and dissociation rates."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from kappa.approximations import Approximation  # noqa: E402
from kappa.interactions import Interaction  # noqa: E402
from kappa.models import ModelsKDiss, ModelsOmega  # noqa: E402
from kappa.particles import Molecule  # noqa: E402


def main() -> None:
    data_dir = Path(os.environ.get("KAPPA_DATA_DIRECTORY", ROOT.parent / "data"))
    particles_yaml = data_dir / "particles.yaml"
    interactions_yaml = data_dir / "interaction.yaml"

    # N2 + N2 interaction demo
    n2_a = Molecule("N2", True, True, particles_yaml)
    n2_b = Molecule("N2", True, True, particles_yaml)
    interaction = Interaction(n2_a, n2_b, interactions_yaml)
    approx = Approximation()

    T = 10000.0
    omega_rs = approx.omega_integral(T, interaction, 1, 1, ModelsOmega.RS)
    omega_vss = approx.omega_integral(T, interaction, 1, 1, ModelsOmega.VSS) if interaction.vss_data else None
    print("Omega_11 RS:", omega_rs)
    if omega_vss is not None:
        print("Omega_11 VSS:", omega_vss)

    # Dissociation rate for ground vibrational level
    k_rs = approx.k_diss(T, n2_a, interaction, 0, 0, ModelsKDiss.RS_THRESH)
    print("k_diss RS (v=0, e=0):", k_rs)
    if interaction.vss_data:
        k_vss = approx.k_diss(T, n2_a, interaction, 0, 0, ModelsKDiss.VSS_THRESH)
        print("k_diss VSS (v=0, e=0):", k_vss)

    # Cross-check with a different temperature range
    temps = np.linspace(3000.0, 15000.0, num=5)
    omega_scan = [approx.omega_integral(Ti, interaction, 1, 2, ModelsOmega.RS) for Ti in temps]
    print("Omega_12 RS scan:", omega_scan)


if __name__ == "__main__":
    main()
