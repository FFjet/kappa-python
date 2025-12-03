"""FHO model checks for k_VT and VT probability (port of test_FHO.cpp)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from kappa.approximations import Approximation  # noqa: E402
from kappa.interactions import Interaction  # noqa: E402
from kappa.models import ModelsKVT  # noqa: E402
from kappa.particles import Atom, Molecule  # noqa: E402


def main() -> None:
    data_dir = Path(os.environ.get("KAPPA_DATA_DIRECTORY", ROOT.parent / "data"))
    particles_yaml = data_dir / "particles.yaml"
    interactions_yaml = data_dir / "interaction.yaml"

    mol = Molecule("N2", False, True, particles_yaml)
    at = Atom("N", particles_yaml)

    appr = Approximation()
    inter = Interaction(mol, at, interactions_yaml)

    print("probability_VT =", appr.probability_VT(20000.0, mol, inter, 1, -1, 0))

    print("k_VT FHO:", appr.k_VT(4000.0, mol, inter, 1, -1, 0, ModelsKVT.RS_FHO))
    print("k_VT SSH:", appr.k_VT(4000.0, mol, inter, 1, -1, 0, ModelsKVT.SSH))
    print("k_VT FHO:", appr.k_VT(4000.0, mol, inter, 2, -1, 0, ModelsKVT.RS_FHO))
    print("k_VT SSH:", appr.k_VT(4000.0, mol, inter, 2, -1, 0, ModelsKVT.SSH))
    print("k_VT FHO:", appr.k_VT(4000.0, mol, inter, 10, -1, 0, ModelsKVT.RS_FHO))
    print("k_VT SSH:", appr.k_VT(4000.0, mol, inter, 10, -1, 0, ModelsKVT.SSH))

    T = 12000.0
    T1 = 10000.0
    n_total = 1e24
    distrib = appr.Boltzmann_distribution(T1, n_total, mol, 0)

    res_kf = 0.0
    res_kb = 0.0
    for i in range(mol.num_vibr_levels[0] - 1):
        res_kf += (i + 1) * appr.k_VT(T, mol, inter, i + 1, -1, 0, ModelsKVT.RS_FHO) * distrib[i + 1] * n_total
        res_kb += (i + 1) * appr.k_VT(T, mol, inter, i + 1, -1, 0, ModelsKVT.RS_FHO) * appr.k_bf_VT(T, mol, i + 1, -1, 0) * distrib[i] * n_total
    for i in range(1, mol.num_vibr_levels[0]):
        res_kf += i * appr.k_VT(T, mol, inter, i, -1, 0, ModelsKVT.RS_FHO) * appr.k_bf_VT(T, mol, i, -1, 0) * distrib[i - 1] * n_total
        res_kb += i * appr.k_VT(T, mol, inter, i, -1, 0, ModelsKVT.RS_FHO) * distrib[i] * n_total

    print("R_VT =", res_kf - res_kb)


if __name__ == "__main__":
    main()
