"""Generate transport coefficient database for air5 (port of mk_TC_air5_database.cpp)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from kappa import constants  # noqa: E402
from kappa.mixtures import Mixture  # noqa: E402
from kappa.models import ModelsOmega  # noqa: E402
from kappa.particles import Atom, Molecule  # noqa: E402


def load_molar_fractions(path: Path, n_samples: int) -> np.ndarray:
    """Load the molar fraction samples produced by run_dirichlet.sh."""
    if not path.exists():
        raise FileNotFoundError(f"Missing molar fraction file: {path}")
    data = []
    with path.open() as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) == 5:
                data.append([float(val) for val in parts])
    arr = np.array(data)
    if arr.shape[0] < n_samples:
        raise ValueError(f"Expected at least {n_samples} samples in {path}, found {arr.shape[0]}")
    return arr[:n_samples, :]


def main() -> None:
    data_dir = Path(os.environ.get("KAPPA_DATA_DIRECTORY", ROOT.parent / "data"))
    particles_yaml = data_dir / "particles.yaml"
    interactions_yaml = data_dir / "interaction.yaml"
    output_dir = Path.cwd()

    print("KAPPA_DATA_DIRECTORY is:", data_dir)
    print("Current directory is:", output_dir)

    molecules: List[Molecule] = [
        Molecule("N2", True, False, particles_yaml),
        Molecule("O2", True, False, particles_yaml),
        Molecule("NO", True, False, particles_yaml),
    ]
    atoms: List[Atom] = [Atom("N", particles_yaml), Atom("O", particles_yaml)]

    mixture = Mixture(
        molecules=molecules,
        atoms=atoms,
        interactions_filename=interactions_yaml,
        particles_filename=particles_yaml,
        anharmonic=True,
        rigid_rotators=False,
    )

    n_samples = 10
    fractions_path = output_dir / "molar_fractions10.out"
    try:
        Xi = load_molar_fractions(fractions_path, n_samples)
    except Exception as exc:
        print(f"Cannot load molar fractions ({exc}); run run_dirichlet.sh first.")
        return

    T_vals = [i * 500.0 for i in range(1, 10)]  # 500–4500 K
    P_vals = [i * 1000.0 for i in range(1, 11)]  # 1000–10000 Pa

    out_file = output_dir / "TCs_air5_MD.txt"
    with out_file.open("w") as fh:
        for T in T_vals:
            for P in P_vals:
                tot_ndens = P / (constants.K_CONST_K * T)
                for sample in Xi:
                    x_N2, x_O2, x_NO, x_N, x_O = sample
                    mol_ndens = [
                        mixture.Boltzmann_distribution(T, (1.0 - x_N2) * tot_ndens, molecules[0], 0),
                        mixture.Boltzmann_distribution(T, (1.0 - x_O2) * tot_ndens, molecules[1], 0),
                        mixture.Boltzmann_distribution(T, (1.0 - x_NO) * tot_ndens, molecules[2], 0),
                    ]
                    atom_ndens = np.array([x_N * tot_ndens, x_O * tot_ndens])

                    mixture.compute_transport_coefficients(
                        T,
                        n_vl_molecule=mol_ndens,
                        n_atom=atom_ndens,
                        model=ModelsOmega.RS,
                        perturbation=0.0,
                    )
                    diff = mixture.get_diffusion()

                    fh.write(f"{T:15.6e}{P:15.6e}")
                    for vec in mol_ndens:
                        for val in vec:
                            fh.write(f"{val:15.6e}")
                    fh.write(f"{atom_ndens[0]:15.6e}{atom_ndens[1]:15.6e}")
                    for val in diff.flatten(order="C"):
                        fh.write(f"{val:15.6e}")
                    fh.write("\n")

    print(f"Wrote transport database to {out_file}")


if __name__ == "__main__":
    main()
