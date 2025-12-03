from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Optional

import numpy as np
from scipy import linalg

from ..approximations import Approximation
from ..exceptions import DataNotFoundException, IncorrectValueException
from .. import constants
from ..interactions import Interaction
from ..models import ModelsOmega
from ..particles import Atom, Molecule, Particle


@dataclass
class Mixture(Approximation):
    """Python port of `kappa::Mixture` (work in progress)."""

    molecules: List[Molecule] = field(default_factory=list)
    atoms: List[Atom] = field(default_factory=list)
    is_ionized: bool = False
    cache_on: bool = False

    def __init__(
        self,
        molecules: Sequence[Molecule] | None = None,
        atoms: Sequence[Atom] | None = None,
        *,
        particle_names: str | None = None,
        interactions_filename: str = "data/interaction.yaml",
        particles_filename: str = "data/particles.yaml",
        anharmonic: bool = True,
        rigid_rotators: bool = True,
    ) -> None:
        super().__init__()
        self.molecules = list(molecules or [])
        self.atoms = list(atoms or [])
        self.is_ionized = False
        self.cache_on = False
        self.molecule_name_map: Dict[str, int] = {}
        self.atom_name_map: Dict[str, int] = {}
        if particle_names:
            self._load_from_names(
                particle_names,
                particles_filename=particles_filename,
                anharmonic=anharmonic,
                rigid_rotators=rigid_rotators,
            )
        self.num_molecules = len(self.molecules)
        self.num_atoms = len(self.atoms)
        self.n_vibr_levels_total = sum(m.num_vibr_levels[0] if m.num_vibr_levels else 0 for m in self.molecules)
        self.n_particles = self.num_molecules + self.num_atoms
        self.vl_offset: List[int] = []
        self.interactions: List[Interaction] = []
        for idx, molecule in enumerate(self.molecules):
            self.molecule_name_map[molecule.name] = idx
            if molecule.charge != 0:
                self.is_ionized = True
        for idx, atom in enumerate(self.atoms):
            self.atom_name_map[atom.name] = idx
            if atom.charge != 0:
                self.is_ionized = True
        self.all_rigid_rotators = all(m.rigid_rotator for m in self.molecules) if self.molecules else True
        self._init_state_caches()
        self.init_matrices(particles_filename)
        self.add_interactions(interactions_filename)

    def _init_state_caches(self) -> None:
        self.molecule_charges = np.zeros(max(self.num_molecules, 1))
        self.atom_charges = np.zeros(max(self.num_atoms, 1))
        self.molecule_charges_sq = np.zeros_like(self.molecule_charges)
        self.atom_charges_sq = np.zeros_like(self.atom_charges)
        self.empty_n_atom = np.zeros(1)
        self.empty_n_vl_molecule = [np.zeros(1)]
        self.this_n_molecules = np.zeros(self.num_molecules)
        self.this_n_atom = np.zeros(self.num_atoms)
        self.this_n_vl_mol: List[np.ndarray] = [np.zeros(m.num_vibr_levels[0]) for m in self.molecules]
        self.this_total_n = 0.0
        self.this_total_dens = 0.0
        self.this_ctr = 0.0
        self.this_crot = 0.0
        self.this_n_electrons = 0.0
        self.electron: Particle | None = None
        self.th_cond = 0.0
        self.sh_visc = 0.0
        self.b_visc = 0.0

    def _load_from_names(
        self,
        names: str,
        *,
        particles_filename: str,
        anharmonic: bool,
        rigid_rotators: bool,
    ) -> None:
        for token in self.split_string(names):
            if token == "e-":
                self.is_ionized = True
                continue
            try:
                molecule = Molecule(token, anharmonic_spectrum=anharmonic, rigid_rotator=rigid_rotators, filename=particles_filename)
                self.molecules.append(molecule)
            except DataNotFoundException:
                atom = Atom(token, filename=particles_filename)
                self.atoms.append(atom)

    def init_matrices(self, particles_filename: str) -> None:
        self.empty_n_atom = np.zeros(1)
        self.empty_n_vl_molecule = [np.zeros(1)]
        self.this_n_molecules = np.zeros(self.num_molecules)
        self.vl_offset = []

        if self.num_molecules:
            self.vl_offset.append(0)
            for molecule in self.molecules:
                self.vl_offset.append(self.vl_offset[-1] + molecule.num_vibr_levels[0])
            self.molecule_charges = np.zeros(self.num_molecules)
            for idx, molecule in enumerate(self.molecules):
                self.molecule_charges[idx] = molecule.charge * constants.K_CONST_ELEMENTARY_CHARGE
                if molecule.charge != 0:
                    self.is_ionized = True
            self.molecule_charges_sq = self.molecule_charges * self.molecule_charges
        else:
            self.molecule_charges = np.zeros(1)
            self.molecule_charges_sq = np.zeros(1)

        if self.num_atoms:
            self.atom_charges = np.zeros(self.num_atoms)
            for idx, atom in enumerate(self.atoms):
                self.atom_charges[idx] = atom.charge * constants.K_CONST_ELEMENTARY_CHARGE
                if atom.charge != 0:
                    self.is_ionized = True
            self.atom_charges_sq = self.atom_charges * self.atom_charges
        else:
            self.atom_charges = np.zeros(1)
            self.atom_charges_sq = np.zeros(1)

        if self.is_ionized and self.electron is None:
            self.electron = Particle("e-", filename=particles_filename)
            self.n_particles += 1

        self.th_diff = np.zeros(self.n_vibr_levels_total + self.num_atoms + (1 if self.is_ionized else 0))

        self.shear_viscosity_RHS = np.zeros(self.n_particles)
        self.shear_viscosity_LHS = np.zeros((self.n_particles, self.n_particles))
        self.shear_viscosity_coeffs = np.zeros(self.n_particles)

        self.thermal_conductivity_RHS = np.zeros(3 * self.n_vibr_levels_total + 2 * self.num_atoms)
        self.thermal_conductivity_LHS = np.zeros((3 * self.n_vibr_levels_total + 2 * self.num_atoms, 3 * self.n_vibr_levels_total + 2 * self.num_atoms))
        self.thermal_conductivity_coeffs = np.zeros(3 * self.n_vibr_levels_total + 2 * self.num_atoms)

        self.thermal_conductivity_rigid_rot_RHS = np.zeros(3 * self.num_molecules + 2 * self.num_atoms)
        self.thermal_conductivity_rigid_rot_LHS = np.zeros((3 * self.num_molecules + 2 * self.num_atoms, 3 * self.num_molecules + 2 * self.num_atoms))
        self.thermal_conductivity_rigid_rot_coeffs = np.zeros(3 * self.num_molecules + 2 * self.num_atoms)

        self.bulk_viscosity_rigid_rot_RHS = np.zeros(2 * self.num_molecules + self.num_atoms)
        self.bulk_viscosity_rigid_rot_LHS = np.zeros((2 * self.num_molecules + self.num_atoms, 2 * self.num_molecules + self.num_atoms))
        self.bulk_viscosity_rigid_rot_coeffs = np.zeros(2 * self.num_molecules + self.num_atoms)

        self.bulk_viscosity_RHS = np.zeros(2 * self.n_vibr_levels_total + self.num_atoms)
        self.bulk_viscosity_LHS = np.zeros((2 * self.n_vibr_levels_total + self.num_atoms, 2 * self.n_vibr_levels_total + self.num_atoms))
        self.bulk_viscosity_coeffs = np.zeros(2 * self.n_vibr_levels_total + self.num_atoms)

        self.diffusion_LHS = np.zeros((self.n_vibr_levels_total + self.num_atoms, self.n_vibr_levels_total + self.num_atoms))
        self.diffusion_RHS = np.zeros(self.n_vibr_levels_total + self.num_atoms)
        self.diffusion_coeffs = np.zeros(self.n_vibr_levels_total + self.num_atoms)
        self.diff = np.zeros((self.n_vibr_levels_total + self.num_atoms, self.n_vibr_levels_total + self.num_atoms))
        self.lite_diff = np.zeros_like(self.diff)

        self.omega_11 = np.zeros((self.n_particles, self.n_particles))
        self.omega_12 = np.zeros((self.n_particles, self.n_particles))
        self.omega_13 = np.zeros((self.n_particles, self.n_particles))
        self.omega_22 = np.zeros((self.n_particles, self.n_particles))

        self.c_rot_arr = np.zeros(self.n_vibr_levels_total)
        self.c_rot_rigid_rot_arr = np.zeros(self.num_molecules)

        self.rot_rel_times = np.zeros((self.num_molecules, self.num_molecules + self.num_atoms))

        self.binary_diff = np.zeros(self.n_vibr_levels_total + 3)
        self.this_total_n = 0.0
        self.this_n_electrons = 0.0

    def add_interactions(self, filename: str) -> None:
        self.interactions.clear()
        for i in range(self.num_molecules):
            for j in range(i, self.num_molecules):
                self.interactions.append(Interaction(self.molecules[i], self.molecules[j], filename))
            for j in range(self.num_atoms):
                self.interactions.append(Interaction(self.molecules[i], self.atoms[j], filename))
            if self.is_ionized and self.electron is not None:
                self.interactions.append(Interaction(self.molecules[i], self.electron, filename))
        for i in range(self.num_atoms):
            for j in range(i, self.num_atoms):
                self.interactions.append(Interaction(self.atoms[i], self.atoms[j], filename))
            if self.is_ionized and self.electron is not None:
                self.interactions.append(Interaction(self.atoms[i], self.electron, filename))
        if self.is_ionized and self.electron is not None:
            self.interactions.append(Interaction(self.electron, self.electron, filename))

    def split_string(self, input_string: str) -> List[str]:
        holder = ""
        result: List[str] = []
        for char in input_string:
            if char == "," and holder:
                result.append(holder)
                holder = ""
            elif char != " ":
                holder += char
        if holder:
            result.append(holder)
        return result

    def inter_index(self, i: int, j: int) -> int:
        if i > j:
            return self.inter_index(j, i)
        return self.n_particles * i + j - i * (i + 1) // 2

    def get_names(self) -> str:
        pieces = [m.name for m in self.molecules] + [a.name for a in self.atoms]
        if self.is_ionized:
            pieces.append("e-")
        return " ".join(pieces).strip()

    def get_n_particles(self) -> int:
        return self.n_particles

    def get_n_vibr_levels(self) -> int:
        return self.n_vibr_levels_total

    def get_n_vibr_levels_array(self) -> List[int]:
        return [m.num_vibr_levels[0] if m.num_vibr_levels else 0 for m in self.molecules]

    def convert_molar_frac_to_mass(self, x: np.ndarray) -> np.ndarray:
        masses: List[float] = [m.mass for m in self.molecules] + [a.mass for a in self.atoms]
        if self.is_ionized and self.electron is not None:
            masses.append(self.electron.mass)
        masses_vec = np.asarray(masses)
        rho = np.dot(x, masses_vec)
        if rho == 0:
            raise IncorrectValueException("Total mass fraction is zero")
        return x * masses_vec / rho

    def convert_mass_frac_to_molar(self, y: np.ndarray) -> np.ndarray:
        inv_masses: List[float] = [1.0 / m.mass for m in self.molecules] + [1.0 / a.mass for a in self.atoms]
        if self.is_ionized and self.electron is not None:
            inv_masses.append(1.0 / self.electron.mass)
        inv_masses_vec = np.asarray(inv_masses)
        denom = np.dot(y, inv_masses_vec)
        if denom == 0:
            raise IncorrectValueException("Total molar fraction is zero")
        return y * inv_masses_vec / denom

    def molecule(self, name: str) -> Molecule:
        return self.molecules[self.molecule_name_map[name]]

    def atom(self, name: str) -> Atom:
        return self.atoms[self.atom_name_map[name]]

    def interaction(self, particle1: Molecule | Atom, particle2: Molecule | Atom) -> Interaction:
        if isinstance(particle1, Molecule) and isinstance(particle2, Molecule):
            return self.interactions[self.inter_index(self.molecule_name_map[particle1.name], self.molecule_name_map[particle2.name])]
        if isinstance(particle1, Molecule) and isinstance(particle2, Atom):
            return self.interactions[self.inter_index(self.molecule_name_map[particle1.name], self.num_molecules + self.atom_name_map[particle2.name])]
        if isinstance(particle1, Atom) and isinstance(particle2, Molecule):
            return self.interactions[self.inter_index(self.num_molecules + self.atom_name_map[particle1.name], self.molecule_name_map[particle2.name])]
        return self.interactions[self.inter_index(self.num_molecules + self.atom_name_map[particle1.name], self.num_molecules + self.atom_name_map[particle2.name])]

    def debye_length(self, temperature: float, n_molecule: np.ndarray, n_atom: np.ndarray, n_electrons: float) -> float:
        numerator = constants.K_CONST_E0 * constants.K_CONST_K * temperature
        denom = float(np.dot(n_molecule, self.molecule_charges_sq) + np.dot(n_atom, self.atom_charges_sq))
        denom += n_electrons * constants.K_CONST_ELEMENTARY_CHARGE ** 2
        if denom <= 0:
            raise IncorrectValueException("Debye length denominator is non-positive")
        return float(np.sqrt(numerator / denom))

    def debye_length_from_v_levels(
        self,
        temperature: float,
        n_vl_molecule: Sequence[np.ndarray],
        n_atom: np.ndarray,
        n_electrons: float,
    ) -> float:
        n_mol = self.compute_n_molecule(n_vl_molecule)
        return self.debye_length(temperature, n_mol, n_atom, n_electrons)

    def debye_length_vibrational(self, temperature: float, n_vl_molecule: Sequence[np.ndarray], n_electrons: float) -> float:
        n_mol = self.compute_n_molecule(n_vl_molecule)
        zero_atoms = np.zeros(self.num_atoms)
        return self.debye_length(temperature, n_mol, zero_atoms, n_electrons)

    def debye_length_from_vector(self, temperature: float, n: np.ndarray) -> float:
        if n.size != self.n_particles:
            raise IncorrectValueException("Incorrect size of number density vector")
        n_mol = n[: self.num_molecules] if self.num_molecules else np.zeros(0)
        start = self.num_molecules
        n_atom = n[start : start + self.num_atoms] if self.num_atoms else np.zeros(0)
        n_e = float(n[start + self.num_atoms]) if self.is_ionized else 0.0
        return self.debye_length(temperature, n_mol, n_atom, n_e)

    def compute_n(self, n_vl_molecule: Sequence[np.ndarray], n_atom: np.ndarray, n_electrons: float = 0.0) -> float:
        total = sum(float(np.sum(vec)) for vec in n_vl_molecule)
        total += float(np.sum(n_atom))
        total += n_electrons
        return total

    def compute_n_from_molecules(self, n_molecule: np.ndarray, n_atom: np.ndarray, n_electrons: float = 0.0) -> float:
        return float(np.sum(n_molecule) + np.sum(n_atom) + n_electrons)

    def compute_n_vibrational(self, n_vl_molecule: Sequence[np.ndarray], n_electrons: float = 0.0) -> float:
        return self.compute_n(n_vl_molecule, np.zeros(self.num_atoms), n_electrons)

    def compute_n_from_vector(self, n: np.ndarray) -> float:
        return float(np.sum(n))

    def compute_n_molecule(self, n_vl_molecule: Sequence[np.ndarray]) -> np.ndarray:
        return np.asarray([float(np.sum(vec)) for vec in n_vl_molecule])

    def compute_density(self, n_vl_molecule: Sequence[np.ndarray], n_atom: np.ndarray, n_electrons: float = 0.0) -> float:
        density = 0.0
        for molecule, populations in zip(self.molecules, n_vl_molecule):
            density += molecule.mass * float(np.sum(populations))
        for atom, value in zip(self.atoms, n_atom):
            density += atom.mass * float(value)
        if self.is_ionized and self.electron is not None:
            density += self.electron.mass * n_electrons
        return density

    def _compute_density_molecules_only(self, n_vl_molecule: Sequence[np.ndarray]) -> float:
        density = 0.0
        for molecule, populations in zip(self.molecules, n_vl_molecule):
            density += molecule.mass * float(np.sum(populations))
        return density

    def compute_density_vibrational(self, n_vl_molecule: Sequence[np.ndarray], n_electrons: float = 0.0) -> float:
        return self.compute_density(n_vl_molecule, np.zeros(self.num_atoms), n_electrons)

    def compute_density_from_vectors(self, n_molecule: np.ndarray, n_atom: np.ndarray, n_electrons: float = 0.0) -> float:
        density = 0.0
        for molecule, value in zip(self.molecules, n_molecule):
            density += molecule.mass * float(value)
        for atom, value in zip(self.atoms, n_atom):
            density += atom.mass * float(value)
        if self.is_ionized and self.electron is not None:
            density += self.electron.mass * n_electrons
        return density

    def compute_density_from_state_vector(self, n: np.ndarray) -> float:
        if n.size != self.n_particles:
            raise IncorrectValueException("Incorrect size of number density vector")
        idx = 0
        density = 0.0
        for molecule in self.molecules:
            density += molecule.mass * float(n[idx])
            idx += 1
        for atom in self.atoms:
            density += atom.mass * float(n[idx])
            idx += 1
        if self.is_ionized and self.electron is not None:
            density += self.electron.mass * float(n[idx])
        return density

    def compute_density_array(
        self,
        n_vl_molecule: Sequence[np.ndarray],
        n_atom: np.ndarray | None = None,
        n_electrons: float = 0.0,
    ) -> np.ndarray:
        total = self.num_molecules + self.num_atoms + (1 if self.is_ionized else 0)
        res = np.zeros(total)
        idx = 0
        for molecule, populations in zip(self.molecules, n_vl_molecule):
            res[idx] = molecule.mass * float(np.sum(populations))
            idx += 1
        if n_atom is None:
            n_atom = np.zeros(self.num_atoms)
        for atom, value in zip(self.atoms, n_atom):
            res[idx] = atom.mass * float(value)
            idx += 1
        if self.is_ionized and self.electron is not None:
            res[idx] = self.electron.mass * n_electrons
        return res

    def compute_pressure(self, temperature: float, n_vl_molecule: Sequence[np.ndarray], n_atom: np.ndarray, n_electrons: float = 0.0) -> float:
        total_n = self.compute_n(n_vl_molecule, n_atom, n_electrons)
        return total_n * constants.K_CONST_K * temperature

    def c_tr(self, n_vl_molecule: Sequence[np.ndarray], n_atom: np.ndarray, n_electrons: float = 0.0) -> float:
        density = self.compute_density(n_vl_molecule, n_atom, n_electrons)
        if density == 0:
            raise IncorrectValueException("Mixture density is zero")
        total_n = self.compute_n(n_vl_molecule, n_atom, n_electrons)
        return 1.5 * constants.K_CONST_K * total_n / density

    def c_tr_vibrational(self, n_vl_molecule: Sequence[np.ndarray], n_electrons: float = 0.0) -> float:
        return self.c_tr(n_vl_molecule, np.zeros(self.num_atoms), n_electrons)

    def c_tr_from_vectors(self, n_molecule: np.ndarray, n_atom: np.ndarray, n_electrons: float = 0.0) -> float:
        density = self.compute_density_from_vectors(n_molecule, n_atom, n_electrons)
        if density == 0:
            raise IncorrectValueException("Mixture density is zero")
        total_n = self.compute_n_from_molecules(n_molecule, n_atom, n_electrons)
        return 1.5 * constants.K_CONST_K * total_n / density

    def c_tr_from_state_vector(self, n: np.ndarray) -> float:
        density = self.compute_density_from_state_vector(n)
        if density == 0:
            raise IncorrectValueException("Mixture density is zero")
        return 1.5 * constants.K_CONST_K * float(np.sum(n)) / density

    def c_rot(self, temperature: float, first, second=None, n_electrons: float = 0.0):
        if isinstance(first, Molecule):
            if second is None:
                vibr_level = 0
                e_level = 0
            elif isinstance(second, (tuple, list)):
                if len(second) == 2:
                    vibr_level, e_level = map(int, second)
                else:
                    vibr_level = int(second[0])
                    e_level = 0
            else:
                vibr_level = int(second)
                e_level = 0
            return super().c_rot(temperature, first, vibr_level, e_level)
        if isinstance(first, (list, tuple)) and isinstance(second, np.ndarray):
            return self._c_rot_from_populations(temperature, first, second, n_electrons)
        if isinstance(first, (list, tuple)) and second is None:
            return self._c_rot_vibrational(temperature, first, n_electrons)
        if isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
            return self._c_rot_from_number_density(temperature, first, second, n_electrons)
        if isinstance(first, np.ndarray) and second is None:
            return self._c_rot_from_state_vector(temperature, first)
        raise IncorrectValueException("Unsupported c_rot signature")

    def _c_rot_from_populations(
        self,
        temperature: float,
        n_vl_molecule: Sequence[np.ndarray],
        n_atom: np.ndarray,
        n_electrons: float,
    ) -> float:
        density = self.compute_density(n_vl_molecule, n_atom, n_electrons)
        if density == 0:
            raise IncorrectValueException("Mixture density is zero")
        res = 0.0
        for molecule, populations in zip(self.molecules, n_vl_molecule):
            for level_idx, population in enumerate(populations):
                res += super().c_rot(temperature, molecule, level_idx, 0) * float(population) * molecule.mass
        return res / density

    def _c_rot_vibrational(self, temperature: float, n_vl_molecule: Sequence[np.ndarray], n_electrons: float) -> float:
        return self._c_rot_from_populations(temperature, n_vl_molecule, np.zeros(self.num_atoms), n_electrons)

    def _c_rot_from_number_density(
        self,
        temperature: float,
        n_molecule: np.ndarray,
        n_atom: np.ndarray,
        n_electrons: float,
    ) -> float:
        if not self.all_rigid_rotators:
            raise IncorrectValueException(
                "Mixture contains non-rigid rotators; vibrational level populations are required",
            )
        density = self.compute_density_from_vectors(n_molecule, n_atom, n_electrons)
        if density == 0:
            raise IncorrectValueException("Mixture density is zero")
        res = 0.0
        for molecule, value in zip(self.molecules, n_molecule):
            res += super().c_rot(temperature, molecule, 0, 0) * float(value) * molecule.mass
        return res / density

    def _c_rot_from_state_vector(self, temperature: float, n: np.ndarray) -> float:
        if not self.all_rigid_rotators:
            raise IncorrectValueException(
                "Mixture contains non-rigid rotators; vibrational level populations are required",
            )
        if self.num_molecules == 0:
            return 0.0
        density = self.compute_density_from_state_vector(n)
        if density == 0:
            raise IncorrectValueException("Mixture density is zero")
        res = 0.0
        for idx, molecule in enumerate(self.molecules):
            res += super().c_rot(temperature, molecule, 0, 0) * float(n[idx]) * molecule.mass
        return res / density

    # --- Transport helpers mirroring the C++ implementation --- #
    def check_n_vl_molecule(self, n_vl_molecule: Sequence[np.ndarray]) -> None:
        if len(n_vl_molecule) != self.num_molecules:
            raise IncorrectValueException("Incorrect number of molecular populations")
        for idx, (vec, mol) in enumerate(zip(n_vl_molecule, self.molecules)):
            if vec.shape[0] != mol.num_vibr_levels[0]:
                raise IncorrectValueException(f"Incorrect size of vibrational populations for molecule #{idx}")
            if np.any(vec < 0):
                raise IncorrectValueException("Negative vibrational population encountered")

    def check_n_molecule(self, n_molecule: np.ndarray) -> None:
        if n_molecule.size != self.num_molecules or np.any(n_molecule < 0):
            raise IncorrectValueException("Incorrect molecular number density vector")

    def check_n_atom(self, n_atom: np.ndarray) -> None:
        if n_atom.size != self.num_atoms or np.any(n_atom < 0):
            raise IncorrectValueException("Incorrect atomic number density vector")

    def check_n(self, n: np.ndarray) -> None:
        if n.size != self.n_particles or np.any(n < 0):
            raise IncorrectValueException("Incorrect state vector size or negative entries")

    def compute_c_rot_rigid_rot(self, temperature: float) -> None:
        for idx, molecule in enumerate(self.molecules):
            self.c_rot_rigid_rot_arr[idx] = super().c_rot(temperature, molecule, 0, 0)

    def compute_c_rot(self, temperature: float) -> None:
        offset = 0
        for molecule in self.molecules:
            for k in range(molecule.num_vibr_levels[0]):
                self.c_rot_arr[offset] = super().c_rot(temperature, molecule, k, 0)
                offset += 1

    def compute_full_crot_rigid_rot(self, temperature: float) -> None:
        res = 0.0
        for idx, molecule in enumerate(self.molecules):
            res += self.c_rot_rigid_rot_arr[idx] * self.this_n_molecules[idx] * molecule.mass
        self.this_crot = res / self.this_total_dens if self.this_total_dens else 0.0

    def compute_full_crot(self, temperature: float) -> None:
        res = 0.0
        offset = 0
        for mol_idx, molecule in enumerate(self.molecules):
            for k in range(molecule.num_vibr_levels[0]):
                res += self.c_rot_arr[offset] * self.this_n_vl_mol[mol_idx][k] * molecule.mass
                offset += 1
        self.this_crot = res / self.this_total_dens if self.this_total_dens else 0.0

    def compute_rot_rel_times(self, temperature: float, number_density: float, model: ModelsOmega) -> None:
        for i, molecule in enumerate(self.molecules):
            for j in range(self.num_molecules):
                self.rot_rel_times[i, j] = self.rot_relaxation_time_parker(temperature, number_density, molecule, self.interactions[self.inter_index(i, j)], model)
            for j in range(self.num_atoms):
                self.rot_rel_times[i, self.num_molecules + j] = self.rot_relaxation_time_parker(
                    temperature,
                    number_density,
                    molecule,
                    self.interactions[self.inter_index(i, self.num_molecules + j)],
                    model,
                )

    def inplace_compute_n_molecule(self, n_vl_molecule: Sequence[np.ndarray]) -> None:
        for idx, vec in enumerate(n_vl_molecule):
            self.this_n_molecules[idx] = float(np.sum(vec))

    def compute_omega11(self, temperature: float, model: ModelsOmega = ModelsOmega.ESA) -> None:
        self.omega_11.fill(0.0)
        for i in range(self.num_molecules):
            for j in range(i, self.num_molecules):
                idx = self.inter_index(i, j)
                val = self.omega_integral(temperature, self.interactions[idx], 1, 1, model, True)
                self.omega_11[i, j] = self.omega_11[j, i] = val
            for j in range(self.num_atoms):
                idx = self.inter_index(i, self.num_molecules + j)
                val = self.omega_integral(temperature, self.interactions[idx], 1, 1, model, True)
                self.omega_11[i, self.num_molecules + j] = self.omega_11[self.num_molecules + j, i] = val
        for i in range(self.num_atoms):
            for j in range(i, self.num_atoms):
                idx = self.inter_index(self.num_molecules + i, self.num_molecules + j)
                val = self.omega_integral(temperature, self.interactions[idx], 1, 1, model, True)
                self.omega_11[self.num_molecules + i, self.num_molecules + j] = self.omega_11[self.num_molecules + j, self.num_molecules + i] = val
        if self.is_ionized:
            electron_idx = self.num_molecules + self.num_atoms
            for i in range(self.n_particles):
                idx = self.inter_index(min(i, electron_idx), max(i, electron_idx))
                val = self.omega_integral(temperature, self.interactions[idx], 1, 1, model, True)
                self.omega_11[i, electron_idx] = self.omega_11[electron_idx, i] = val

    def compute_omega12(self, temperature: float, model: ModelsOmega = ModelsOmega.ESA) -> None:
        self.omega_12.fill(0.0)
        for i in range(self.num_molecules):
            for j in range(i, self.num_molecules):
                idx = self.inter_index(i, j)
                val = self.omega_integral(temperature, self.interactions[idx], 1, 2, model, True)
                self.omega_12[i, j] = self.omega_12[j, i] = val
            for j in range(self.num_atoms):
                idx = self.inter_index(i, self.num_molecules + j)
                val = self.omega_integral(temperature, self.interactions[idx], 1, 2, model, True)
                self.omega_12[i, self.num_molecules + j] = self.omega_12[self.num_molecules + j, i] = val
        for i in range(self.num_atoms):
            for j in range(i, self.num_atoms):
                idx = self.inter_index(self.num_molecules + i, self.num_molecules + j)
                val = self.omega_integral(temperature, self.interactions[idx], 1, 2, model, True)
                self.omega_12[self.num_molecules + i, self.num_molecules + j] = self.omega_12[self.num_molecules + j, self.num_molecules + i] = val
        if self.is_ionized:
            electron_idx = self.num_molecules + self.num_atoms
            for i in range(self.n_particles):
                idx = self.inter_index(min(i, electron_idx), max(i, electron_idx))
                val = self.omega_integral(temperature, self.interactions[idx], 1, 2, model, True)
                self.omega_12[i, electron_idx] = self.omega_12[electron_idx, i] = val

    def compute_omega13(self, temperature: float, model: ModelsOmega = ModelsOmega.ESA) -> None:
        self.omega_13.fill(0.0)
        for i in range(self.num_molecules):
            for j in range(i, self.num_molecules):
                idx = self.inter_index(i, j)
                val = self.omega_integral(temperature, self.interactions[idx], 1, 3, model, True)
                self.omega_13[i, j] = self.omega_13[j, i] = val
            for j in range(self.num_atoms):
                idx = self.inter_index(i, self.num_molecules + j)
                val = self.omega_integral(temperature, self.interactions[idx], 1, 3, model, True)
                self.omega_13[i, self.num_molecules + j] = self.omega_13[self.num_molecules + j, i] = val
        for i in range(self.num_atoms):
            for j in range(i, self.num_atoms):
                idx = self.inter_index(self.num_molecules + i, self.num_molecules + j)
                val = self.omega_integral(temperature, self.interactions[idx], 1, 3, model, True)
                self.omega_13[self.num_molecules + i, self.num_molecules + j] = self.omega_13[self.num_molecules + j, self.num_molecules + i] = val
        if self.is_ionized:
            electron_idx = self.num_molecules + self.num_atoms
            for i in range(self.n_particles):
                idx = self.inter_index(min(i, electron_idx), max(i, electron_idx))
                val = self.omega_integral(temperature, self.interactions[idx], 1, 3, model, True)
                self.omega_13[i, electron_idx] = self.omega_13[electron_idx, i] = val

    def compute_omega22(self, temperature: float, model: ModelsOmega = ModelsOmega.ESA) -> None:
        self.omega_22.fill(0.0)
        for i in range(self.num_molecules):
            for j in range(i, self.num_molecules):
                idx = self.inter_index(i, j)
                val = self.omega_integral(temperature, self.interactions[idx], 2, 2, model, True)
                self.omega_22[i, j] = self.omega_22[j, i] = val
            for j in range(self.num_atoms):
                idx = self.inter_index(i, self.num_molecules + j)
                val = self.omega_integral(temperature, self.interactions[idx], 2, 2, model, True)
                self.omega_22[i, self.num_molecules + j] = self.omega_22[self.num_molecules + j, i] = val
        for i in range(self.num_atoms):
            for j in range(i, self.num_atoms):
                idx = self.inter_index(self.num_molecules + i, self.num_molecules + j)
                val = self.omega_integral(temperature, self.interactions[idx], 2, 2, model, True)
                self.omega_22[self.num_molecules + i, self.num_molecules + j] = self.omega_22[self.num_molecules + j, self.num_molecules + i] = val
        if self.is_ionized:
            electron_idx = self.num_molecules + self.num_atoms
            for i in range(self.n_particles):
                idx = self.inter_index(min(i, electron_idx), max(i, electron_idx))
                val = self.omega_integral(temperature, self.interactions[idx], 2, 2, model, True)
                self.omega_22[i, electron_idx] = self.omega_22[electron_idx, i] = val

    def compute_omega11_with_debye(self, temperature: float, debye_length: float, model: ModelsOmega) -> None:
        if not self.is_ionized:
            return
        electron_idx = self.num_molecules + self.num_atoms
        for i in range(self.n_particles):
            idx = self.inter_index(min(i, electron_idx), max(i, electron_idx))
            val = self.omega_integral_with_debye(temperature, self.interactions[idx], 1, 1, debye_length, model, True)
            self.omega_11[i, electron_idx] = self.omega_11[electron_idx, i] = val

    def compute_omega12_with_debye(self, temperature: float, debye_length: float, model: ModelsOmega) -> None:
        if not self.is_ionized:
            return
        electron_idx = self.num_molecules + self.num_atoms
        for i in range(self.n_particles):
            idx = self.inter_index(min(i, electron_idx), max(i, electron_idx))
            val = self.omega_integral_with_debye(temperature, self.interactions[idx], 1, 2, debye_length, model, True)
            self.omega_12[i, electron_idx] = self.omega_12[electron_idx, i] = val

    def compute_omega13_with_debye(self, temperature: float, debye_length: float, model: ModelsOmega) -> None:
        if not self.is_ionized:
            return
        electron_idx = self.num_molecules + self.num_atoms
        for i in range(self.n_particles):
            idx = self.inter_index(min(i, electron_idx), max(i, electron_idx))
            val = self.omega_integral_with_debye(temperature, self.interactions[idx], 1, 3, debye_length, model, True)
            self.omega_13[i, electron_idx] = self.omega_13[electron_idx, i] = val

    def compute_omega22_with_debye(self, temperature: float, debye_length: float, model: ModelsOmega) -> None:
        if not self.is_ionized:
            return
        electron_idx = self.num_molecules + self.num_atoms
        for i in range(self.n_particles):
            idx = self.inter_index(min(i, electron_idx), max(i, electron_idx))
            val = self.omega_integral_with_debye(temperature, self.interactions[idx], 2, 2, debye_length, model, True)
            self.omega_22[i, electron_idx] = self.omega_22[electron_idx, i] = val

    @staticmethod
    def _solve_linear_system(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return linalg.lstsq(lhs, rhs, cond=None, check_finite=False)[0]

    # ---------------- Bulk viscosity ---------------- #
    def compute_bulk_viscosity_LHS(self, temperature: float, model: ModelsOmega = ModelsOmega.ESA) -> None:
        self.bulk_viscosity_LHS.fill(0.0)
        n = self.this_total_n
        kT = constants.K_CONST_K * temperature

        for i in range(self.num_molecules):
            for j in range(i, self.num_molecules):
                coll_mass = self.interactions[self.inter_index(i, j)].collision_mass
                A_cd = 0.5 * self.omega_22[i, j] / self.omega_11[i, j]
                eta_cd = 0.625 * kT / self.omega_22[i, j]
                rm = 32.0 * n * self.omega_22[i, j] / (5 * constants.K_CONST_PI)
                for k, n_i in enumerate(self.this_n_vl_mol[i]):
                    p1 = 2 * (self.vl_offset[i] + k)
                    o1 = self.vl_offset[i] + k
                    for l, n_j in enumerate(self.this_n_vl_mol[j]):
                        n_ij = n_i * n_j
                        p2 = 2 * (self.vl_offset[j] + l)
                        o2 = self.vl_offset[j] + l
                        if j != i or l > k:
                            term_rot = (
                                4.0
                                * temperature
                                * n_ij
                                * coll_mass
                                * (
                                    self.molecules[i].mass * self.c_rot_arr[o1] / (rm * self.rot_rel_times[i, j])
                                    + self.molecules[j].mass * self.c_rot_arr[o2] / (rm * self.rot_rel_times[j, i])
                                )
                                / (constants.K_CONST_PI * eta_cd * (self.molecules[i].mass + self.molecules[j].mass))
                            )
                            self.bulk_viscosity_LHS[p1, p2] = (
                                -5 * kT * n_ij * coll_mass / (A_cd * eta_cd * (self.molecules[i].mass + self.molecules[j].mass))
                                + term_rot
                            )
                            self.bulk_viscosity_LHS[p1 + 1, p2] = (
                                -4
                                * temperature
                                * n_ij
                                * self.molecules[j].mass
                                * self.molecules[j].mass
                                * self.c_rot_arr[o2]
                                / (
                                    (self.molecules[i].mass + self.molecules[j].mass)
                                    * constants.K_CONST_PI
                                    * eta_cd
                                    * rm
                                    * self.rot_rel_times[j, i]
                                )
                            )
                            self.bulk_viscosity_LHS[p1, p2 + 1] = (
                                -4
                                * temperature
                                * n_ij
                                * self.molecules[i].mass
                                * self.molecules[i].mass
                                * self.c_rot_arr[o1]
                                / (
                                    (self.molecules[i].mass + self.molecules[j].mass)
                                    * constants.K_CONST_PI
                                    * eta_cd
                                    * rm
                                    * self.rot_rel_times[i, j]
                                )
                            )
                            self.bulk_viscosity_LHS[p1 + 1, p2 + 1] = 0.0
                            self.bulk_viscosity_LHS[p2, p1] = self.bulk_viscosity_LHS[p1, p2]
                            self.bulk_viscosity_LHS[p2, p1 + 1] = self.bulk_viscosity_LHS[p1 + 1, p2]
                            self.bulk_viscosity_LHS[p2 + 1, p1] = self.bulk_viscosity_LHS[p1, p2 + 1]
                            self.bulk_viscosity_LHS[p2 + 1, p1 + 1] = self.bulk_viscosity_LHS[p1 + 1, p2 + 1]

        for i in range(self.num_molecules):
            coll_mass = self.interactions[self.inter_index(i, i)].collision_mass
            A_cd = 0.5 * self.omega_22[i, i] / self.omega_11[i, i]
            eta_cd = 0.625 * kT / self.omega_22[i, i]
            rm = 32.0 * n * self.omega_22[i, i] / (5 * constants.K_CONST_PI)
            for k, n_i in enumerate(self.this_n_vl_mol[i]):
                p1 = 2 * (self.vl_offset[i] + k)
                o1 = self.vl_offset[i] + k
                n_ij = n_i * n_i
                tmp = 4.0 * temperature * n_ij * self.molecules[i].mass * self.c_rot_arr[o1] / (
                    constants.K_CONST_PI * eta_cd * rm * self.rot_rel_times[i, i]
                )
                self.bulk_viscosity_LHS[p1, p1] = tmp
                self.bulk_viscosity_LHS[p1 + 1, p1] = -tmp
                self.bulk_viscosity_LHS[p1, p1 + 1] = -tmp
                self.bulk_viscosity_LHS[p1 + 1, p1 + 1] = tmp

            for j in range(self.num_molecules):
                coll_mass = self.interactions[self.inter_index(i, j)].collision_mass
                A_cd = 0.5 * self.omega_22[i, j] / self.omega_11[i, j]
                eta_cd = 0.625 * kT / self.omega_22[i, j]
                rm = 32.0 * n * self.omega_22[i, j] / (5 * constants.K_CONST_PI)
                for k, n_i in enumerate(self.this_n_vl_mol[i]):
                    p1 = 2 * (self.vl_offset[i] + k)
                    o1 = self.vl_offset[i] + k
                    for l, n_j in enumerate(self.this_n_vl_mol[j]):
                        if k == l and i == j:
                            continue
                        n_ij = n_i * n_j
                        o2 = self.vl_offset[j] + l
                        self.bulk_viscosity_LHS[p1, p1] += (
                            5
                            * kT
                            * n_ij
                            * coll_mass
                            / ((self.molecules[i].mass + self.molecules[j].mass) * A_cd * eta_cd)
                            + 4
                            * temperature
                            * n_ij
                            * self.molecules[j].mass
                            * self.molecules[j].mass
                            * (
                                self.molecules[i].mass * self.c_rot_arr[o1] / (rm * self.rot_rel_times[i, j])
                                + self.molecules[j].mass * self.c_rot_arr[o2] / (rm * self.rot_rel_times[j, i])
                            )
                            / (
                                constants.K_CONST_PI
                                * eta_cd
                                * (self.molecules[i].mass + self.molecules[j].mass)
                                * (self.molecules[i].mass + self.molecules[j].mass)
                            )
                        )
                        tmp = (
                            -4
                            * temperature
                            * n_ij
                            * self.molecules[j].mass
                            * self.molecules[i].mass
                            * self.c_rot_arr[o1]
                            / (
                                rm
                                * self.rot_rel_times[i, j]
                                * constants.K_CONST_PI
                                * eta_cd
                                * (self.molecules[i].mass + self.molecules[j].mass)
                            )
                        )
                        self.bulk_viscosity_LHS[p1 + 1, p1] += tmp
                        self.bulk_viscosity_LHS[p1, p1 + 1] += tmp
                        self.bulk_viscosity_LHS[p1 + 1, p1 + 1] += (
                            4
                            * temperature
                            * n_ij
                            * self.molecules[i].mass
                            * self.c_rot_arr[o1]
                            / (constants.K_CONST_PI * eta_cd * rm * self.rot_rel_times[i, j])
                        )

            for j in range(self.num_atoms):
                coll_mass = self.interactions[self.inter_index(i, self.num_molecules + j)].collision_mass
                A_cd = 0.5 * self.omega_22[i, self.num_molecules + j] / self.omega_11[i, self.num_molecules + j]
                eta_cd = 0.625 * kT / self.omega_22[i, self.num_molecules + j]
                rm = 32.0 * n * self.omega_22[i, self.num_molecules + j] / (5 * constants.K_CONST_PI)
                for k, n_i in enumerate(self.this_n_vl_mol[i]):
                    p1 = 2 * (self.vl_offset[i] + k)
                    o1 = self.vl_offset[i] + k
                    n_ij = n_i * self.this_n_atom[j]
                    self.bulk_viscosity_LHS[p1, p1] += (
                        5
                        * kT
                        * n_ij
                        * coll_mass
                        / ((self.molecules[i].mass + self.atoms[j].mass) * A_cd * eta_cd)
                        + 4
                        * temperature
                        * n_ij
                        * self.atoms[j].mass
                        * self.atoms[j].mass
                        * self.molecules[i].mass
                        * self.c_rot_arr[o1]
                        / (
                            rm
                            * self.rot_rel_times[i, self.num_molecules + j]
                            * constants.K_CONST_PI
                            * eta_cd
                            * (self.molecules[i].mass + self.atoms[j].mass)
                            * (self.molecules[i].mass + self.atoms[j].mass)
                        )
                    )
                    tmp = (
                        -4
                        * temperature
                        * n_ij
                        * self.atoms[j].mass
                        * self.molecules[i].mass
                        * self.c_rot_arr[o1]
                        / (
                            rm
                            * self.rot_rel_times[i, self.num_molecules + j]
                            * constants.K_CONST_PI
                            * eta_cd
                            * (self.molecules[i].mass + self.atoms[j].mass)
                        )
                    )
                    self.bulk_viscosity_LHS[p1 + 1, p1] += tmp
                    self.bulk_viscosity_LHS[p1, p1 + 1] += tmp
                    self.bulk_viscosity_LHS[p1 + 1, p1 + 1] += (
                        4
                        * temperature
                        * n_ij
                        * self.molecules[i].mass
                        * self.c_rot_arr[o1]
                        / (constants.K_CONST_PI * eta_cd * rm * self.rot_rel_times[i, self.num_molecules + j])
                    )

        for i in range(self.num_molecules):
            for j in range(self.num_atoms):
                coll_mass = self.interactions[self.inter_index(i, self.num_molecules + j)].collision_mass
                A_cd = 0.5 * self.omega_22[i, self.num_molecules + j] / self.omega_11[i, self.num_molecules + j]
                eta_cd = 0.625 * kT / self.omega_22[i, self.num_molecules + j]
                rm = 32.0 * n * self.omega_22[i, self.num_molecules + j] / (5 * constants.K_CONST_PI)
                for k, n_i in enumerate(self.this_n_vl_mol[i]):
                    n_ij = n_i * self.this_n_atom[j]
                    p1 = 2 * (self.vl_offset[i] + k)
                    o1 = self.vl_offset[i] + k
                    idx = self.n_vibr_levels_total * 2 + j
                    self.bulk_viscosity_LHS[p1, idx] = (
                        -5 * kT * n_ij * coll_mass / (A_cd * eta_cd * (self.molecules[i].mass + self.atoms[j].mass))
                        + 4
                        * temperature
                        * n_ij
                        * coll_mass
                        * (self.molecules[i].mass * self.c_rot_arr[o1] / (rm * self.rot_rel_times[i, self.num_molecules + j]))
                        / (constants.K_CONST_PI * eta_cd * (self.molecules[i].mass + self.atoms[j].mass))
                    )
                    self.bulk_viscosity_LHS[p1 + 1, idx] = (
                        -4
                        * temperature
                        * n_ij
                        * self.molecules[i].mass
                        * self.molecules[i].mass
                        * self.c_rot_arr[o1]
                        / (
                            (self.molecules[i].mass + self.atoms[j].mass)
                            * constants.K_CONST_PI
                            * eta_cd
                            * rm
                            * self.rot_rel_times[i, self.num_molecules + j]
                        )
                    )
                    self.bulk_viscosity_LHS[idx, p1] = self.bulk_viscosity_LHS[p1, idx]
                    self.bulk_viscosity_LHS[idx, p1 + 1] = self.bulk_viscosity_LHS[p1 + 1, idx]

        for i in range(self.num_atoms - 1):
            for j in range(i + 1, self.num_atoms):
                coll_mass = self.interactions[self.inter_index(self.num_molecules + i, self.num_molecules + j)].collision_mass
                n_ij = self.this_n_atom[i] * self.this_n_atom[j]
                A_cd = 0.5 * self.omega_22[self.num_molecules + i, self.num_molecules + j] / self.omega_11[self.num_molecules + i, self.num_molecules + j]
                eta_cd = 0.625 * kT / self.omega_22[self.num_molecules + i, self.num_molecules + j]
                idx_i = self.n_vibr_levels_total * 2 + i
                idx_j = self.n_vibr_levels_total * 2 + j
                self.bulk_viscosity_LHS[idx_i, idx_j] = -5 * kT * n_ij * coll_mass / (A_cd * eta_cd * (self.atoms[i].mass + self.atoms[j].mass))
                self.bulk_viscosity_LHS[idx_j, idx_i] = self.bulk_viscosity_LHS[idx_i, idx_j]

        for i in range(self.num_atoms):
            idx_i = self.n_vibr_levels_total * 2 + i
            self.bulk_viscosity_LHS[idx_i, idx_i] = 0.0
            for j in range(self.num_molecules):
                coll_mass = self.interactions[self.inter_index(self.num_molecules + i, j)].collision_mass
                A_cd = 0.5 * self.omega_22[self.num_molecules + i, j] / self.omega_11[self.num_molecules + i, j]
                eta_cd = 0.625 * kT / self.omega_22[self.num_molecules + i, j]
                rm = 32.0 * n * self.omega_22[self.num_molecules + i, j] / (5 * constants.K_CONST_PI)
                for k, n_j in enumerate(self.this_n_vl_mol[j]):
                    n_ij = self.this_n_atom[i] * n_j
                    # Mirror the C++ implementation (uses vl_offset indexed by the atom loop variable)
                    o1 = self.vl_offset[i] + k
                    self.bulk_viscosity_LHS[idx_i, idx_i] += (
                        5
                        * kT
                        * n_ij
                        * coll_mass
                        / ((self.atoms[i].mass + self.molecules[j].mass) * A_cd * eta_cd)
                        + 4
                        * temperature
                        * n_ij
                        * self.molecules[j].mass
                        * self.molecules[j].mass
                        * self.molecules[j].mass
                        * self.c_rot_arr[o1]
                        / (
                            rm
                            * self.rot_rel_times[j, self.num_molecules + i]
                            * constants.K_CONST_PI
                            * eta_cd
                            * (self.atoms[i].mass + self.molecules[j].mass)
                            * (self.atoms[i].mass + self.molecules[j].mass)
                        )
                    )
            for j in range(self.num_atoms):
                if j == i:
                    continue
                coll_mass = self.interactions[self.inter_index(self.num_molecules + i, self.num_molecules + j)].collision_mass
                n_ij = self.this_n_atom[i] * self.this_n_atom[j]
                A_cd = 0.5 * self.omega_22[self.num_molecules + i, self.num_molecules + j] / self.omega_11[self.num_molecules + i, self.num_molecules + j]
                eta_cd = 0.625 * kT / self.omega_22[self.num_molecules + i, self.num_molecules + j]
                self.bulk_viscosity_LHS[idx_i, idx_i] += 5 * kT * n_ij * coll_mass / ((self.atoms[i].mass + self.atoms[j].mass) * A_cd * eta_cd)

        self.bulk_viscosity_LHS /= n * n
        for i in range(self.num_molecules):
            for k, n_i in enumerate(self.this_n_vl_mol[i]):
                p1 = 2 * (self.vl_offset[i] + k)
                self.bulk_viscosity_LHS[0, p1] = n_i * self.this_ctr / (n * 1e20)
                self.bulk_viscosity_LHS[0, p1 + 1] = n_i * self.c_rot_arr[self.vl_offset[i] + k] / (n * 1e20)
        for i in range(self.num_atoms):
            self.bulk_viscosity_LHS[0, self.n_vibr_levels_total * 2 + i] = self.this_n_atom[i] * self.this_ctr / (n * 1e20)

    def compute_bulk_viscosity_RHS(self, temperature: float) -> None:
        self.bulk_viscosity_RHS.fill(0.0)
        j = 0
        for i in range(self.num_molecules):
            for k, n_i in enumerate(self.this_n_vl_mol[i]):
                self.bulk_viscosity_RHS[2 * j] = -n_i * self.this_crot / self.this_total_n
                self.bulk_viscosity_RHS[2 * j + 1] = n_i * self.molecules[i].mass * self.c_rot_arr[j] / self.this_total_dens
                j += 1
        for i in range(self.num_atoms):
            self.bulk_viscosity_RHS[self.n_vibr_levels_total * 2 + i] = -self.this_n_atom[i] * self.this_crot / self.this_total_n
        self.bulk_viscosity_RHS[0] = 0.0
        self.bulk_viscosity_RHS /= self.this_ctr + self.this_crot

    def compute_bulk_viscosity_coeffs(self, temperature: float, model: ModelsOmega = ModelsOmega.ESA) -> None:
        self.compute_bulk_viscosity_LHS(temperature, model)
        self.compute_bulk_viscosity_RHS(temperature)
        lhs = self.bulk_viscosity_LHS * 1e20
        rhs = self.bulk_viscosity_RHS
        self.bulk_viscosity_coeffs = self._solve_linear_system(lhs, rhs) * 1e20

    def compute_bulk_viscosity_rigid_rot_LHS(self, temperature: float, model: ModelsOmega = ModelsOmega.ESA) -> None:
        self.bulk_viscosity_rigid_rot_LHS.fill(0.0)
        n = self.this_total_n
        kT = constants.K_CONST_K * temperature

        for i in range(self.num_molecules):
            for j in range(i + 1, self.num_molecules):
                coll_mass = self.interactions[self.inter_index(i, j)].collision_mass
                A_cd = 0.5 * self.omega_22[i, j] / self.omega_11[i, j]
                eta_cd = 0.625 * kT / self.omega_22[i, j]
                rm = 32.0 * n * self.omega_22[i, j] / (5 * constants.K_CONST_PI)
                n_ij = self.this_n_molecules[i] * self.this_n_molecules[j]

                tmp = (
                    -5 * kT * n_ij * coll_mass / (A_cd * eta_cd * (self.molecules[i].mass + self.molecules[j].mass))
                    + 4
                    * temperature
                    * n_ij
                    * coll_mass
                    * (
                        self.molecules[i].mass * self.c_rot_rigid_rot_arr[i] / (rm * self.rot_rel_times[i, j])
                        + self.molecules[j].mass * self.c_rot_rigid_rot_arr[j] / (rm * self.rot_rel_times[j, i])
                    )
                    / (constants.K_CONST_PI * eta_cd * (self.molecules[i].mass + self.molecules[j].mass))
                )
                self.bulk_viscosity_rigid_rot_LHS[2 * i, 2 * j] = tmp
                self.bulk_viscosity_rigid_rot_LHS[2 * j, 2 * i] = tmp

                tmp = (
                    -4
                    * temperature
                    * n_ij
                    * self.molecules[j].mass
                    * self.molecules[j].mass
                    * self.c_rot_rigid_rot_arr[j]
                    / (
                        (self.molecules[i].mass + self.molecules[j].mass)
                        * constants.K_CONST_PI
                        * eta_cd
                        * rm
                        * self.rot_rel_times[j, i]
                    )
                )
                self.bulk_viscosity_rigid_rot_LHS[2 * i + 1, 2 * j] = tmp
                self.bulk_viscosity_rigid_rot_LHS[2 * j, 2 * i + 1] = tmp

                tmp = (
                    -4
                    * temperature
                    * n_ij
                    * self.molecules[i].mass
                    * self.molecules[i].mass
                    * self.c_rot_rigid_rot_arr[i]
                    / (
                        (self.molecules[i].mass + self.molecules[j].mass)
                        * constants.K_CONST_PI
                        * eta_cd
                        * rm
                        * self.rot_rel_times[i, j]
                    )
                )
                self.bulk_viscosity_rigid_rot_LHS[2 * i, 2 * j + 1] = tmp
                self.bulk_viscosity_rigid_rot_LHS[2 * j + 1, 2 * i] = tmp
                self.bulk_viscosity_rigid_rot_LHS[2 * i + 1, 2 * j + 1] = 0.0

        for i in range(self.num_molecules):
            coll_mass = self.interactions[self.inter_index(i, i)].collision_mass
            A_cd = 0.5 * self.omega_22[i, i] / self.omega_11[i, i]
            eta_cd = 0.625 * kT / self.omega_22[i, i]
            rm = 32.0 * n * self.omega_22[i, i] / (5 * constants.K_CONST_PI)
            n_ij = self.this_n_molecules[i] * self.this_n_molecules[i]
            tmp = 4.0 * temperature * n_ij * self.molecules[i].mass * self.c_rot_rigid_rot_arr[i] / (
                constants.K_CONST_PI * eta_cd * rm * self.rot_rel_times[i, i]
            )
            self.bulk_viscosity_rigid_rot_LHS[2 * i, 2 * i] = tmp
            self.bulk_viscosity_rigid_rot_LHS[2 * i + 1, 2 * i] = -tmp
            self.bulk_viscosity_rigid_rot_LHS[2 * i, 2 * i + 1] = -tmp
            self.bulk_viscosity_rigid_rot_LHS[2 * i + 1, 2 * i + 1] = tmp

            for j in range(self.num_molecules):
                coll_mass = self.interactions[self.inter_index(i, j)].collision_mass
                A_cd = 0.5 * self.omega_22[i, j] / self.omega_11[i, j]
                eta_cd = 0.625 * kT / self.omega_22[i, j]
                rm = 32.0 * n * self.omega_22[i, j] / (5 * constants.K_CONST_PI)
                n_ij = self.this_n_molecules[i] * self.this_n_molecules[j]
                if i != j:
                    self.bulk_viscosity_rigid_rot_LHS[2 * i, 2 * i] += (
                        5
                        * kT
                        * n_ij
                        * coll_mass
                        / ((self.molecules[i].mass + self.molecules[j].mass) * A_cd * eta_cd)
                        + 4
                        * temperature
                        * n_ij
                        * self.molecules[j].mass
                        * self.molecules[j].mass
                        * (
                            self.molecules[i].mass * self.c_rot_rigid_rot_arr[i] / (rm * self.rot_rel_times[i, j])
                            + self.molecules[j].mass * self.c_rot_rigid_rot_arr[j] / (rm * self.rot_rel_times[j, i])
                        )
                        / (
                            constants.K_CONST_PI
                            * eta_cd
                            * (self.molecules[i].mass + self.molecules[j].mass)
                            * (self.molecules[i].mass + self.molecules[j].mass)
                        )
                    )
                    tmp = (
                        -4
                        * temperature
                        * n_ij
                        * self.molecules[j].mass
                        * self.molecules[i].mass
                        * self.c_rot_rigid_rot_arr[i]
                        / (
                            rm
                            * self.rot_rel_times[i, j]
                            * constants.K_CONST_PI
                            * eta_cd
                            * (self.molecules[i].mass + self.molecules[j].mass)
                        )
                    )
                    self.bulk_viscosity_rigid_rot_LHS[2 * i + 1, 2 * i] += tmp
                    self.bulk_viscosity_rigid_rot_LHS[2 * i, 2 * i + 1] += tmp
                    self.bulk_viscosity_rigid_rot_LHS[2 * i + 1, 2 * i + 1] += (
                        4
                        * temperature
                        * n_ij
                        * self.molecules[i].mass
                        * self.c_rot_rigid_rot_arr[i]
                        / (constants.K_CONST_PI * eta_cd * rm * self.rot_rel_times[i, j])
                    )

            for j in range(self.num_atoms):
                coll_mass = self.interactions[self.inter_index(i, self.num_molecules + j)].collision_mass
                A_cd = 0.5 * self.omega_22[i, self.num_molecules + j] / self.omega_11[i, self.num_molecules + j]
                eta_cd = 0.625 * kT / self.omega_22[i, self.num_molecules + j]
                rm = 32.0 * n * self.omega_22[i, self.num_molecules + j] / (5 * constants.K_CONST_PI)
                n_ij = self.this_n_molecules[i] * self.this_n_atom[j]
                self.bulk_viscosity_rigid_rot_LHS[2 * i, 2 * i] += (
                    5
                    * kT
                    * n_ij
                    * coll_mass
                    / ((self.molecules[i].mass + self.atoms[j].mass) * A_cd * eta_cd)
                    + 4
                    * temperature
                    * n_ij
                    * self.atoms[j].mass
                    * self.atoms[j].mass
                    * self.molecules[i].mass
                    * self.c_rot_rigid_rot_arr[i]
                    / (
                        rm
                        * self.rot_rel_times[i, self.num_molecules + j]
                        * constants.K_CONST_PI
                        * eta_cd
                        * (self.molecules[i].mass + self.atoms[j].mass)
                        * (self.molecules[i].mass + self.atoms[j].mass)
                    )
                )
                tmp = (
                    -4
                    * temperature
                    * n_ij
                    * self.atoms[j].mass
                    * self.molecules[i].mass
                    * self.c_rot_rigid_rot_arr[i]
                    / (
                        rm
                        * self.rot_rel_times[i, self.num_molecules + j]
                        * constants.K_CONST_PI
                        * eta_cd
                        * (self.molecules[i].mass + self.atoms[j].mass)
                    )
                )
                self.bulk_viscosity_rigid_rot_LHS[2 * i + 1, 2 * i] += tmp
                self.bulk_viscosity_rigid_rot_LHS[2 * i, 2 * i + 1] += tmp
                self.bulk_viscosity_rigid_rot_LHS[2 * i + 1, 2 * i + 1] += (
                    4
                    * temperature
                    * n_ij
                    * self.molecules[i].mass
                    * self.c_rot_rigid_rot_arr[i]
                    / (constants.K_CONST_PI * eta_cd * rm * self.rot_rel_times[i, self.num_molecules + j])
                )

        for i in range(self.num_molecules):
            for j in range(self.num_atoms):
                coll_mass = self.interactions[self.inter_index(i, self.num_molecules + j)].collision_mass
                A_cd = 0.5 * self.omega_22[i, self.num_molecules + j] / self.omega_11[i, self.num_molecules + j]
                eta_cd = 0.625 * kT / self.omega_22[i, self.num_molecules + j]
                rm = 32.0 * n * self.omega_22[i, self.num_molecules + j] / (5 * constants.K_CONST_PI)
                n_ij = self.this_n_molecules[i] * self.this_n_atom[j]
                tmp = (
                    -5 * kT * n_ij * coll_mass / (A_cd * eta_cd * (self.molecules[i].mass + self.atoms[j].mass))
                    + 4
                    * temperature
                    * n_ij
                    * coll_mass
                    * (self.molecules[i].mass * self.c_rot_rigid_rot_arr[i] / (rm * self.rot_rel_times[i, self.num_molecules + j]))
                    / (constants.K_CONST_PI * eta_cd * (self.molecules[i].mass + self.atoms[j].mass))
                )
                self.bulk_viscosity_rigid_rot_LHS[2 * i, self.num_molecules * 2 + j] = tmp
                self.bulk_viscosity_rigid_rot_LHS[self.num_molecules * 2 + j, 2 * i] = tmp
                tmp = (
                    -4
                    * temperature
                    * n_ij
                    * self.molecules[i].mass
                    * self.molecules[i].mass
                    * self.c_rot_rigid_rot_arr[i]
                    / (
                        (self.molecules[i].mass + self.atoms[j].mass)
                        * constants.K_CONST_PI
                        * eta_cd
                        * rm
                        * self.rot_rel_times[i, self.num_molecules + j]
                    )
                )
                self.bulk_viscosity_rigid_rot_LHS[2 * i + 1, self.num_molecules * 2 + j] = tmp
                self.bulk_viscosity_rigid_rot_LHS[self.num_molecules * 2 + j, 2 * i + 1] = tmp

        for i in range(self.num_atoms - 1):
            for j in range(i + 1, self.num_atoms):
                coll_mass = self.interactions[self.inter_index(self.num_molecules + i, self.num_molecules + j)].collision_mass
                A_cd = 0.5 * self.omega_22[self.num_molecules + i, self.num_molecules + j] / self.omega_11[self.num_molecules + i, self.num_molecules + j]
                eta_cd = 0.625 * kT / self.omega_22[self.num_molecules + i, self.num_molecules + j]
                n_ij = self.this_n_atom[i] * self.this_n_atom[j]
                tmp = -5 * kT * n_ij * coll_mass / (A_cd * eta_cd * (self.atoms[i].mass + self.atoms[j].mass))
                self.bulk_viscosity_rigid_rot_LHS[self.num_molecules * 2 + i, self.num_molecules * 2 + j] = tmp
                self.bulk_viscosity_rigid_rot_LHS[self.num_molecules * 2 + j, self.num_molecules * 2 + i] = tmp

        for i in range(self.num_atoms):
            idx = self.num_molecules * 2 + i
            self.bulk_viscosity_rigid_rot_LHS[idx, idx] = 0.0
            for j in range(self.num_molecules):
                coll_mass = self.interactions[self.inter_index(self.num_molecules + i, j)].collision_mass
                A_cd = 0.5 * self.omega_22[self.num_molecules + i, j] / self.omega_11[self.num_molecules + i, j]
                eta_cd = 0.625 * kT / self.omega_22[self.num_molecules + i, j]
                rm = 32.0 * n * self.omega_22[self.num_molecules + i, j] / (5 * constants.K_CONST_PI)
                n_ij = self.this_n_atom[i] * self.this_n_molecules[j]
                self.bulk_viscosity_rigid_rot_LHS[idx, idx] += (
                    5
                    * kT
                    * n_ij
                    * coll_mass
                    / ((self.atoms[i].mass + self.molecules[j].mass) * A_cd * eta_cd)
                    + 4
                    * temperature
                    * n_ij
                    * self.molecules[j].mass
                    * self.molecules[j].mass
                    * self.molecules[j].mass
                    * self.c_rot_rigid_rot_arr[j]
                    / (
                        rm
                        * self.rot_rel_times[j, self.num_molecules + i]
                        * constants.K_CONST_PI
                        * eta_cd
                        * (self.atoms[i].mass + self.molecules[j].mass)
                        * (self.atoms[i].mass + self.molecules[j].mass)
                    )
                )
            for j in range(self.num_atoms):
                if j == i:
                    continue
                coll_mass = self.interactions[self.inter_index(self.num_molecules + i, self.num_molecules + j)].collision_mass
                A_cd = 0.5 * self.omega_22[self.num_molecules + i, self.num_molecules + j] / self.omega_11[self.num_molecules + i, self.num_molecules + j]
                eta_cd = 0.625 * kT / self.omega_22[self.num_molecules + i, self.num_molecules + j]
                n_ij = self.this_n_atom[i] * self.this_n_atom[j]
                self.bulk_viscosity_rigid_rot_LHS[idx, idx] += 5 * kT * n_ij * coll_mass / ((self.atoms[i].mass + self.atoms[j].mass) * A_cd * eta_cd)

        self.bulk_viscosity_rigid_rot_LHS /= n * n
        for i in range(self.num_molecules):
            self.bulk_viscosity_rigid_rot_LHS[0, 2 * i] = self.this_n_molecules[i] * self.this_ctr / (n * 1e20)
            self.bulk_viscosity_rigid_rot_LHS[0, 2 * i + 1] = self.this_n_molecules[i] * self.c_rot_rigid_rot_arr[i] / (n * 1e20)
        for i in range(self.num_atoms):
            self.bulk_viscosity_rigid_rot_LHS[0, self.num_molecules * 2 + i] = self.this_n_atom[i] * self.this_ctr / (n * 1e20)

    def compute_bulk_viscosity_rigid_rot_RHS(self, temperature: float) -> None:
        self.bulk_viscosity_rigid_rot_RHS.fill(0.0)
        for i in range(self.num_molecules):
            self.bulk_viscosity_rigid_rot_RHS[2 * i] = -self.this_n_molecules[i] * self.this_crot / self.this_total_n
            self.bulk_viscosity_rigid_rot_RHS[2 * i + 1] = (
                self.this_n_molecules[i] * self.molecules[i].mass * self.c_rot_rigid_rot_arr[i] / self.this_total_dens
            )
        for i in range(self.num_atoms):
            self.bulk_viscosity_rigid_rot_RHS[self.num_molecules * 2 + i] = -self.this_n_atom[i] * self.this_crot / self.this_total_n
        self.bulk_viscosity_rigid_rot_RHS[0] = 0.0
        self.bulk_viscosity_rigid_rot_RHS /= self.this_ctr + self.this_crot

    def compute_bulk_viscosity_rigid_rot_coeffs(self, temperature: float, model: ModelsOmega = ModelsOmega.ESA) -> None:
        self.compute_bulk_viscosity_rigid_rot_LHS(temperature, model)
        self.compute_bulk_viscosity_rigid_rot_RHS(temperature)
        lhs = self.bulk_viscosity_rigid_rot_LHS * 1e17
        rhs = self.bulk_viscosity_rigid_rot_RHS
        self.bulk_viscosity_rigid_rot_coeffs = self._solve_linear_system(lhs, rhs) * 1e17

    def bulk_viscosity(self, temperature: float, model: ModelsOmega = ModelsOmega.ESA) -> float:
        res = 0.0
        if self.all_rigid_rotators:
            self.compute_bulk_viscosity_rigid_rot_coeffs(temperature, model)
            for i in range(self.num_molecules):
                res += self.this_n_molecules[i] * self.bulk_viscosity_rigid_rot_coeffs[2 * i]
            for i in range(self.num_atoms):
                res += self.this_n_atom[i] * self.bulk_viscosity_rigid_rot_coeffs[self.num_molecules * 2 + i]
        else:
            self.compute_bulk_viscosity_coeffs(temperature, model)
            j = 0
            for i in range(self.num_molecules):
                for k in range(self.molecules[i].num_vibr_levels[0]):
                    res += self.this_n_vl_mol[i][k] * self.bulk_viscosity_coeffs[2 * j]
                    j += 1
            for i in range(self.num_atoms):
                res += self.this_n_atom[i] * self.bulk_viscosity_coeffs[2 * self.n_vibr_levels_total + i]
        return -res * constants.K_CONST_K * temperature / self.this_total_n

    # ---------------- Thermal conductivity ---------------- #
    def compute_thermal_conductivity_LHS(self, temperature: float, model: ModelsOmega = ModelsOmega.ESA) -> None:
        self.thermal_conductivity_LHS.fill(0.0)
        n = self.this_total_n
        kT = constants.K_CONST_K * temperature

        for i in range(self.num_molecules):
            for j in range(i, self.num_molecules):
                coll_mass = self.interactions[self.inter_index(i, j)].collision_mass
                A_cd = 0.5 * self.omega_22[i, j] / self.omega_11[i, j]
                B_cd = (5 * self.omega_12[i, j] - self.omega_13[i, j]) / (3 * self.omega_11[i, j])
                C_cd = self.omega_12[i, j] / (3 * self.omega_11[i, j])
                D_cd = (3.0 / 16.0) * kT / (n * coll_mass * self.omega_11[i, j])
                rm = 32.0 * n * self.omega_22[i, j] / (5 * constants.K_CONST_PI)
                for k, n_i in enumerate(self.this_n_vl_mol[i]):
                    p1 = 3 * (self.vl_offset[i] + k)
                    o1 = self.vl_offset[i] + k
                    for l, n_j in enumerate(self.this_n_vl_mol[j]):
                        n_ij = n_i * n_j
                        p2 = 3 * (self.vl_offset[j] + l)
                        o2 = self.vl_offset[j] + l
                        if j != i or l > k:
                            self.thermal_conductivity_LHS[p1, p2] = -1.5 * kT * n_ij / (n * D_cd)
                            self.thermal_conductivity_LHS[p1 + 1, p2] = (
                                0.75
                                * kT
                                * n_ij
                                * (6 * C_cd - 5)
                                * self.molecules[j].mass
                                / ((self.molecules[i].mass + self.molecules[j].mass) * n * D_cd)
                            )
                            self.thermal_conductivity_LHS[p1 + 2, p2] = 0.0
                            self.thermal_conductivity_LHS[p1, p2 + 1] = (
                                0.75
                                * kT
                                * n_ij
                                * (6 * C_cd - 5)
                                * self.molecules[i].mass
                                / ((self.molecules[i].mass + self.molecules[j].mass) * n * D_cd)
                            )
                            self.thermal_conductivity_LHS[p1 + 1, p2 + 1] = (
                                -1.5
                                * kT
                                * n_ij
                                * coll_mass
                                / ((self.molecules[i].mass + self.molecules[j].mass) * n * D_cd)
                                * (
                                    13.75
                                    - 3 * B_cd
                                    - 4 * A_cd
                                    - (20.0 / 3.0)
                                    * A_cd
                                    / (constants.K_CONST_K * constants.K_CONST_PI)
                                    * (
                                        self.molecules[i].mass * self.c_rot_arr[o1] / (rm * self.rot_rel_times[i, j])
                                        + self.molecules[j].mass * self.c_rot_arr[o2] / (rm * self.rot_rel_times[j, i])
                                    )
                                )
                            )
                            rot_term = (
                                -6
                                * temperature
                                / (n * D_cd * constants.K_CONST_PI)
                                * A_cd
                                * n_ij
                                * self.molecules[i].mass
                                * self.molecules[i].mass
                                * self.c_rot_arr[o1]
                                / (rm * self.rot_rel_times[i, j] * (self.molecules[i].mass + self.molecules[j].mass))
                            )
                            self.thermal_conductivity_LHS[p1 + 2, p2 + 1] = rot_term
                            self.thermal_conductivity_LHS[p1, p2 + 2] = 0.0
                            rot_term_2 = (
                                -6
                                * temperature
                                / (n * D_cd * constants.K_CONST_PI)
                                * A_cd
                                * n_ij
                                * self.molecules[j].mass
                                * self.molecules[j].mass
                                * self.c_rot_arr[o2]
                                / (rm * self.rot_rel_times[j, i] * (self.molecules[i].mass + self.molecules[j].mass))
                            )
                            self.thermal_conductivity_LHS[p1 + 1, p2 + 2] = rot_term_2
                            self.thermal_conductivity_LHS[p1 + 2, p2 + 2] = 0.0
                            self.thermal_conductivity_LHS[p2, p1] = self.thermal_conductivity_LHS[p1, p2]
                            self.thermal_conductivity_LHS[p2, p1 + 1] = self.thermal_conductivity_LHS[p1 + 1, p2]
                            self.thermal_conductivity_LHS[p2, p1 + 2] = self.thermal_conductivity_LHS[p1 + 2, p2]
                            self.thermal_conductivity_LHS[p2 + 1, p1] = self.thermal_conductivity_LHS[p1, p2 + 1]
                            self.thermal_conductivity_LHS[p2 + 1, p1 + 1] = self.thermal_conductivity_LHS[p1 + 1, p2 + 1]
                            self.thermal_conductivity_LHS[p2 + 1, p1 + 2] = self.thermal_conductivity_LHS[p1 + 2, p2 + 1]
                            self.thermal_conductivity_LHS[p2 + 2, p1] = self.thermal_conductivity_LHS[p1, p2 + 2]
                            self.thermal_conductivity_LHS[p2 + 2, p1 + 1] = self.thermal_conductivity_LHS[p1 + 1, p2 + 2]
                            self.thermal_conductivity_LHS[p2 + 2, p1 + 2] = self.thermal_conductivity_LHS[p1 + 2, p2 + 2]

        for i in range(self.num_molecules):
            coll_mass = self.interactions[self.inter_index(i, i)].collision_mass
            A_cd = 0.5 * self.omega_22[i, i] / self.omega_11[i, i]
            B_cd = (5 * self.omega_12[i, i] - self.omega_13[i, i]) / (3 * self.omega_11[i, i])
            C_cd = self.omega_12[i, i] / (3 * self.omega_11[i, i])
            D_cd = (3.0 / 8.0) * kT / (n * self.molecules[i].mass * self.omega_11[i, i])
            rm = 32.0 * n * self.omega_22[i, i] / (5 * constants.K_CONST_PI)
            D_cd_rot = D_cd
            for k, n_i in enumerate(self.this_n_vl_mol[i]):
                p1 = 3 * (self.vl_offset[i] + k)
                o1 = self.vl_offset[i] + k
                n_ij = n_i * n_i
                self.thermal_conductivity_LHS[p1, p1] = 0.0
                self.thermal_conductivity_LHS[p1 + 1, p1] = 0.0
                self.thermal_conductivity_LHS[p1 + 2, p1] = 0.0
                self.thermal_conductivity_LHS[p1, p1 + 1] = 0.0
                self.thermal_conductivity_LHS[p1 + 1, p1 + 1] = (
                    1.5
                    * kT
                    * n_ij
                    * A_cd
                    * (2 + (20.0 / 3.0) * self.molecules[i].mass * self.c_rot_arr[o1] / (constants.K_CONST_K * constants.K_CONST_PI * rm * self.rot_rel_times[i, i]))
                    / (n * D_cd)
                )
                tmp = -6.0 * temperature * A_cd * n_ij * self.molecules[i].mass * self.c_rot_arr[o1] / (
                    constants.K_CONST_PI * rm * self.rot_rel_times[i, i] * n * D_cd
                )
                self.thermal_conductivity_LHS[p1 + 2, p1 + 1] = tmp
                self.thermal_conductivity_LHS[p1, p1 + 2] = 0.0
                self.thermal_conductivity_LHS[p1 + 1, p1 + 2] = tmp
                self.thermal_conductivity_LHS[p1 + 2, p1 + 2] = (
                    temperature
                    * n_ij
                    * (self.molecules[i].mass * self.c_rot_arr[o1] / n)
                    * (1.5 / D_cd_rot + 3.6 * A_cd / (constants.K_CONST_PI * D_cd * rm * self.rot_rel_times[i, i]))
                )

            for j in range(self.num_molecules):
                coll_mass = self.interactions[self.inter_index(i, j)].collision_mass
                A_cd = 0.5 * self.omega_22[i, j] / self.omega_11[i, j]
                B_cd = (5 * self.omega_12[i, j] - self.omega_13[i, j]) / (3 * self.omega_11[i, j])
                C_cd = self.omega_12[i, j] / (3 * self.omega_11[i, j])
                D_cd = (3.0 / 16.0) * kT / (n * coll_mass * self.omega_11[i, j])
                D_cd_rot = D_cd
                rm = 32.0 * n * self.omega_22[i, j] / (5 * constants.K_CONST_PI)
                for k, n_i in enumerate(self.this_n_vl_mol[i]):
                    p1 = 3 * (self.vl_offset[i] + k)
                    o1 = self.vl_offset[i] + k
                    for l, n_j in enumerate(self.this_n_vl_mol[j]):
                        if k == l and i == j:
                            continue
                        n_ij = n_i * n_j
                        o2 = self.vl_offset[j] + l
                        self.thermal_conductivity_LHS[p1, p1] += 1.5 * kT * n_ij / (D_cd * n)
                        tmp = (
                            0.75
                            * kT
                            * n_ij
                            * self.molecules[j].mass
                            * (6 * C_cd - 5)
                            / ((self.molecules[i].mass + self.molecules[j].mass) * n * D_cd)
                        )
                        self.thermal_conductivity_LHS[p1 + 1, p1] -= tmp
                        self.thermal_conductivity_LHS[p1, p1 + 1] -= tmp
                        self.thermal_conductivity_LHS[p1 + 1, p1 + 1] += (
                            1.5
                            * kT
                            * coll_mass
                            * n_ij
                            * (
                                7.5 * self.molecules[i].mass / self.molecules[j].mass
                                + 6.25 * self.molecules[j].mass / self.molecules[i].mass
                                - 3 * self.molecules[j].mass * B_cd / self.molecules[i].mass
                                + 4 * A_cd
                                + (20.0 / 3.0)
                                * A_cd
                                * (
                                    self.molecules[i].mass * self.c_rot_arr[o1] / (rm * self.rot_rel_times[i, j])
                                    + self.molecules[j].mass * self.c_rot_arr[o2] / (rm * self.rot_rel_times[j, i])
                                )
                                / (constants.K_CONST_PI * constants.K_CONST_K)
                            )
                            / (n * D_cd * (self.molecules[i].mass + self.molecules[j].mass))
                        )
                        tmp_rot = (
                            -6.0
                            * temperature
                            * A_cd
                            * n_ij
                            * self.molecules[i].mass
                            * self.c_rot_arr[o1]
                            / (rm * self.rot_rel_times[i, j])
                            * self.molecules[i].mass
                            / (n * D_cd * constants.K_CONST_PI * (self.molecules[i].mass + self.molecules[j].mass))
                        )
                        self.thermal_conductivity_LHS[p1 + 2, p1 + 1] += tmp_rot
                        self.thermal_conductivity_LHS[p1 + 1, p1 + 2] += tmp_rot
                        self.thermal_conductivity_LHS[p1 + 2, p1 + 2] += (
                            temperature
                            * n_ij
                            * self.molecules[i].mass
                            * self.c_rot_arr[o1]
                            * (
                                1.5 / D_cd_rot
                                + 3.6
                                * A_cd
                                * self.molecules[i].mass
                                / (constants.K_CONST_PI * D_cd * self.molecules[j].mass * rm * self.rot_rel_times[i, j])
                            )
                            / n
                        )

            for j in range(self.num_atoms):
                coll_mass = self.interactions[self.inter_index(i, self.num_molecules + j)].collision_mass
                A_cd = 0.5 * self.omega_22[i, self.num_molecules + j] / self.omega_11[i, self.num_molecules + j]
                B_cd = (5 * self.omega_12[i, self.num_molecules + j] - self.omega_13[i, self.num_molecules + j]) / (
                    3 * self.omega_11[i, self.num_molecules + j]
                )
                C_cd = self.omega_12[i, self.num_molecules + j] / (3 * self.omega_11[i, self.num_molecules + j])
                D_cd = (3.0 / 16.0) * kT / (n * coll_mass * self.omega_11[i, self.num_molecules + j])
                D_cd_rot = D_cd
                rm = 32.0 * n * self.omega_22[i, self.num_molecules + j] / (5 * constants.K_CONST_PI)
                for k, n_i in enumerate(self.this_n_vl_mol[i]):
                    p1 = 3 * (self.vl_offset[i] + k)
                    o1 = self.vl_offset[i] + k
                    n_ij = n_i * self.this_n_atom[j]
                    self.thermal_conductivity_LHS[p1, p1] += 1.5 * kT * n_ij / (D_cd * n)
                    tmp = (
                        0.75
                        * kT
                        * n_ij
                        * self.atoms[j].mass
                        * (6 * C_cd - 5)
                        / ((self.molecules[i].mass + self.atoms[j].mass) * n * D_cd)
                    )
                    self.thermal_conductivity_LHS[p1 + 1, p1] -= tmp
                    self.thermal_conductivity_LHS[p1, p1 + 1] -= tmp
                    self.thermal_conductivity_LHS[p1 + 1, p1 + 1] += (
                        1.5
                        * kT
                        * coll_mass
                        * n_ij
                        * (
                            7.5 * self.molecules[i].mass / self.atoms[j].mass
                            + 6.25 * self.atoms[j].mass / self.molecules[i].mass
                            - 3 * self.atoms[j].mass * B_cd / self.molecules[i].mass
                            + 4 * A_cd
                            + (20.0 / 3.0)
                            * A_cd
                            * (self.molecules[i].mass * self.c_rot_arr[o1] / (rm * self.rot_rel_times[i, self.num_molecules + j]))
                            / (constants.K_CONST_PI * constants.K_CONST_K)
                        )
                        / (n * D_cd * (self.molecules[i].mass + self.atoms[j].mass))
                    )
                    tmp_rot = (
                        -6.0
                        * temperature
                        * A_cd
                        * n_ij
                        * self.molecules[i].mass
                        * self.c_rot_arr[o1]
                        / (rm * self.rot_rel_times[i, self.num_molecules + j])
                        * self.molecules[i].mass
                        / (n * D_cd * constants.K_CONST_PI * (self.molecules[i].mass + self.atoms[j].mass))
                    )
                    self.thermal_conductivity_LHS[p1 + 2, p1 + 1] += tmp_rot
                    self.thermal_conductivity_LHS[p1 + 1, p1 + 2] += tmp_rot
                    self.thermal_conductivity_LHS[p1 + 2, p1 + 2] += (
                        temperature
                        * n_ij
                        * self.molecules[i].mass
                        * self.c_rot_arr[o1]
                        * (
                            1.5 / D_cd_rot
                            + 3.6
                            * A_cd
                            * self.molecules[i].mass
                            / (constants.K_CONST_PI * D_cd * self.atoms[j].mass * rm * self.rot_rel_times[i, self.num_molecules + j])
                        )
                        / n
                    )

        for i in range(self.num_molecules):
            for j in range(self.num_atoms):
                coll_mass = self.interactions[self.inter_index(i, self.num_molecules + j)].collision_mass
                A_cd = 0.5 * self.omega_22[i, self.num_molecules + j] / self.omega_11[i, self.num_molecules + j]
                B_cd = (5 * self.omega_12[i, self.num_molecules + j] - self.omega_13[i, self.num_molecules + j]) / (
                    3 * self.omega_11[i, self.num_molecules + j]
                )
                C_cd = self.omega_12[i, self.num_molecules + j] / (3 * self.omega_11[i, self.num_molecules + j])
                D_cd = (3.0 / 16.0) * kT / (n * coll_mass * self.omega_11[i, self.num_molecules + j])
                rm = 32.0 * n * self.omega_22[i, self.num_molecules + j] / (5 * constants.K_CONST_PI)
                for k, n_i in enumerate(self.this_n_vl_mol[i]):
                    n_ij = n_i * self.this_n_atom[j]
                    p1 = 3 * (self.vl_offset[i] + k)
                    o1 = self.vl_offset[i] + k
                    idx_atom = self.n_vibr_levels_total * 3 + 2 * j
                    self.thermal_conductivity_LHS[p1, idx_atom] = -1.5 * kT * n_ij / (n * D_cd)
                    self.thermal_conductivity_LHS[p1 + 1, idx_atom] = (
                        0.75
                        * kT
                        * n_ij
                        * (6 * C_cd - 5)
                        * self.atoms[j].mass
                        / ((self.molecules[i].mass + self.atoms[j].mass) * n * D_cd)
                    )
                    self.thermal_conductivity_LHS[p1 + 2, idx_atom] = 0.0
                    self.thermal_conductivity_LHS[p1, idx_atom + 1] = (
                        0.75
                        * kT
                        * n_ij
                        * (6 * C_cd - 5)
                        * self.molecules[i].mass
                        / ((self.molecules[i].mass + self.atoms[j].mass) * n * D_cd)
                    )
                    self.thermal_conductivity_LHS[p1 + 1, idx_atom + 1] = (
                        -1.5
                        * kT
                        * n_ij
                        * coll_mass
                        / (n * D_cd * (self.molecules[i].mass + self.atoms[j].mass))
                        * (
                            13.75
                            - 3 * B_cd
                            - 4 * A_cd
                            - (20.0 / 3.0)
                            * A_cd
                            / (constants.K_CONST_K * constants.K_CONST_PI)
                            * (self.molecules[i].mass * self.c_rot_arr[o1] / (rm * self.rot_rel_times[i, self.num_molecules + j]))
                        )
                    )
                    self.thermal_conductivity_LHS[p1 + 2, idx_atom + 1] = (
                        -6
                        * temperature
                        / (n * D_cd * constants.K_CONST_PI)
                        * A_cd
                        * n_ij
                        * self.molecules[i].mass
                        * self.molecules[i].mass
                        * self.c_rot_arr[o1]
                        / (rm * self.rot_rel_times[i, self.num_molecules + j] * (self.molecules[i].mass + self.atoms[j].mass))
                    )
                    self.thermal_conductivity_LHS[idx_atom, p1] = self.thermal_conductivity_LHS[p1, idx_atom]
                    self.thermal_conductivity_LHS[idx_atom, p1 + 1] = self.thermal_conductivity_LHS[p1 + 1, idx_atom]
                    self.thermal_conductivity_LHS[idx_atom, p1 + 2] = self.thermal_conductivity_LHS[p1 + 2, idx_atom]
                    self.thermal_conductivity_LHS[idx_atom + 1, p1] = self.thermal_conductivity_LHS[p1, idx_atom + 1]
                    self.thermal_conductivity_LHS[idx_atom + 1, p1 + 1] = self.thermal_conductivity_LHS[p1 + 1, idx_atom + 1]
                    self.thermal_conductivity_LHS[idx_atom + 1, p1 + 2] = self.thermal_conductivity_LHS[p1 + 2, idx_atom + 1]

        for i in range(self.num_atoms):
            for j in range(i, self.num_atoms):
                coll_mass = self.interactions[self.inter_index(self.num_molecules + i, self.num_molecules + j)].collision_mass
                A_cd = 0.5 * self.omega_22[self.num_molecules + i, self.num_molecules + j] / self.omega_11[self.num_molecules + i, self.num_molecules + j]
                B_cd = (5 * self.omega_12[self.num_molecules + i, self.num_molecules + j] - self.omega_13[self.num_molecules + i, self.num_molecules + j]) / (
                    3 * self.omega_11[self.num_molecules + i, self.num_molecules + j]
                )
                C_cd = self.omega_12[self.num_molecules + i, self.num_molecules + j] / (3 * self.omega_11[self.num_molecules + i, self.num_molecules + j])
                D_cd = (3.0 / 16.0) * kT / (n * coll_mass * self.omega_11[self.num_molecules + i, self.num_molecules + j])
                if j > i:
                    idx_i = self.n_vibr_levels_total * 3 + 2 * i
                    idx_j = self.n_vibr_levels_total * 3 + 2 * j
                    n_ij = self.this_n_atom[i] * self.this_n_atom[j]
                    self.thermal_conductivity_LHS[idx_i, idx_j] = -1.5 * kT * n_ij / (n * D_cd)
                    self.thermal_conductivity_LHS[idx_i + 1, idx_j] = (
                        0.75
                        * kT
                        * n_ij
                        * (6 * C_cd - 5)
                        * self.atoms[j].mass
                        / ((self.atoms[i].mass + self.atoms[j].mass) * n * D_cd)
                    )
                    self.thermal_conductivity_LHS[idx_i, idx_j + 1] = (
                        0.75
                        * kT
                        * n_ij
                        * (6 * C_cd - 5)
                        * self.atoms[i].mass
                        / ((self.atoms[i].mass + self.atoms[j].mass) * n * D_cd)
                    )
                    self.thermal_conductivity_LHS[idx_i + 1, idx_j + 1] = (
                        -1.5
                        * kT
                        * n_ij
                        * coll_mass
                        / (n * D_cd * (self.atoms[i].mass + self.atoms[j].mass))
                        * (13.75 - 3 * B_cd - 4 * A_cd)
                    )
                    self.thermal_conductivity_LHS[idx_j, idx_i] = self.thermal_conductivity_LHS[idx_i, idx_j]
                    self.thermal_conductivity_LHS[idx_j, idx_i + 1] = self.thermal_conductivity_LHS[idx_i + 1, idx_j]
                    self.thermal_conductivity_LHS[idx_j + 1, idx_i] = self.thermal_conductivity_LHS[idx_i, idx_j + 1]
                    self.thermal_conductivity_LHS[idx_j + 1, idx_i + 1] = self.thermal_conductivity_LHS[idx_i + 1, idx_j + 1]

        for i in range(self.num_atoms):
            coll_mass = self.interactions[self.inter_index(self.num_molecules + i, self.num_molecules + i)].collision_mass
            A_cd = 0.5 * self.omega_22[self.num_molecules + i, self.num_molecules + i] / self.omega_11[self.num_molecules + i, self.num_molecules + i]
            C_cd = self.omega_12[self.num_molecules + i, self.num_molecules + i] / (3 * self.omega_11[self.num_molecules + i, self.num_molecules + i])
            D_cd = (3.0 / 8.0) * kT / (n * self.atoms[i].mass * self.omega_11[self.num_molecules + i, self.num_molecules + i])
            n_ij = self.this_n_atom[i] * self.this_n_atom[i]
            idx_i = self.n_vibr_levels_total * 3 + 2 * i
            self.thermal_conductivity_LHS[idx_i, idx_i] = 0.0
            self.thermal_conductivity_LHS[idx_i + 1, idx_i] = 0.0
            self.thermal_conductivity_LHS[idx_i, idx_i + 1] = 0.0
            self.thermal_conductivity_LHS[idx_i + 1, idx_i + 1] = 1.5 * kT * n_ij * 2 * A_cd / (n * D_cd)

            # atom i + molecule j collisions
            for j in range(self.num_molecules):
                coll_mass = self.interactions[self.inter_index(self.num_molecules + i, j)].collision_mass
                A_cd = 0.5 * self.omega_22[self.num_molecules + i, j] / self.omega_11[self.num_molecules + i, j]
                B_cd = (5 * self.omega_12[self.num_molecules + i, j] - self.omega_13[self.num_molecules + i, j]) / (
                    3 * self.omega_11[self.num_molecules + i, j]
                )
                C_cd = self.omega_12[self.num_molecules + i, j] / (3 * self.omega_11[self.num_molecules + i, j])
                D_cd = (3.0 / 16.0) * kT / (n * coll_mass * self.omega_11[self.num_molecules + i, j])
                rm = 32.0 * n * self.omega_22[self.num_molecules + i, j] / (5 * constants.K_CONST_PI)
                for k, n_k in enumerate(self.this_n_vl_mol[j]):
                    n_ij = self.this_n_atom[i] * n_k
                    # BugFix by Qizhen Hong: use molecule j vibrational offset
                    o1 = self.vl_offset[j] + k
                    self.thermal_conductivity_LHS[idx_i, idx_i] += 1.5 * kT * n_ij / (D_cd * n)
                    tmp = (
                        0.75
                        * kT
                        * n_ij
                        * self.molecules[j].mass
                        * (6 * C_cd - 5)
                        / ((self.atoms[i].mass + self.molecules[j].mass) * n * D_cd)
                    )
                    self.thermal_conductivity_LHS[idx_i + 1, idx_i] -= tmp
                    self.thermal_conductivity_LHS[idx_i, idx_i + 1] -= tmp
                    self.thermal_conductivity_LHS[idx_i + 1, idx_i + 1] += (
                        1.5
                        * kT
                        * coll_mass
                        * n_ij
                        * (
                            7.5 * self.atoms[i].mass / self.molecules[j].mass
                            + 6.25 * self.molecules[j].mass / self.atoms[i].mass
                            - 3 * self.molecules[j].mass * B_cd / self.atoms[i].mass
                            + 4 * A_cd
                            + (20.0 / 3.0)
                            * A_cd
                            * (self.molecules[j].mass * self.c_rot_arr[o1] / (rm * self.rot_rel_times[j, self.num_molecules + i]))
                            / (constants.K_CONST_PI * constants.K_CONST_K)
                        )
                        / (n * D_cd * (self.atoms[i].mass + self.molecules[j].mass))
                    )

            for j in range(self.num_atoms):
                coll_mass = self.interactions[self.inter_index(self.num_molecules + i, self.num_molecules + j)].collision_mass
                A_cd = 0.5 * self.omega_22[self.num_molecules + i, self.num_molecules + j] / self.omega_11[self.num_molecules + i, self.num_molecules + j]
                B_cd = (5 * self.omega_12[self.num_molecules + i, self.num_molecules + j] - self.omega_13[self.num_molecules + i, self.num_molecules + j]) / (
                    3 * self.omega_11[self.num_molecules + i, self.num_molecules + j]
                )
                C_cd = self.omega_12[self.num_molecules + i, self.num_molecules + j] / (3 * self.omega_11[self.num_molecules + i, self.num_molecules + j])
                D_cd = (3.0 / 16.0) * kT / (n * coll_mass * self.omega_11[self.num_molecules + i, self.num_molecules + j])
                if j != i:
                    n_ij = self.this_n_atom[i] * self.this_n_atom[j]
                    self.thermal_conductivity_LHS[idx_i, idx_i] += 1.5 * kT * n_ij / (D_cd * n)
                    tmp = (
                        0.75
                        * kT
                        * n_ij
                        * self.atoms[j].mass
                        * (6 * C_cd - 5)
                        / ((self.atoms[i].mass + self.atoms[j].mass) * n * D_cd)
                    )
                    self.thermal_conductivity_LHS[idx_i + 1, idx_i] -= tmp
                    self.thermal_conductivity_LHS[idx_i, idx_i + 1] -= tmp
                    self.thermal_conductivity_LHS[idx_i + 1, idx_i + 1] += (
                        1.5
                        * kT
                        * coll_mass
                        * n_ij
                        * (
                            7.5 * self.atoms[i].mass / self.atoms[j].mass
                            + 6.25 * self.atoms[j].mass / self.atoms[i].mass
                            - 3 * self.atoms[j].mass * B_cd / self.atoms[i].mass
                            + 4 * A_cd
                        )
                        / (n * D_cd * (self.atoms[i].mass + self.atoms[j].mass))
                    )

        self.thermal_conductivity_LHS /= n * n
        for i in range(self.num_molecules):
            for k, n_i in enumerate(self.this_n_vl_mol[i]):
                p1 = 3 * (self.vl_offset[i] + k)
                self.thermal_conductivity_LHS[0, p1] = self.molecules[i].mass * n_i / (n * 1e24)
                self.thermal_conductivity_LHS[0, p1 + 1] = 0.0
                self.thermal_conductivity_LHS[0, p1 + 2] = 0.0
        for i in range(self.num_atoms):
            idx_i = self.n_vibr_levels_total * 3 + 2 * i
            self.thermal_conductivity_LHS[0, idx_i] = self.atoms[i].mass * self.this_n_atom[i] / (n * 1e24)
            self.thermal_conductivity_LHS[0, idx_i + 1] = 0.0

    def compute_thermal_conductivity_RHS(self, temperature: float) -> None:
        self.thermal_conductivity_RHS.fill(0.0)
        j = 0
        for i in range(self.num_molecules):
            for k, n_i in enumerate(self.this_n_vl_mol[i]):
                self.thermal_conductivity_RHS[3 * j] = 0.0
                self.thermal_conductivity_RHS[3 * j + 1] = 7.5 * constants.K_CONST_K * n_i
                self.thermal_conductivity_RHS[3 * j + 2] = 3.0 * self.molecules[i].mass * n_i * self.c_rot_arr[j]
                j += 1
        for i, n_i in enumerate(self.this_n_atom):
            base = self.n_vibr_levels_total * 3 + 2 * i
            self.thermal_conductivity_RHS[base] = 0.0
            self.thermal_conductivity_RHS[base + 1] = 7.5 * constants.K_CONST_K * n_i
        self.thermal_conductivity_RHS *= temperature / self.this_total_n

    def compute_thermal_conductivity_coeffs(self, temperature: float, model: ModelsOmega = ModelsOmega.ESA) -> None:
        self.compute_thermal_conductivity_LHS(temperature, model)
        self.compute_thermal_conductivity_RHS(temperature)
        lhs = self.thermal_conductivity_LHS * 1e40
        rhs = self.thermal_conductivity_RHS * 1e20
        self.thermal_conductivity_coeffs = self._solve_linear_system(lhs, rhs) * 1e20

    def compute_thermal_conductivity_rigid_rot_LHS(self, temperature: float, model: ModelsOmega = ModelsOmega.ESA) -> None:
        self.thermal_conductivity_rigid_rot_LHS.fill(0.0)
        n = self.this_total_n
        kT = constants.K_CONST_K * temperature

        for i in range(self.num_molecules):
            for j in range(i, self.num_molecules):
                coll_mass = self.interactions[self.inter_index(i, j)].collision_mass
                A_cd = 0.5 * self.omega_22[i, j] / self.omega_11[i, j]
                B_cd = (5 * self.omega_12[i, j] - self.omega_13[i, j]) / (3 * self.omega_11[i, j])
                C_cd = self.omega_12[i, j] / (3 * self.omega_11[i, j])
                D_cd = (3.0 / 16.0) * kT / (n * coll_mass * self.omega_11[i, j])
                rm = 32.0 * n * self.omega_22[i, j] / (5 * constants.K_CONST_PI)
                if j != i:
                    p1 = 3 * i
                    p2 = 3 * j
                    n_ij = self.this_n_molecules[i] * self.this_n_molecules[j]
                    self.thermal_conductivity_rigid_rot_LHS[p1, p2] = -1.5 * kT * n_ij / (n * D_cd)
                    self.thermal_conductivity_rigid_rot_LHS[p1 + 1, p2] = (
                        0.75
                        * kT
                        * n_ij
                        * (6 * C_cd - 5)
                        * self.molecules[j].mass
                        / ((self.molecules[i].mass + self.molecules[j].mass) * n * D_cd)
                    )
                    self.thermal_conductivity_rigid_rot_LHS[p1 + 2, p2] = 0.0
                    self.thermal_conductivity_rigid_rot_LHS[p1, p2 + 1] = (
                        0.75
                        * kT
                        * n_ij
                        * (6 * C_cd - 5)
                        * self.molecules[i].mass
                        / ((self.molecules[i].mass + self.molecules[j].mass) * n * D_cd)
                    )
                    self.thermal_conductivity_rigid_rot_LHS[p1 + 1, p2 + 1] = (
                        -1.5
                        * kT
                        * n_ij
                        * coll_mass
                        / (n * D_cd * (self.molecules[i].mass + self.molecules[j].mass))
                        * (
                            13.75
                            - 3 * B_cd
                            - 4 * A_cd
                            - (20.0 / 3.0)
                            * A_cd
                            / (constants.K_CONST_K * constants.K_CONST_PI)
                            * (
                                self.molecules[i].mass * self.c_rot_rigid_rot_arr[i] / (rm * self.rot_rel_times[i, j])
                                + self.molecules[j].mass * self.c_rot_rigid_rot_arr[j] / (rm * self.rot_rel_times[j, i])
                            )
                        )
                    )
                    rot_tmp = (
                        -6
                        * temperature
                        / (n * D_cd * constants.K_CONST_PI)
                        * A_cd
                        * n_ij
                        * self.molecules[i].mass
                        * self.molecules[i].mass
                        * self.c_rot_rigid_rot_arr[i]
                        / (rm * self.rot_rel_times[i, j] * (self.molecules[i].mass + self.molecules[j].mass))
                    )
                    self.thermal_conductivity_rigid_rot_LHS[p1 + 2, p2 + 1] = rot_tmp
                    self.thermal_conductivity_rigid_rot_LHS[p1, p2 + 2] = 0.0
                    rot_tmp_2 = (
                        -6
                        * temperature
                        / (n * D_cd * constants.K_CONST_PI)
                        * A_cd
                        * n_ij
                        * self.molecules[j].mass
                        * self.molecules[j].mass
                        * self.c_rot_rigid_rot_arr[j]
                        / (rm * self.rot_rel_times[j, i] * (self.molecules[i].mass + self.molecules[j].mass))
                    )
                    self.thermal_conductivity_rigid_rot_LHS[p1 + 1, p2 + 2] = rot_tmp_2
                    self.thermal_conductivity_rigid_rot_LHS[p1 + 2, p2 + 2] = 0.0
                    self.thermal_conductivity_rigid_rot_LHS[p2, p1] = self.thermal_conductivity_rigid_rot_LHS[p1, p2]
                    self.thermal_conductivity_rigid_rot_LHS[p2, p1 + 1] = self.thermal_conductivity_rigid_rot_LHS[p1 + 1, p2]
                    self.thermal_conductivity_rigid_rot_LHS[p2, p1 + 2] = self.thermal_conductivity_rigid_rot_LHS[p1 + 2, p2]
                    self.thermal_conductivity_rigid_rot_LHS[p2 + 1, p1] = self.thermal_conductivity_rigid_rot_LHS[p1, p2 + 1]
                    self.thermal_conductivity_rigid_rot_LHS[p2 + 1, p1 + 1] = self.thermal_conductivity_rigid_rot_LHS[p1 + 1, p2 + 1]
                    self.thermal_conductivity_rigid_rot_LHS[p2 + 1, p1 + 2] = self.thermal_conductivity_rigid_rot_LHS[p1 + 2, p2 + 1]
                    self.thermal_conductivity_rigid_rot_LHS[p2 + 2, p1] = self.thermal_conductivity_rigid_rot_LHS[p1, p2 + 2]
                    self.thermal_conductivity_rigid_rot_LHS[p2 + 2, p1 + 1] = self.thermal_conductivity_rigid_rot_LHS[p1 + 1, p2 + 2]
                    self.thermal_conductivity_rigid_rot_LHS[p2 + 2, p1 + 2] = self.thermal_conductivity_rigid_rot_LHS[p1 + 2, p2 + 2]

        for i in range(self.num_molecules):
            coll_mass = self.interactions[self.inter_index(i, i)].collision_mass
            A_cd = 0.5 * self.omega_22[i, i] / self.omega_11[i, i]
            B_cd = (5 * self.omega_12[i, i] - self.omega_13[i, i]) / (3 * self.omega_11[i, i])
            C_cd = self.omega_12[i, i] / (3 * self.omega_11[i, i])
            D_cd = (3.0 / 8.0) * kT / (n * self.molecules[i].mass * self.omega_11[i, i])
            rm = 32.0 * n * self.omega_22[i, i] / (5 * constants.K_CONST_PI)
            n_ij = self.this_n_molecules[i] * self.this_n_molecules[i]
            p1 = 3 * i
            self.thermal_conductivity_rigid_rot_LHS[p1, p1] = 0.0
            self.thermal_conductivity_rigid_rot_LHS[p1 + 1, p1] = 0.0
            self.thermal_conductivity_rigid_rot_LHS[p1 + 2, p1] = 0.0
            self.thermal_conductivity_rigid_rot_LHS[p1, p1 + 1] = 0.0
            self.thermal_conductivity_rigid_rot_LHS[p1 + 1, p1 + 1] = (
                1.5
                * kT
                * n_ij
                * A_cd
                * (2 + (20.0 / 3.0) * self.molecules[i].mass * self.c_rot_rigid_rot_arr[i] / (constants.K_CONST_K * constants.K_CONST_PI * rm * self.rot_rel_times[i, i]))
                / (n * D_cd)
            )
            tmp = -6.0 * temperature * A_cd * n_ij * self.molecules[i].mass * self.c_rot_rigid_rot_arr[i] / (
                constants.K_CONST_PI * rm * self.rot_rel_times[i, i] * n * D_cd
            )
            self.thermal_conductivity_rigid_rot_LHS[p1 + 2, p1 + 1] = tmp
            self.thermal_conductivity_rigid_rot_LHS[p1, p1 + 2] = 0.0
            self.thermal_conductivity_rigid_rot_LHS[p1 + 1, p1 + 2] = tmp
            self.thermal_conductivity_rigid_rot_LHS[p1 + 2, p1 + 2] = (
                temperature
                * n_ij
                * (self.molecules[i].mass * self.c_rot_rigid_rot_arr[i] / n)
                * (1.5 / D_cd + 3.6 * A_cd / (constants.K_CONST_PI * D_cd * rm * self.rot_rel_times[i, i]))
            )

            for j in range(self.num_molecules):
                coll_mass = self.interactions[self.inter_index(i, j)].collision_mass
                A_cd = 0.5 * self.omega_22[i, j] / self.omega_11[i, j]
                B_cd = (5 * self.omega_12[i, j] - self.omega_13[i, j]) / (3 * self.omega_11[i, j])
                C_cd = self.omega_12[i, j] / (3 * self.omega_11[i, j])
                D_cd = (3.0 / 16.0) * kT / (n * coll_mass * self.omega_11[i, j])
                rm = 32.0 * n * self.omega_22[i, j] / (5 * constants.K_CONST_PI)
                if i != j:
                    n_ij = self.this_n_molecules[i] * self.this_n_molecules[j]
                    self.thermal_conductivity_rigid_rot_LHS[p1, p1] += 1.5 * kT * n_ij / (D_cd * n)
                    tmp = (
                        0.75
                        * kT
                        * n_ij
                        * self.molecules[j].mass
                        * (6 * C_cd - 5)
                        / ((self.molecules[i].mass + self.molecules[j].mass) * n * D_cd)
                    )
                    self.thermal_conductivity_rigid_rot_LHS[p1 + 1, p1] -= tmp
                    self.thermal_conductivity_rigid_rot_LHS[p1, p1 + 1] -= tmp
                    self.thermal_conductivity_rigid_rot_LHS[p1 + 1, p1 + 1] += (
                        1.5
                        * kT
                        * coll_mass
                        * n_ij
                        * (
                            7.5 * self.molecules[i].mass / self.molecules[j].mass
                            + 6.25 * self.molecules[j].mass / self.molecules[i].mass
                            - 3 * self.molecules[j].mass * B_cd / self.molecules[i].mass
                            + 4 * A_cd
                            + (20.0 / 3.0)
                            * A_cd
                            * (self.molecules[i].mass * self.c_rot_rigid_rot_arr[i] / (rm * self.rot_rel_times[i, j]))
                            / (constants.K_CONST_PI * constants.K_CONST_K)
                        )
                        / (n * D_cd * (self.molecules[i].mass + self.molecules[j].mass))
                    )
                    tmp_rot = (
                        -6.0
                        * temperature
                        * A_cd
                        * n_ij
                        * self.molecules[i].mass
                        * self.c_rot_rigid_rot_arr[i]
                        / (rm * self.rot_rel_times[i, j])
                        * self.molecules[i].mass
                        / (n * D_cd * constants.K_CONST_PI * (self.molecules[i].mass + self.molecules[j].mass))
                    )
                    self.thermal_conductivity_rigid_rot_LHS[p1 + 2, p1 + 1] += tmp_rot
                    self.thermal_conductivity_rigid_rot_LHS[p1 + 1, p1 + 2] += tmp_rot
                    self.thermal_conductivity_rigid_rot_LHS[p1 + 2, p1 + 2] += (
                        temperature
                        * n_ij
                        * self.molecules[i].mass
                        * self.c_rot_rigid_rot_arr[i]
                        * (
                            1.5 / D_cd
                            + 3.6
                            * A_cd
                            * self.molecules[i].mass
                            / (constants.K_CONST_PI * D_cd * self.molecules[j].mass * rm * self.rot_rel_times[i, j])
                        )
                        / n
                    )

            for j in range(self.num_atoms):
                coll_mass = self.interactions[self.inter_index(i, self.num_molecules + j)].collision_mass
                A_cd = 0.5 * self.omega_22[i, self.num_molecules + j] / self.omega_11[i, self.num_molecules + j]
                B_cd = (5 * self.omega_12[i, self.num_molecules + j] - self.omega_13[i, self.num_molecules + j]) / (
                    3 * self.omega_11[i, self.num_molecules + j]
                )
                C_cd = self.omega_12[i, self.num_molecules + j] / (3 * self.omega_11[i, self.num_molecules + j])
                D_cd = (3.0 / 16.0) * kT / (n * coll_mass * self.omega_11[i, self.num_molecules + j])
                rm = 32.0 * n * self.omega_22[i, self.num_molecules + j] / (5 * constants.K_CONST_PI)
                n_ij = self.this_n_molecules[i] * self.this_n_atom[j]
                self.thermal_conductivity_rigid_rot_LHS[p1, p1] += 1.5 * kT * n_ij / (D_cd * n)
                tmp = (
                    0.75
                    * kT
                    * n_ij
                    * self.atoms[j].mass
                    * (6 * C_cd - 5)
                    / ((self.molecules[i].mass + self.atoms[j].mass) * n * D_cd)
                )
                self.thermal_conductivity_rigid_rot_LHS[p1 + 1, p1] -= tmp
                self.thermal_conductivity_rigid_rot_LHS[p1, p1 + 1] -= tmp
                self.thermal_conductivity_rigid_rot_LHS[p1 + 1, p1 + 1] += (
                    1.5
                    * kT
                    * coll_mass
                    * n_ij
                    * (
                        7.5 * self.molecules[i].mass / self.atoms[j].mass
                        + 6.25 * self.atoms[j].mass / self.molecules[i].mass
                        - 3 * self.atoms[j].mass * B_cd / self.molecules[i].mass
                        + 4 * A_cd
                        + (20.0 / 3.0)
                        * A_cd
                        * (self.molecules[i].mass * self.c_rot_rigid_rot_arr[i] / (rm * self.rot_rel_times[i, self.num_molecules + j]))
                        / (constants.K_CONST_PI * constants.K_CONST_K)
                    )
                    / (n * D_cd * (self.molecules[i].mass + self.atoms[j].mass))
                )
                tmp_rot = (
                    -6.0
                    * temperature
                    * A_cd
                    * n_ij
                    * self.molecules[i].mass
                    * self.c_rot_rigid_rot_arr[i]
                    / (rm * self.rot_rel_times[i, self.num_molecules + j])
                    * self.molecules[i].mass
                    / (n * D_cd * constants.K_CONST_PI * (self.molecules[i].mass + self.atoms[j].mass))
                )
                self.thermal_conductivity_rigid_rot_LHS[p1 + 2, p1 + 1] += tmp_rot
                self.thermal_conductivity_rigid_rot_LHS[p1 + 1, p1 + 2] += tmp_rot
                self.thermal_conductivity_rigid_rot_LHS[p1 + 2, p1 + 2] += (
                    temperature
                    * n_ij
                    * self.molecules[i].mass
                    * self.c_rot_rigid_rot_arr[i]
                    * (
                        1.5 / D_cd
                        + 3.6
                        * A_cd
                        * self.molecules[i].mass
                        / (constants.K_CONST_PI * D_cd * self.atoms[j].mass * rm * self.rot_rel_times[i, self.num_molecules + j])
                    )
                    / n
                )

        for i in range(self.num_molecules):
            for j in range(self.num_atoms):
                coll_mass = self.interactions[self.inter_index(i, self.num_molecules + j)].collision_mass
                A_cd = 0.5 * self.omega_22[i, self.num_molecules + j] / self.omega_11[i, self.num_molecules + j]
                B_cd = (5 * self.omega_12[i, self.num_molecules + j] - self.omega_13[i, self.num_molecules + j]) / (
                    3 * self.omega_11[i, self.num_molecules + j]
                )
                C_cd = self.omega_12[i, self.num_molecules + j] / (3 * self.omega_11[i, self.num_molecules + j])
                D_cd = (3.0 / 16.0) * constants.K_CONST_K * temperature / (
                    n * coll_mass * self.omega_11[i, self.num_molecules + j]
                )
                rm = 32.0 * n * self.omega_22[i, self.num_molecules + j] / (5 * constants.K_CONST_PI)
                n_ij = self.this_n_molecules[i] * self.this_n_atom[j]
                idx_i = 3 * i
                idx_j = self.num_molecules * 3 + 2 * j
                self.thermal_conductivity_rigid_rot_LHS[idx_i, idx_j] = -1.5 * constants.K_CONST_K * temperature * n_ij / (n * D_cd)
                self.thermal_conductivity_rigid_rot_LHS[idx_i + 1, idx_j] = (
                    0.75
                    * constants.K_CONST_K
                    * temperature
                    * n_ij
                    * (6 * C_cd - 5)
                    * self.atoms[j].mass
                    / ((self.molecules[i].mass + self.atoms[j].mass) * n * D_cd)
                )
                self.thermal_conductivity_rigid_rot_LHS[idx_i + 2, idx_j] = 0.0
                self.thermal_conductivity_rigid_rot_LHS[idx_i, idx_j + 1] = (
                    0.75
                    * constants.K_CONST_K
                    * temperature
                    * n_ij
                    * (6 * C_cd - 5)
                    * self.molecules[i].mass
                    / ((self.molecules[i].mass + self.atoms[j].mass) * n * D_cd)
                )
                self.thermal_conductivity_rigid_rot_LHS[idx_i + 1, idx_j + 1] = (
                    -1.5
                    * constants.K_CONST_K
                    * temperature
                    * n_ij
                    * coll_mass
                    / (n * D_cd * (self.molecules[i].mass + self.atoms[j].mass))
                    * (
                        13.75
                        - 3 * B_cd
                        - 4 * A_cd
                        - (20.0 / 3.0)
                        * A_cd
                        / (constants.K_CONST_K * constants.K_CONST_PI)
                        * (self.molecules[i].mass * self.c_rot_rigid_rot_arr[i] / (rm * self.rot_rel_times[i, self.num_molecules + j]))
                    )
                )
                tmp = (
                    -6
                    * temperature
                    / (n * D_cd * constants.K_CONST_PI)
                    * A_cd
                    * n_ij
                    * self.molecules[i].mass
                    * self.molecules[i].mass
                    * self.c_rot_rigid_rot_arr[i]
                    / (rm * self.rot_rel_times[i, self.num_molecules + j] * (self.molecules[i].mass + self.atoms[j].mass))
                )
                self.thermal_conductivity_rigid_rot_LHS[idx_i + 2, idx_j + 1] = tmp
                self.thermal_conductivity_rigid_rot_LHS[idx_i, idx_j + 2] = 0.0
                self.thermal_conductivity_rigid_rot_LHS[idx_i + 1, idx_j + 2] = tmp
                self.thermal_conductivity_rigid_rot_LHS[idx_i + 2, idx_j + 2] = 0.0
                self.thermal_conductivity_rigid_rot_LHS[idx_j, idx_i] = self.thermal_conductivity_rigid_rot_LHS[idx_i, idx_j]
                self.thermal_conductivity_rigid_rot_LHS[idx_j, idx_i + 1] = self.thermal_conductivity_rigid_rot_LHS[idx_i + 1, idx_j]
                self.thermal_conductivity_rigid_rot_LHS[idx_j, idx_i + 2] = self.thermal_conductivity_rigid_rot_LHS[idx_i + 2, idx_j]
                self.thermal_conductivity_rigid_rot_LHS[idx_j + 1, idx_i] = self.thermal_conductivity_rigid_rot_LHS[idx_i, idx_j + 1]
                self.thermal_conductivity_rigid_rot_LHS[idx_j + 1, idx_i + 1] = self.thermal_conductivity_rigid_rot_LHS[idx_i + 1, idx_j + 1]
                self.thermal_conductivity_rigid_rot_LHS[idx_j + 1, idx_i + 2] = self.thermal_conductivity_rigid_rot_LHS[idx_i + 2, idx_j + 1]

        for i in range(self.num_atoms):
            for j in range(i, self.num_atoms):
                coll_mass = self.interactions[self.inter_index(self.num_molecules + i, self.num_molecules + j)].collision_mass
                A_cd = 0.5 * self.omega_22[self.num_molecules + i, self.num_molecules + j] / self.omega_11[self.num_molecules + i, self.num_molecules + j]
                B_cd = (5 * self.omega_12[self.num_molecules + i, self.num_molecules + j] - self.omega_13[self.num_molecules + i, self.num_molecules + j]) / (
                    3 * self.omega_11[self.num_molecules + i, self.num_molecules + j]
                )
                C_cd = self.omega_12[self.num_molecules + i, self.num_molecules + j] / (3 * self.omega_11[self.num_molecules + i, self.num_molecules + j])
                D_cd = (3.0 / 16.0) * constants.K_CONST_K * temperature / (
                    n * coll_mass * self.omega_11[self.num_molecules + i, self.num_molecules + j]
                )
                if j != i:
                    idx_i = self.num_molecules * 3 + 2 * i
                    idx_j = self.num_molecules * 3 + 2 * j
                    n_ij = self.this_n_atom[i] * self.this_n_atom[j]
                    self.thermal_conductivity_rigid_rot_LHS[idx_i, idx_j] = -1.5 * constants.K_CONST_K * temperature * n_ij / (n * D_cd)
                    self.thermal_conductivity_rigid_rot_LHS[idx_i + 1, idx_j] = (
                        0.75
                        * constants.K_CONST_K
                        * temperature
                        * n_ij
                        * (6 * C_cd - 5)
                        * self.atoms[j].mass
                        / ((self.atoms[i].mass + self.atoms[j].mass) * n * D_cd)
                    )
                    self.thermal_conductivity_rigid_rot_LHS[idx_i, idx_j + 1] = (
                        0.75
                        * constants.K_CONST_K
                        * temperature
                        * n_ij
                        * (6 * C_cd - 5)
                        * self.atoms[i].mass
                        / ((self.atoms[i].mass + self.atoms[j].mass) * n * D_cd)
                    )
                    self.thermal_conductivity_rigid_rot_LHS[idx_i + 1, idx_j + 1] = (
                        -1.5
                        * constants.K_CONST_K
                        * temperature
                        * n_ij
                        * coll_mass
                        / (n * D_cd * (self.atoms[i].mass + self.atoms[j].mass))
                        * (13.75 - 3 * B_cd - 4 * A_cd)
                    )
                    self.thermal_conductivity_rigid_rot_LHS[idx_j, idx_i] = self.thermal_conductivity_rigid_rot_LHS[idx_i, idx_j]
                    self.thermal_conductivity_rigid_rot_LHS[idx_j, idx_i + 1] = self.thermal_conductivity_rigid_rot_LHS[idx_i + 1, idx_j]
                    self.thermal_conductivity_rigid_rot_LHS[idx_j + 1, idx_i] = self.thermal_conductivity_rigid_rot_LHS[idx_i, idx_j + 1]
                    self.thermal_conductivity_rigid_rot_LHS[idx_j + 1, idx_i + 1] = self.thermal_conductivity_rigid_rot_LHS[idx_i + 1, idx_j + 1]

        for i in range(self.num_atoms):
            coll_mass = self.interactions[self.inter_index(self.num_molecules + i, self.num_molecules + i)].collision_mass
            A_cd = 0.5 * self.omega_22[self.num_molecules + i, self.num_molecules + i] / self.omega_11[self.num_molecules + i, self.num_molecules + i]
            C_cd = self.omega_12[self.num_molecules + i, self.num_molecules + i] / (3 * self.omega_11[self.num_molecules + i, self.num_molecules + i])
            D_cd = (3.0 / 8.0) * constants.K_CONST_K * temperature / (n * self.atoms[i].mass * self.omega_11[self.num_molecules + i, self.num_molecules + i])
            n_ij = self.this_n_atom[i] * self.this_n_atom[i]
            idx_i = self.num_molecules * 3 + 2 * i
            self.thermal_conductivity_rigid_rot_LHS[idx_i, idx_i] = 0.0
            self.thermal_conductivity_rigid_rot_LHS[idx_i + 1, idx_i] = 0.0
            self.thermal_conductivity_rigid_rot_LHS[idx_i, idx_i + 1] = 0.0
            self.thermal_conductivity_rigid_rot_LHS[idx_i + 1, idx_i + 1] = 1.5 * constants.K_CONST_K * temperature * n_ij * 2 * A_cd / (n * D_cd)

            for j in range(self.num_atoms):
                if j == i:
                    continue
                coll_mass = self.interactions[self.inter_index(self.num_molecules + i, self.num_molecules + j)].collision_mass
                A_cd = 0.5 * self.omega_22[self.num_molecules + i, self.num_molecules + j] / self.omega_11[self.num_molecules + i, self.num_molecules + j]
                B_cd = (5 * self.omega_12[self.num_molecules + i, self.num_molecules + j] - self.omega_13[self.num_molecules + i, self.num_molecules + j]) / (
                    3 * self.omega_11[self.num_molecules + i, self.num_molecules + j]
                )
                C_cd = self.omega_12[self.num_molecules + i, self.num_molecules + j] / (3 * self.omega_11[self.num_molecules + i, self.num_molecules + j])
                D_cd = (3.0 / 16.0) * constants.K_CONST_K * temperature / (
                    n * coll_mass * self.omega_11[self.num_molecules + i, self.num_molecules + j]
                )
                n_ij = self.this_n_atom[i] * self.this_n_atom[j]
                self.thermal_conductivity_rigid_rot_LHS[idx_i, idx_i] += 1.5 * constants.K_CONST_K * temperature * n_ij / (D_cd * n)
                tmp = (
                    0.75
                    * constants.K_CONST_K
                    * temperature
                    * n_ij
                    * self.atoms[j].mass
                    * (6 * C_cd - 5)
                    / ((self.atoms[i].mass + self.atoms[j].mass) * n * D_cd)
                )
                self.thermal_conductivity_rigid_rot_LHS[idx_i + 1, idx_i] -= tmp
                self.thermal_conductivity_rigid_rot_LHS[idx_i, idx_i + 1] -= tmp
                self.thermal_conductivity_rigid_rot_LHS[idx_i + 1, idx_i + 1] += (
                    1.5
                    * constants.K_CONST_K
                    * temperature
                    * coll_mass
                    * n_ij
                    * (
                        7.5 * self.atoms[i].mass / self.atoms[j].mass
                        + 6.25 * self.atoms[j].mass / self.atoms[i].mass
                        - 3 * self.atoms[j].mass * B_cd / self.atoms[i].mass
                        + 4 * A_cd
                    )
                    / (n * D_cd * (self.atoms[i].mass + self.atoms[j].mass))
                )

        self.thermal_conductivity_rigid_rot_LHS /= n * n
        for i in range(self.num_molecules):
            p1 = 3 * i
            self.thermal_conductivity_rigid_rot_LHS[0, p1] = self.molecules[i].mass * self.this_n_molecules[i] / (self.this_total_n * 1e24)
            self.thermal_conductivity_rigid_rot_LHS[0, p1 + 1] = 0.0
            self.thermal_conductivity_rigid_rot_LHS[0, p1 + 2] = 0.0
        for i in range(self.num_atoms):
            idx = self.num_molecules * 3 + 2 * i
            self.thermal_conductivity_rigid_rot_LHS[0, idx] = self.atoms[i].mass * self.this_n_atom[i] / (self.this_total_n * 1e24)
            self.thermal_conductivity_rigid_rot_LHS[0, idx + 1] = 0.0

    def compute_thermal_conductivity_rigid_rot_RHS(self, temperature: float) -> None:
        self.thermal_conductivity_rigid_rot_RHS.fill(0.0)
        for i, n_i in enumerate(self.this_n_molecules):
            self.thermal_conductivity_rigid_rot_RHS[3 * i] = 0.0
            self.thermal_conductivity_rigid_rot_RHS[3 * i + 1] = 7.5 * constants.K_CONST_K * n_i
            self.thermal_conductivity_rigid_rot_RHS[3 * i + 2] = 3.0 * self.molecules[i].mass * n_i * self.c_rot_rigid_rot_arr[i]
        base = self.num_molecules * 3
        for i, n_i in enumerate(self.this_n_atom):
            self.thermal_conductivity_rigid_rot_RHS[base + 2 * i] = 0.0
            self.thermal_conductivity_rigid_rot_RHS[base + 2 * i + 1] = 7.5 * constants.K_CONST_K * n_i
        self.thermal_conductivity_rigid_rot_RHS *= temperature / self.this_total_n

    def compute_thermal_conductivity_rigid_rot_coeffs(self, temperature: float, model: ModelsOmega = ModelsOmega.ESA) -> None:
        self.compute_thermal_conductivity_rigid_rot_LHS(temperature, model)
        self.compute_thermal_conductivity_rigid_rot_RHS(temperature)
        lhs = self.thermal_conductivity_rigid_rot_LHS * 1e40
        rhs = self.thermal_conductivity_rigid_rot_RHS * 1e20
        self.thermal_conductivity_rigid_rot_coeffs = self._solve_linear_system(lhs, rhs) * 1e20

    def thermal_conductivity(self, temperature: float, model: ModelsOmega = ModelsOmega.ESA) -> float:
        res = 0.0
        if self.all_rigid_rotators:
            self.compute_thermal_conductivity_rigid_rot_coeffs(temperature, model)
            for i, n_i in enumerate(self.this_n_molecules):
                res += n_i * (
                    1.25 * constants.K_CONST_K * self.thermal_conductivity_rigid_rot_coeffs[3 * i + 1]
                    + 0.5 * self.thermal_conductivity_rigid_rot_coeffs[3 * i + 2]
                )
            base = self.num_molecules * 3
            for i, n_i in enumerate(self.this_n_atom):
                res += n_i * 1.25 * constants.K_CONST_K * self.thermal_conductivity_rigid_rot_coeffs[base + 2 * i + 1]
        else:
            self.compute_thermal_conductivity_coeffs(temperature, model)
            j = 0
            for i, n_levels in enumerate(self.this_n_vl_mol):
                for n_i in n_levels:
                    res += n_i * (
                        1.25 * constants.K_CONST_K * self.thermal_conductivity_coeffs[3 * j + 1]
                        + 0.5 * self.thermal_conductivity_coeffs[3 * j + 2] * self.c_rot_arr[j] * self.molecules[i].mass
                    )
                    j += 1
            base = self.n_vibr_levels_total * 3
            for i, n_i in enumerate(self.this_n_atom):
                res += n_i * 1.25 * constants.K_CONST_K * self.thermal_conductivity_coeffs[base + 2 * i + 1]
        return res / self.this_total_n

    def thermodiffusion(self, temperature: float, model: ModelsOmega = ModelsOmega.ESA) -> None:
        self.th_diff.fill(0.0)
        j = 0
        if self.all_rigid_rotators:
            for i in range(self.num_molecules):
                for _ in range(self.molecules[i].num_vibr_levels[0]):
                    self.th_diff[j] = self.thermal_conductivity_rigid_rot_coeffs[3 * i]
                    j += 1
            for i in range(self.num_atoms):
                self.th_diff[j] = self.thermal_conductivity_rigid_rot_coeffs[self.num_molecules * 3 + 2 * i]
                j += 1
        else:
            for i in range(self.num_molecules):
                for _ in range(self.molecules[i].num_vibr_levels[0]):
                    self.th_diff[j] = self.thermal_conductivity_coeffs[3 * j]
                    j += 1
            for i in range(self.num_atoms):
                self.th_diff[j] = self.thermal_conductivity_coeffs[self.n_vibr_levels_total * 3 + 2 * i]
                j += 1
        self.th_diff = -0.5 * self.th_diff / self.this_total_n

    # ---------------- Shear viscosity ---------------- #
    def compute_shear_viscosity_LHS(self, temperature: float, model: ModelsOmega = ModelsOmega.ESA) -> None:
        self.shear_viscosity_LHS.fill(0.0)
        kT = constants.K_CONST_K * temperature
        n = self.this_total_n

        for i1 in range(self.num_molecules - 1):
            for i2 in range(i1 + 1, self.num_molecules):
                tmp = (
                    3.2
                    * (self.this_n_molecules[i1] / n)
                    * (self.this_n_molecules[i2] / n)
                    * (self.molecules[i1].mass * self.molecules[i2].mass)
                    * self.omega_22[i1, i2]
                    / (kT * (self.molecules[i1].mass + self.molecules[i2].mass) ** 2)
                )
                tmp *= 1 - 10 * self.omega_11[i1, i2] / (3.0 * self.omega_22[i1, i2])
                self.shear_viscosity_LHS[i1, i2] = tmp
                self.shear_viscosity_LHS[i2, i1] = tmp

        for i1 in range(self.num_molecules):
            tmp = 1.6 * (self.this_n_molecules[i1] / n) * (self.this_n_molecules[i1] / n) * self.omega_22[i1, i1] / kT
            for i2 in range(self.num_molecules):
                if i2 == i1:
                    continue
                tmp += (
                    3.2
                    * (self.this_n_molecules[i1] / n)
                    * (self.this_n_molecules[i2] / n)
                    * self.omega_22[i1, i2]
                    / kT
                    * (
                        10.0
                        * self.omega_11[i1, i2]
                        * self.molecules[i2].mass
                        * self.molecules[i1].mass
                        / ((self.molecules[i2].mass + self.molecules[i1].mass) ** 2 * 3.0 * self.omega_22[i1, i2])
                        + self.molecules[i2].mass**2 / (self.molecules[i2].mass + self.molecules[i1].mass) ** 2
                    )
                )
            for i2 in range(self.num_atoms):
                tmp += (
                    3.2
                    * (self.this_n_molecules[i1] / n)
                    * (self.this_n_atom[i2] / n)
                    * self.omega_22[i1, self.num_molecules + i2]
                    / kT
                    * (
                        10.0
                        * self.omega_11[i1, self.num_molecules + i2]
                        / (3.0 * self.omega_22[i1, self.num_molecules + i2])
                        * self.atoms[i2].mass
                        * self.molecules[i1].mass
                        / (self.atoms[i2].mass + self.molecules[i1].mass) ** 2
                        + self.atoms[i2].mass**2 / (self.atoms[i2].mass + self.molecules[i1].mass) ** 2
                    )
                )
            self.shear_viscosity_LHS[i1, i1] = tmp

        for i1 in range(self.num_molecules):
            for i2 in range(self.num_atoms):
                tmp = (
                    3.2
                    * (self.this_n_molecules[i1] / n)
                    * (self.this_n_atom[i2] / n)
                    * (self.molecules[i1].mass * self.atoms[i2].mass)
                    * self.omega_22[i1, self.num_molecules + i2]
                    / (kT * (self.molecules[i1].mass + self.atoms[i2].mass) ** 2)
                )
                tmp *= 1.0 - 10.0 * self.omega_11[i1, self.num_molecules + i2] / (3.0 * self.omega_22[i1, self.num_molecules + i2])
                self.shear_viscosity_LHS[i1, self.num_molecules + i2] = tmp
                self.shear_viscosity_LHS[self.num_molecules + i2, i1] = tmp

        for i1 in range(self.num_atoms - 1):
            for i2 in range(i1 + 1, self.num_atoms):
                tmp = (
                    3.2
                    * (self.this_n_atom[i1] / n)
                    * (self.this_n_atom[i2] / n)
                    * (self.atoms[i1].mass * self.atoms[i2].mass / (self.atoms[i1].mass + self.atoms[i2].mass))
                    * self.omega_22[self.num_molecules + i1, self.num_molecules + i2]
                    / (kT * (self.atoms[i1].mass + self.atoms[i2].mass))
                )
                tmp *= 1 - 10 * self.omega_11[self.num_molecules + i1, self.num_molecules + i2] / (
                    3.0 * self.omega_22[self.num_molecules + i1, self.num_molecules + i2]
                )
                self.shear_viscosity_LHS[self.num_molecules + i1, self.num_molecules + i2] = tmp
                self.shear_viscosity_LHS[self.num_molecules + i2, self.num_molecules + i1] = tmp

        for i1 in range(self.num_atoms):
            tmp = 1.6 * (self.this_n_atom[i1] / n) * (self.this_n_atom[i1] / n) * self.omega_22[self.num_molecules + i1, self.num_molecules + i1] / kT
            for i2 in range(self.num_molecules):
                tmp += (
                    3.2
                    * (self.this_n_atom[i1] / n)
                    * (self.this_n_molecules[i2] / n)
                    * self.omega_22[i2, self.num_molecules + i1]
                    / kT
                    * (
                        10.0
                        * self.omega_11[i2, self.num_molecules + i1]
                        / (3.0 * self.omega_22[i2, self.num_molecules + i1])
                        * self.molecules[i2].mass
                        * self.atoms[i1].mass
                        / (self.molecules[i2].mass + self.atoms[i1].mass) ** 2
                        + self.molecules[i2].mass**2 / (self.molecules[i2].mass + self.atoms[i1].mass) ** 2
                    )
                )
            for i2 in range(self.num_atoms):
                if i2 == i1:
                    continue
                tmp += (
                    3.2
                    * (self.this_n_atom[i1] / n)
                    * (self.this_n_atom[i2] / n)
                    * self.omega_22[self.num_molecules + i2, self.num_molecules + i1]
                    / kT
                    * (
                        10.0
                        * self.omega_11[self.num_molecules + i2, self.num_molecules + i1]
                        / (3.0 * self.omega_22[self.num_molecules + i2, self.num_molecules + i1])
                        * self.atoms[i2].mass
                        * self.atoms[i1].mass
                        / (self.atoms[i2].mass + self.atoms[i1].mass) ** 2
                        + self.atoms[i2].mass**2 / (self.atoms[i2].mass + self.atoms[i1].mass) ** 2
                    )
                )
            self.shear_viscosity_LHS[self.num_molecules + i1, self.num_molecules + i1] = tmp

    def compute_shear_viscosity_RHS(self, temperature: float) -> None:
        self.shear_viscosity_RHS.fill(0.0)
        for i in range(self.num_molecules):
            self.shear_viscosity_RHS[i] = self.this_n_molecules[i]
        for i in range(self.num_atoms):
            self.shear_viscosity_RHS[self.num_molecules + i] = self.this_n_atom[i]
        self.shear_viscosity_RHS *= 2 / (constants.K_CONST_K * temperature * self.this_total_n)

    def compute_shear_viscosity_coeffs(self, temperature: float, model: ModelsOmega = ModelsOmega.ESA) -> None:
        self.compute_shear_viscosity_LHS(temperature, model)
        self.compute_shear_viscosity_RHS(temperature)
        lhs = self.shear_viscosity_LHS * 1e20
        rhs = self.shear_viscosity_RHS
        self.shear_viscosity_coeffs = self._solve_linear_system(lhs, rhs) * 1e20

    def shear_viscosity(self, temperature: float, model: ModelsOmega = ModelsOmega.ESA) -> float:
        self.compute_shear_viscosity_coeffs(temperature, model)
        res = 0.0
        for i in range(self.num_molecules):
            res += self.this_n_molecules[i] * self.shear_viscosity_coeffs[i]
        for i in range(self.num_atoms):
            res += self.this_n_atom[i] * self.shear_viscosity_coeffs[self.num_molecules + i]
        return res * constants.K_CONST_K * temperature / (2 * self.this_total_n)

    # ---------------- Diffusion ---------------- #
    def compute_diffusion_LHS(self, temperature: float) -> None:
        for i in range(self.n_vibr_levels_total):
            for k in range(self.n_vibr_levels_total):
                self.diffusion_LHS[i, k] = self.thermal_conductivity_LHS[i * 3, k * 3]
            for k in range(self.num_atoms):
                self.diffusion_LHS[i, self.n_vibr_levels_total + k] = self.thermal_conductivity_LHS[
                    i * 3, self.n_vibr_levels_total * 3 + k * 2
                ]
                self.diffusion_LHS[self.n_vibr_levels_total + k, i] = self.thermal_conductivity_LHS[
                    self.n_vibr_levels_total * 3 + k * 2, i * 3
                ]
        for i in range(self.num_atoms):
            for k in range(self.num_atoms):
                self.diffusion_LHS[self.n_vibr_levels_total + i, self.n_vibr_levels_total + k] = self.thermal_conductivity_LHS[
                    self.n_vibr_levels_total * 3 + i * 2, self.n_vibr_levels_total * 3 + k * 2
                ]

        offset = 0
        for i in range(self.num_molecules):
            for k in range(self.molecules[i].num_vibr_levels[0]):
                self.diffusion_LHS[0, offset] = self.this_n_vl_mol[i][k] * self.molecules[i].mass / (self.this_total_dens * 1e41)
                offset += 1
        for i in range(self.num_atoms):
            self.diffusion_LHS[0, offset] = self.this_n_atom[i] * self.atoms[i].mass / (self.this_total_dens * 1e41)
            offset += 1

    def compute_diffusion_RHS(self, temperature: float, b: int, n_level: int) -> None:
        self.diffusion_RHS.fill(0.0)
        offset = 0
        for i in range(self.num_molecules):
            for k in range(self.molecules[i].num_vibr_levels[0]):
                if i == b and k == n_level:
                    self.diffusion_RHS[offset] += 1.0
                self.diffusion_RHS[offset] -= self.this_n_vl_mol[i][k] * self.molecules[i].mass / self.this_total_dens
                offset += 1
        for i in range(self.num_atoms):
            if self.num_molecules + i == b:
                self.diffusion_RHS[offset] += 1.0
            self.diffusion_RHS[offset] -= self.this_n_atom[i] * self.atoms[i].mass / self.this_total_dens
            offset += 1
        self.diffusion_RHS *= 3 * constants.K_CONST_K * temperature
        self.diffusion_RHS[0] = 0.0

    def compute_diffusion_coeffs(self, temperature: float, b: int, n_level: int) -> None:
        self.compute_diffusion_RHS(temperature, b, n_level)
        self.diffusion_coeffs = self._solve_linear_system(self.diffusion_LHS, self.diffusion_RHS)

    def diffusion(self, temperature: float) -> None:
        self.compute_diffusion_LHS(temperature)
        offset = 0
        for i in range(self.num_molecules):
            for k in range(self.molecules[i].num_vibr_levels[0]):
                self.compute_diffusion_coeffs(temperature, i, k)
                self.diff[:, offset] = self.diffusion_coeffs
                offset += 1
        for i in range(self.num_atoms):
            self.compute_diffusion_coeffs(temperature, self.num_molecules + i, 0)
            self.diff[:, offset] = self.diffusion_coeffs
            offset += 1
        self.diff /= 2 * self.this_total_n

    def binary_diffusion(self, temperature: float, model: ModelsOmega = ModelsOmega.ESA) -> None:
        if self.num_molecules == 0:
            self.binary_diff.fill(0.0)
            return
        n = self.this_total_n
        rho = self.compute_density(self.this_n_vl_mol, self.this_n_atom)
        rho_A2 = self._compute_density_molecules_only(self.this_n_vl_mol)
        rho_A = self.this_n_atom[0] * self.atoms[0].mass if self.num_atoms else 0.0
        Y_A2 = rho_A2 / rho if rho else 0.0
        Y_A = rho_A / rho if rho else 0.0
        coll_mass_A_A2 = self.interactions[self.inter_index(0, self.num_molecules)].collision_mass if self.num_atoms else 0.0
        m_A2 = self.molecules[0].mass if self.num_molecules else 0.0
        m_A = self.atoms[0].mass if self.num_atoms else 0.0
        D_AA2 = (3.0 / 16.0) * (constants.K_CONST_K * temperature) / (n * coll_mass_A_A2 * self.omega_11[0, self.num_molecules]) if self.num_atoms else 0.0
        D_A2 = (3.0 / 8.0) * (constants.K_CONST_K * temperature) / (n * m_A2 * self.omega_11[0, 0]) if self.num_molecules else 0.0

        self.binary_diff.fill(0.0)
        if self.num_atoms == 0:
            for k in range(self.molecules[0].num_vibr_levels[0]):
                self.binary_diff[k] = D_A2 * (1.0 / ((self.molecules[0].mass * self.this_n_vl_mol[0][k]) / rho) - 1.0)
            self.binary_diff[self.molecules[0].num_vibr_levels[0]] = -D_A2
            return

        for k in range(self.molecules[0].num_vibr_levels[0]):
            self.binary_diff[k] = (
                D_AA2
                * (m_A / (rho / n))
                * (m_A / (rho / n))
                * (
                    (Y_A / D_A2)
                    + (2.0 / D_AA2)
                    * (1.0 / ((self.molecules[0].mass * self.this_n_vl_mol[0][k]) / rho) - Y_A - 1.0)
                )
                / ((Y_A2 / (2.0 * D_A2)) + (Y_A / D_AA2))
            )
        self.binary_diff[self.molecules[0].num_vibr_levels[0]] = (
            D_AA2
            * (m_A / (rho / n))
            * (m_A / (rho / n))
            * ((Y_A / D_A2) - (2.0 / D_AA2) * (Y_A + 1.0))
            / ((Y_A2 / (2.0 * D_A2)) + (Y_A / D_AA2))
        )
        self.binary_diff[self.molecules[0].num_vibr_levels[0] + 1] = -D_AA2 * (m_A2 * m_A) / ((rho / n) * (rho / n))
        self.binary_diff[self.molecules[0].num_vibr_levels[0] + 2] = (
            D_AA2 * (m_A2 * m_A) / ((rho / n) * (rho / n)) * ((1.0 / Y_A) - 1.0)
        )

    # ---------------- Transport driver and getters ---------------- #
    def compute_transport_coefficients(
        self,
        temperature: float,
        n_vl_molecule: Sequence[np.ndarray] | np.ndarray,
        n_atom: Optional[np.ndarray] = None,
        n_electrons: float = 0.0,
        model: ModelsOmega = ModelsOmega.ESA,
        perturbation: float = 1e-9,
    ) -> None:
        if isinstance(n_vl_molecule, np.ndarray) and n_atom is None:
            return self._compute_transport_from_state_vector(temperature, n_vl_molecule, model=model, perturbation=perturbation)

        if n_atom is None:
            n_atom = self.empty_n_atom
        self.cache_on = True
        self.this_n_atom = np.asarray(n_atom, dtype=float).copy()
        self.this_n_vl_mol = [np.asarray(vec, dtype=float).copy() for vec in n_vl_molecule]
        self.this_n_molecules = self.compute_n_molecule(self.this_n_vl_mol)
        self.this_total_n = self.compute_n(self.this_n_molecules, self.this_n_atom, n_electrons)
        self.this_total_dens = self.compute_density(self.this_n_molecules, self.this_n_atom, n_electrons)

        addcounter = 1 if self.is_ionized else 0
        if self.is_ionized:
            self.this_n_electrons = (1 - perturbation) * n_electrons + self.this_total_n * perturbation / (
                self.n_vibr_levels_total + self.num_atoms + addcounter
            )

        for i, vec in enumerate(self.this_n_vl_mol):
            for k in range(vec.size):
                vec[k] = (1 - perturbation) * vec[k] + self.this_total_n * perturbation / (
                    self.n_vibr_levels_total + self.num_atoms + addcounter
                )
        for i in range(self.this_n_atom.size):
            self.this_n_atom[i] = (1 - perturbation) * self.this_n_atom[i] + self.this_total_n * perturbation / (
                self.n_vibr_levels_total + self.num_atoms + addcounter
            )

        # Refresh molecule totals and overall totals after perturbation
        self.this_n_molecules = self.compute_n_molecule(self.this_n_vl_mol)
        self.this_total_n = self.compute_n(self.this_n_vl_mol, self.this_n_atom, n_electrons)
        self.this_total_dens = self.compute_density(self.this_n_vl_mol, self.this_n_atom, n_electrons)

        n_dens_cons = sum(float(np.sum(vec)) for vec in self.this_n_vl_mol) + float(np.sum(self.this_n_atom))
        if not (0.99 <= n_dens_cons / self.this_total_n <= 1.01):
            raise IncorrectValueException(f"Total number density is NOT conserved! {n_dens_cons/self.this_total_n}")

        if self.all_rigid_rotators:
            self.compute_c_rot_rigid_rot(temperature)
            self.compute_full_crot_rigid_rot(temperature)
        else:
            self.compute_c_rot(temperature)
            self.compute_full_crot(temperature)
        self.this_ctr = self.c_tr(self.this_n_vl_mol, self.this_n_atom, n_electrons)

        self.compute_omega11(temperature, model)
        self.compute_omega12(temperature, model)
        self.compute_omega13(temperature, model)
        self.compute_omega22(temperature, model)
        if self.is_ionized:
            dl = self.debye_length(temperature, self.this_n_vl_mol, self.this_n_atom, n_electrons)
            self.compute_omega11_with_debye(temperature, dl, model)
            self.compute_omega12_with_debye(temperature, dl, model)
            self.compute_omega13_with_debye(temperature, dl, model)
            self.compute_omega22_with_debye(temperature, dl, model)

        self.compute_rot_rel_times(temperature, self.this_total_n, model)
        self.th_cond = self.thermal_conductivity(temperature, model)
        self.sh_visc = self.shear_viscosity(temperature, model)
        self.b_visc = self.bulk_viscosity(temperature, model)
        self.thermodiffusion(temperature, model)
        self.diffusion(temperature)
        self.binary_diffusion(temperature, model)
        self.cache_on = False

    def _compute_transport_from_state_vector(
        self,
        temperature: float,
        n: np.ndarray,
        model: ModelsOmega = ModelsOmega.ESA,
        perturbation: float = 1e-9,
    ) -> None:
        if not self.all_rigid_rotators:
            raise IncorrectValueException(
                "Molecules in mixture are not rigid rotators, must pass vibrational level populations",
            )
        self.cache_on = True
        if self.num_molecules:
            self.this_n_molecules = np.asarray(n[: self.num_molecules], dtype=float).copy()
            self.this_n_vl_mol = [np.zeros(m.num_vibr_levels[0]) for m in self.molecules]
            for idx, mol in enumerate(self.molecules):
                if mol.num_vibr_levels:
                    self.this_n_vl_mol[idx][0] = self.this_n_molecules[idx]
        if self.num_atoms:
            self.this_n_atom = np.asarray(n[self.num_molecules : self.num_molecules + self.num_atoms], dtype=float).copy()
        if self.is_ionized:
            self.this_n_electrons = float(n[-1])

        self.this_total_n = self.compute_n_from_state_vector(n)
        self.this_total_dens = self.compute_density_from_state_vector(n)

        addcounter = 1 if self.is_ionized else 0
        if self.is_ionized:
            self.this_n_electrons = (1 - perturbation) * self.this_n_electrons + self.this_total_n * perturbation / (
                self.num_molecules + self.num_atoms + addcounter
            )
        for i in range(self.num_molecules):
            self.this_n_molecules[i] = (1 - perturbation) * self.this_n_molecules[i] + self.this_total_n * perturbation / (
                self.num_molecules + self.num_atoms + addcounter
            )
        for i in range(self.num_atoms):
            self.this_n_atom[i] = (1 - perturbation) * self.this_n_atom[i] + self.this_total_n * perturbation / (
                self.num_molecules + self.num_atoms + addcounter
            )

        self.compute_c_rot_rigid_rot(temperature)
        self.compute_full_crot_rigid_rot(temperature)
        self.this_ctr = self.c_tr_from_state_vector(n)

        self.compute_omega11(temperature, model)
        self.compute_omega12(temperature, model)
        self.compute_omega13(temperature, model)
        self.compute_omega22(temperature, model)
        if self.is_ionized:
            dl = self.debye_length_from_vector(temperature, n)
            self.compute_omega11_with_debye(temperature, dl, model)
            self.compute_omega12_with_debye(temperature, dl, model)
            self.compute_omega13_with_debye(temperature, dl, model)
            self.compute_omega22_with_debye(temperature, dl, model)

        self.compute_rot_rel_times(temperature, self.this_total_n, model)
        self.th_cond = self.thermal_conductivity(temperature, model)
        self.sh_visc = self.shear_viscosity(temperature, model)
        self.b_visc = self.bulk_viscosity(temperature, model)
        self.thermodiffusion(temperature, model)
        self.diffusion(temperature)
        self.binary_diffusion(temperature, model)
        self.cache_on = False

    def get_thermal_conductivity(self) -> float:
        return self.th_cond

    def get_shear_viscosity(self) -> float:
        return self.sh_visc

    def get_bulk_viscosity(self) -> float:
        return self.b_visc

    def get_thermodiffusion(self) -> np.ndarray:
        return self.th_diff

    def get_diffusion(self) -> np.ndarray:
        return self.diff

    def get_lite_diffusion(self) -> np.ndarray:
        return self.lite_diff

    def get_binary_diffusion(self) -> np.ndarray:
        return self.binary_diff
