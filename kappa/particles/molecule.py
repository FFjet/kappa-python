from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import yaml

from .. import constants
from ..exceptions import DataNotFoundException, UnopenedFileException
from ..yaml_loader import safe_load_no_bool
from .particle import Particle


@dataclass
class Molecule(Particle):
    """Python translation of `kappa::Molecule`."""

    anharmonic_spectrum: bool = True
    rigid_rotator: bool = True
    reduced_osc_mass: float = 0.0
    mA_mAB: float = 0.0
    mB_mAB: float = 0.0
    rot_inertia: float = 0.0
    internuclear_distance: float = 0.0
    rot_symmetry: int = 1
    vibr_frequency: np.ndarray = field(default_factory=lambda: np.zeros(0))
    vibr_we_xe: np.ndarray = field(default_factory=lambda: np.zeros(0))
    vibr_we_ye: np.ndarray = field(default_factory=lambda: np.zeros(0))
    vibr_we_ze: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rot_be: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rot_ae: np.ndarray = field(default_factory=lambda: np.zeros(0))
    characteristic_vibr_temperatures: np.ndarray = field(default_factory=lambda: np.zeros(0))
    diss_energy: np.ndarray = field(default_factory=lambda: np.zeros(0))
    num_rot_levels: List[List[int]] = field(default_factory=list)
    level_vibr_energies: List[float] = field(default_factory=list)
    num_vibr_levels: List[int] = field(default_factory=list)
    rot_energy: List[List[np.ndarray]] = field(default_factory=list)
    vibr_energy: List[np.ndarray] = field(default_factory=list)
    parker_const: float = 0.0

    def __init__(
        self,
        name: str,
        anharmonic_spectrum: bool = True,
        rigid_rotator: bool = True,
        filename: str = "particles.yaml",
    ) -> None:
        super().__init__(name=None, filename=filename, auto_load=False)
        self.anharmonic_spectrum = anharmonic_spectrum
        self.rigid_rotator = rigid_rotator
        self.reduced_osc_mass = 0.0
        self.mA_mAB = 0.0
        self.mB_mAB = 0.0
        self.rot_inertia = 0.0
        self.internuclear_distance = 0.0
        self.rot_symmetry = 1
        self.vibr_frequency = np.zeros(0)
        self.vibr_we_xe = np.zeros(0)
        self.vibr_we_ye = np.zeros(0)
        self.vibr_we_ze = np.zeros(0)
        self.rot_be = np.zeros(0)
        self.rot_ae = np.zeros(0)
        self.characteristic_vibr_temperatures = np.zeros(0)
        self.diss_energy = np.zeros(0)
        self.num_rot_levels = []
        self.level_vibr_energies = []
        self.num_vibr_levels = []
        self.rot_energy = []
        self.vibr_energy = []
        self.parker_const = 0.0
        super().read_data(name, filename)
        self._read_molecule_data(name, filename)
        self._build_spectra(anharmonic_spectrum, rigid_rotator)

    def _read_molecule_data(self, name: str, filename: str) -> None:
        path = Path(filename)
        if not path.exists():
            raise UnopenedFileException(f"Could not open database file {filename}")
        try:
            raw = path.read_text()
            file = safe_load_no_bool(raw.replace("\t", " "))
        except yaml.YAMLError as exc:  # pragma: no cover - propagate parsing issue
            raise UnopenedFileException(f"Failed to parse {filename}: {exc}") from exc
        if name not in file:
            raise DataNotFoundException(f"No data found for {name} in the database")
        particle = file[name]
        if "Dissociation energy, J" not in particle:
            raise DataNotFoundException(f"No data found for {name} in the database")
        self.diss_energy = np.asarray(particle.get("Dissociation energy, J", []), dtype=float)
        self.rot_symmetry = int(particle.get("Factor of symmetry", self.rot_symmetry))
        self.parker_const = float(particle.get("Parker (zeta^infty)", self.parker_const))
        self.reduced_osc_mass = float(particle.get("Reduced oscillator mass", self.reduced_osc_mass))
        self.rot_inertia = float(particle.get("Moment of Inertia, J*s^2", self.rot_inertia))
        self.vibr_frequency = np.asarray(particle.get("Frequency of vibrations (we), m^-1", []), dtype=float)
        self.vibr_we_xe = np.asarray(particle.get("wexe, m^-1", []), dtype=float)
        self.vibr_we_ye = np.asarray(particle.get("weye, m^-1", []), dtype=float)
        self.vibr_we_ze = np.asarray(particle.get("weze, m^-1", []), dtype=float)
        self.rot_be = np.asarray(particle.get("Be, m^-1", []), dtype=float)
        self.rot_ae = np.asarray(particle.get("ae, m^-1", []), dtype=float)
        self.internuclear_distance = float(particle.get("internuclear distance, r_e , m", self.internuclear_distance))
        self.mA_mAB = float(particle.get("mA/mAB", self.mA_mAB))
        self.mB_mAB = float(particle.get("mB/mAB", self.mB_mAB))

    def _build_spectra(self, anharmonic_spectrum: bool, rigid_rotator: bool) -> None:
        num_levels = int(self.num_electron_levels)
        self.num_rot_levels = []
        self.num_vibr_levels = []
        self.rot_energy = []
        self.vibr_energy = []
        self.level_vibr_energies = []
        for e in range(num_levels):
            if not anharmonic_spectrum:
                level_we_xe = 0.0
                level_we_ye = 0.0
                level_we_ze = 0.0
                self.anharmonic_spectrum = False
            else:
                level_we_xe = float(self.vibr_we_xe[e]) if self.vibr_we_xe.size > e else 0.0
                level_we_ye = float(self.vibr_we_ye[e]) if self.vibr_we_ye.size > e else 0.0
                level_we_ze = float(self.vibr_we_ze[e]) if self.vibr_we_ze.size > e else 0.0
                self.anharmonic_spectrum = True
            tmp = 0.0
            i = 0
            new_vibr_energies: List[float] = []
            limit = float(self.diss_energy[e] - self.electron_energy[e])
            while tmp < limit:
                tmp = constants.K_CONST_H * constants.K_CONST_C * (
                    self.vibr_frequency[e] * (i + 0.5)
                    - level_we_xe * (i + 0.5) ** 2
                    + level_we_ye * (i + 0.5) ** 3
                    + level_we_ze * (i + 0.5) ** 4
                )
                if tmp < limit:
                    if i == 0 or tmp > new_vibr_energies[-1]:
                        new_vibr_energies.append(tmp)
                        i += 1
                    else:
                        break
            self.num_vibr_levels.append(i)
            self.vibr_energy.append(np.asarray(new_vibr_energies, dtype=float))
            if rigid_rotator:
                ev_level_rot_energies: List[float] = []
                tmp = 0.0
                j = 0
                while tmp < limit:
                    tmp = constants.K_CONST_H * constants.K_CONST_C * (self.rot_be[e] * j * (j + 1))
                    if tmp < limit:
                        if j == 0 or tmp > ev_level_rot_energies[-1]:
                            ev_level_rot_energies.append(tmp)
                            j += 1
                        else:
                            break
                e_level_rot_amt = [j] * i
                e_level_rot_energies = [np.asarray(ev_level_rot_energies, dtype=float) for _ in range(i)]
                self.num_rot_levels.append(e_level_rot_amt)
                self.rot_energy.append(e_level_rot_energies)
            else:
                e_level_rot_amt: List[int] = []
                e_level_rot_energies: List[np.ndarray] = []
                for vib_level in range(i):
                    ev_level_rot_energies: List[float] = []
                    tmp = 0.0
                    j = 0
                    while tmp + self.vibr_energy[e][vib_level] < limit:
                        tmp = constants.K_CONST_H * constants.K_CONST_C * (
                            (self.rot_be[e] - self.rot_ae[e] * (vib_level + 0.5)) * j * (j + 1)
                        )
                        if tmp + self.vibr_energy[e][vib_level] < limit:
                            if j == 0 or tmp > ev_level_rot_energies[-1]:
                                ev_level_rot_energies.append(tmp)
                                j += 1
                            else:
                                break
                    e_level_rot_amt.append(j)
                    e_level_rot_energies.append(np.asarray(ev_level_rot_energies, dtype=float))
                self.num_rot_levels.append(e_level_rot_amt)
                self.rot_energy.append(e_level_rot_energies)
        if self.vibr_frequency.size:
            self.characteristic_vibr_temperatures = (
                constants.K_CONST_H * constants.K_CONST_C * self.vibr_frequency / constants.K_CONST_K
            )
        else:
            self.characteristic_vibr_temperatures = np.zeros(0)
