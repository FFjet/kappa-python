from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml

from .. import constants
from ..exceptions import DataNotFoundException, UnopenedFileException
from ..yaml_loader import safe_load_no_bool


@dataclass
class Particle:
    """Python counterpart of the C++ `kappa::Particle` class."""

    name: str = ""
    mass: float = 0.0
    diameter: float = 0.0
    charge: int = 0
    formation_energy: float = 0.0
    lennard_jones_epsilon: float = 0.0
    ionization_potential: float = 0.0
    num_electron_levels: int = 0
    electron_energy: np.ndarray = field(default_factory=lambda: np.zeros(0))
    statistical_weight: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    particle_type: str = ""
    stoichiometry: List[Tuple[str, int]] = field(default_factory=list)
    atomic_number: int = 0
    atomic_weight: float = 0.0
    element_list: List[Tuple[str, float]] = field(default_factory=list)

    def __init__(
        self,
        name: str | None = None,
        filename: str = "particles.yaml",
        *,
        auto_load: bool = True,
    ) -> None:
        self.name = ""
        self.mass = 0.0
        self.diameter = 0.0
        self.charge = 0
        self.formation_energy = 0.0
        self.lennard_jones_epsilon = 0.0
        self.ionization_potential = 0.0
        self.num_electron_levels = 0
        self.electron_energy = np.zeros(0)
        self.statistical_weight = np.zeros(0, dtype=np.int64)
        self.particle_type = ""
        self.stoichiometry = []
        self.atomic_number = 0
        self.atomic_weight = 0.0
        self.element_list = []
        if name and auto_load:
            self.read_data(name, filename)

    def read_data(self, name: str, filename: str) -> None:
        path = Path(filename)
        if not path.exists():
            raise UnopenedFileException(f"Could not load database file {filename}")
        try:
            raw = path.read_text()
            file = safe_load_no_bool(raw.replace("\t", " "))
        except yaml.YAMLError as exc:  # pragma: no cover - propagate parsing issue
            raise UnopenedFileException(f"Failed to parse {filename}: {exc}") from exc
        if name not in file:
            raise DataNotFoundException(f"No data found for {name} in the database")
        self.name = name
        particle = file[name]
        self.mass = float(particle.get("Mass, kg", self.mass))
        self.diameter = float(particle.get("Diameter, m", self.diameter))
        self.formation_energy = float(particle.get("Formation energy, J", self.formation_energy))
        self.charge = int(particle.get("Charge", self.charge))
        self.particle_type = particle.get("Type", self.particle_type)
        stoichiometry = particle.get("Stoichiometry", {})
        if isinstance(stoichiometry, dict):
            self.stoichiometry = [(str(k), int(v)) for k, v in stoichiometry.items()]
        lj_key = next(
            (key for key in particle.keys() if "Parameter" in key and "Lennard-Jones" in key),
            None,
        )
        if lj_key is not None:
            self.lennard_jones_epsilon = float(particle[lj_key])
        self.ionization_potential = float(particle.get("Ionization potential, J", self.ionization_potential))
        electron_energy = particle.get("Electronic energy, J")
        if electron_energy is not None:
            self.electron_energy = np.asarray(electron_energy, dtype=float)
        statistical_weight = particle.get("Statistical weight")
        if statistical_weight is not None:
            self.statistical_weight = np.asarray(statistical_weight, dtype=np.int64)
            self.num_electron_levels = self.statistical_weight.size
        else:
            self.statistical_weight = np.zeros(0, dtype=np.int64)
            self.num_electron_levels = 0
        element_list = particle.get("Element list")
        if isinstance(element_list, dict):
            self.element_list = [(str(k), float(v)) for k, v in element_list.items()]
        self.atomic_number = int(particle.get("Atomic number", self.atomic_number))
        self.atomic_weight = float(particle.get("Atomic weight", self.atomic_weight))
