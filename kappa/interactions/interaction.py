from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import yaml

from .. import constants
from ..exceptions import DataNotFoundException, UnopenedFileException
from ..particles import Particle
from ..yaml_loader import safe_load_no_bool


class InteractionType:
    INTERACTION_NEUTRAL_NEUTRAL = "interaction_neutral_neutral"
    INTERACTION_NEUTRAL_ION = "interaction_neutral_ion"
    INTERACTION_NEUTRAL_ELECTRON = "interaction_neutral_electron"
    INTERACTION_CHARGED_CHARGED = "interaction_charged_charged"


@dataclass
class Interaction:
    particle1_name: str
    particle2_name: str
    charge1: int
    charge2: int
    collision_mass: float
    collision_diameter: float
    epsilon: float
    vss_data: bool
    vss_dref: float
    vss_omega: float
    vss_alpha: float
    vss_Tref: float
    vss_c_d: float
    vss_c_cs: float
    interaction_type: str
    data: Dict[str, float] = field(default_factory=dict)

    def __init__(self, particle1: Particle, particle2: Particle, filename: str = "interaction.yaml") -> None:
        self.particle1_name = particle1.name
        self.particle2_name = particle2.name
        self.charge1 = particle1.charge
        self.charge2 = particle2.charge
        self.vss_Tref = 273.0
        self.vss_dref = 0.0
        self.vss_omega = 0.0
        self.vss_alpha = 0.0
        self.vss_data = False
        self.vss_c_d = 0.0
        self.vss_c_cs = 0.0
        self.data = {}
        try:
            self._read_data(f"{self.particle1_name} + {self.particle2_name}", filename)
        except DataNotFoundException:
            self._read_data(f"{self.particle2_name} + {self.particle1_name}", filename, optional=True)
        self.interaction_type = self._infer_type(particle1, particle2)
        cd = 0.5 * (particle1.diameter + particle2.diameter)
        self.collision_mass = particle1.mass * particle2.mass / (particle1.mass + particle2.mass)
        self.collision_diameter = cd
        # Consistent with C++: epsilon mixed as sqrt(e1*e2)*(d1*d2)^3 / cd^6
        self.epsilon = (
            (particle1.lennard_jones_epsilon * particle2.lennard_jones_epsilon) ** 0.5
            * (particle1.diameter * particle2.diameter) ** 3
            / (cd**6)
        )
        if self.vss_data:
            gref = (2 * constants.K_CONST_K * self.vss_Tref / self.collision_mass) ** 0.5
            self.vss_c_d = self.vss_dref * gref ** (self.vss_omega - 0.5)
            self.vss_c_cs = constants.K_CONST_PI * self.vss_dref ** 2 * gref ** (2 * self.vss_omega - 1)
        else:
            self.vss_c_d = 0.0
            self.vss_c_cs = 0.0

    def _infer_type(self, particle1: Particle, particle2: Particle) -> str:
        if particle1.charge == 0 and particle2.charge == 0:
            return InteractionType.INTERACTION_NEUTRAL_NEUTRAL
        if (particle1.charge == 0) ^ (particle2.charge == 0):
            if particle1.name == "e-" or particle2.name == "e-":
                return InteractionType.INTERACTION_NEUTRAL_ELECTRON
            return InteractionType.INTERACTION_NEUTRAL_ION
        return InteractionType.INTERACTION_CHARGED_CHARGED

    def _read_data(self, name: str, filename: str, optional: bool = False) -> None:
        path = Path(filename)
        if not path.exists():
            raise UnopenedFileException(f"Could not load database file {filename}")
        try:
            raw = path.read_text()
            file = safe_load_no_bool(raw.replace("\t", " "))
        except yaml.YAMLError as exc:
            raise UnopenedFileException(f"Failed to parse {filename}: {exc}") from exc
        if name not in file:
            if optional:
                return
            raise DataNotFoundException(f"No data found for {name} interaction in the database")
        interaction = file[name]
        vss_counter = 0
        for key, value in interaction.items():
            if value is None:
                continue
            if isinstance(value, list):
                for idx, scalar in enumerate(value):
                    self.data[f"_{key}_{idx}"] = float(scalar)
            else:
                numeric_value = float(value)
                self.data[key] = numeric_value
                if key == "VSS, Tref":
                    self.vss_Tref = numeric_value
                    vss_counter += 1
                elif key == "VSS, dref":
                    self.vss_dref = numeric_value
                    vss_counter += 1
                elif key == "VSS, omega":
                    self.vss_omega = numeric_value
                    vss_counter += 1
                elif key == "VSS, alpha":
                    self.vss_alpha = numeric_value
                    vss_counter += 1
        self.vss_data = vss_counter == 4

    def __getitem__(self, name: str) -> float:
        if name not in self.data:
            raise DataNotFoundException(
                f"No {name} interaction parameter found for {self.particle1_name}+{self.particle2_name} interaction"
            )
        return self.data[name]
