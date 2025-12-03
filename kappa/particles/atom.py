from __future__ import annotations

from .particle import Particle


class Atom(Particle):
    """Python version of `kappa::Atom`."""

    def __init__(self, name: str, filename: str = "particles.yaml") -> None:
        super().__init__(name=name, filename=filename)
