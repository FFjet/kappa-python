"""Exception hierarchy mirroring the original C++ layer."""
from __future__ import annotations

class KappaError(RuntimeError):
    """Base class for domain specific exceptions."""


class UnopenedFileException(KappaError):
    pass


class DataNotFoundException(KappaError):
    pass


class ModelParameterException(KappaError):
    pass


class IncorrectValueException(KappaError):
    pass
