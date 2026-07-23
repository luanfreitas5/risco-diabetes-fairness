"""Exceções de configuração."""

from __future__ import annotations

from exceptions.base import DiabetesFairnessError


class ConfigurationError(DiabetesFairnessError):
    """Levantada quando a configuração é inválida ou está incompleta."""
