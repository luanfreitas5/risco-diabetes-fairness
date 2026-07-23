"""Exceções relacionadas a dados."""

from __future__ import annotations

from exceptions.base import DiabetesFairnessError


class DataNotFoundError(DiabetesFairnessError):
    """Levantada quando um arquivo de dados esperado não é encontrado."""


class DataValidationError(DiabetesFairnessError):
    """Levantada quando os dados violam o contrato (schema Pandera)."""
