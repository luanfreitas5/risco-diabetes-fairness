"""Exceções relacionadas a modelos."""

from __future__ import annotations

from exceptions.base import DiabetesFairnessError


class ModelNotFittedError(DiabetesFairnessError):
    """Levantada ao tentar prever/avaliar com um modelo ainda não treinado."""


class ModelPersistenceError(DiabetesFairnessError):
    """Levantada em falhas ao salvar ou carregar um modelo do disco."""
