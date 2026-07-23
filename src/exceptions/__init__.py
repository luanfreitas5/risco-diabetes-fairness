"""Exceções customizadas do projeto.

Fornecem mensagens contextualizadas em pt-BR e uma hierarquia comum, facilitando
o tratamento de erros específicos de dados, modelos e configuração.

Módulos
-------
base
    Exceção base do projeto (:class:`DiabetesFairnessError`).
data
    Exceções relacionadas a dados (:class:`DataValidationError`, etc.).
model
    Exceções relacionadas a modelos (:class:`ModelNotFittedError`, etc.).
configuration
    Exceções de configuração (:class:`ConfigurationError`).
"""

from exceptions.base import DiabetesFairnessError
from exceptions.configuration import ConfigurationError
from exceptions.data import DataNotFoundError, DataValidationError
from exceptions.model import ModelNotFittedError, ModelPersistenceError

__all__ = [
    "ConfigurationError",
    "DataNotFoundError",
    "DataValidationError",
    "DiabetesFairnessError",
    "ModelNotFittedError",
    "ModelPersistenceError",
]
