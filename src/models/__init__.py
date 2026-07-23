"""Definição, construção e persistência de modelos.

Módulos
-------
factory
    Fábrica de pipelines de modelo (baseline logístico e LightGBM) com
    pré-processamento acoplado e calibração opcional.
persistence
    Salva e carrega modelos treinados com ``joblib``.
"""

from models.factory import build_model, list_available_models
from models.persistence import load_model, save_model

__all__ = ["build_model", "list_available_models", "load_model", "save_model"]
