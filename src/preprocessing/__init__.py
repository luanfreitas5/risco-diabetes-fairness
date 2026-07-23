"""Pré-processamento e construção de pipelines de transformação.

Módulos
-------
pipeline
    Constrói o ``ColumnTransformer`` de pré-processamento por tipo de modelo.
"""

from preprocessing.pipeline import build_preprocessor

__all__ = ["build_preprocessor"]
