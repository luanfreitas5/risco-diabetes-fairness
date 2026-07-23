"""Engenharia de features do projeto.

Módulos
-------
engineering
    Cria features derivadas (ex.: ``AgeBand`` a partir de ``Age``).
"""

from features.engineering import add_age_band, engineer_features

__all__ = ["add_age_band", "engineer_features"]
