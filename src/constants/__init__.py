"""Constantes, nomes de colunas, rótulos e métricas do projeto.

Centraliza valores usados em todo o pipeline para evitar strings mágicas e
manter consistência entre dados, features e avaliação.

Módulos
-------
columns
    Nomes e agrupamentos das colunas do dataset BRFSS.
labels
    Rótulos da variável-alvo e mapeamentos de subgrupos sensíveis.
metrics
    Nomes das métricas de avaliação e de justiça.
"""

from constants import columns, labels, metrics

__all__ = ["columns", "labels", "metrics"]
