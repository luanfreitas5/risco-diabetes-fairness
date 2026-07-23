"""Exceção base do projeto."""

from __future__ import annotations


class DiabetesFairnessError(Exception):
    """Exceção base para todos os erros do projeto.

    Todas as exceções customizadas herdam desta classe, permitindo capturar
    qualquer erro do domínio com um único ``except``.
    """
