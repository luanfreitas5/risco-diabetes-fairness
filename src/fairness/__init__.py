"""Auditoria de justiça (fairness) por subgrupos sensíveis.

Diferencial do projeto: além da métrica agregada, medimos o desempenho por
subgrupo (sexo, faixa etária, renda, escolaridade) e reportamos as disparidades
de forma transparente, usando ``fairlearn``.

Módulos
-------
audit
    Calcula métricas por subgrupo e disparidades (``MetricFrame``).
"""

from fairness.audit import FairnessReport, audit_fairness

__all__ = ["FairnessReport", "audit_fairness"]
