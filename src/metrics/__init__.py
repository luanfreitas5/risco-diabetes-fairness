"""Cálculo de métricas de classificação e calibração.

Módulos
-------
classification
    Métricas de classificação (ROC-AUC, PR-AUC, recall, F1, Brier).
"""

from metrics.classification import (
    compute_classification_metrics,
    recall_at_threshold,
)

__all__ = ["compute_classification_metrics", "recall_at_threshold"]
