"""Nomes das métricas de avaliação e de justiça usadas no projeto.

Centralizar os nomes evita divergência entre o código de avaliação, os
relatórios e o Model Card.
"""

from __future__ import annotations

from typing import Final

# --- Métricas de classificação -----------------------------------------------
ROC_AUC: Final[str] = "roc_auc"
PR_AUC: Final[str] = "pr_auc"  # average precision (informativa sob desbalanceamento)
RECALL: Final[str] = "recall"  # sensibilidade — crítica em triagem de saúde
PRECISION: Final[str] = "precision"
F1: Final[str] = "f1"
BALANCED_ACCURACY: Final[str] = "balanced_accuracy"
BRIER: Final[str] = "brier_score"  # qualidade de calibração (quanto menor, melhor)

CLASSIFICATION_METRICS: Final[list[str]] = [
    ROC_AUC,
    PR_AUC,
    RECALL,
    PRECISION,
    F1,
    BALANCED_ACCURACY,
    BRIER,
]

# --- Métricas de justiça (fairness) ------------------------------------------
TRUE_POSITIVE_RATE: Final[str] = "true_positive_rate"  # paridade de oportunidade
FALSE_POSITIVE_RATE: Final[str] = "false_positive_rate"
SELECTION_RATE: Final[str] = "selection_rate"  # paridade demográfica

FAIRNESS_METRICS: Final[list[str]] = [
    TRUE_POSITIVE_RATE,
    FALSE_POSITIVE_RATE,
    SELECTION_RATE,
    RECALL,
    PRECISION,
]
