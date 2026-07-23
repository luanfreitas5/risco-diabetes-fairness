"""Métricas de classificação para triagem de risco desbalanceada.

A métrica-alvo para seleção de modelo é a **ROC-AUC** (independente de limiar).
Como a prevalência é ~14%, reportamos também a **PR-AUC** (average precision) e
o **recall** no limiar operacional — em triagem, falsos negativos (não sinalizar
alguém em risco) são o erro mais custoso. O **Brier score** mede a calibração.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from constants import metrics as m

# O stub do scikit-learn tipa ``zero_division`` como ``str``, mas a implementação
# aceita 0/1/np.nan em tempo de execução. ``Any`` evita ruído de tipagem no ponto
# de uso sem mascarar outros erros reais.
_ZERO_DIVISION: Any = 0


def recall_at_threshold(y_true: ArrayLike, y_proba: ArrayLike, threshold: float) -> float:
    """Calcula o recall (sensibilidade) aplicando um limiar às probabilidades.

    Parameters
    ----------
    y_true : ArrayLike
        Rótulos verdadeiros (0/1).
    y_proba : ArrayLike
        Probabilidades previstas da classe positiva.
    threshold : float
        Limiar de decisão (0 a 1).

    Returns
    -------
    float
        Recall no limiar informado.

    Examples
    --------
    >>> recall_at_threshold([0, 1, 1], [0.1, 0.6, 0.4], 0.3)
    1.0
    """
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return float(recall_score(y_true, y_pred, zero_division=_ZERO_DIVISION))


def compute_classification_metrics(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    *,
    threshold: float = 0.30,
) -> dict[str, float]:
    """Calcula o conjunto de métricas de classificação e calibração.

    Parameters
    ----------
    y_true : ArrayLike
        Rótulos verdadeiros (0/1).
    y_proba : ArrayLike
        Probabilidades previstas da classe positiva.
    threshold : float, optional
        Limiar de decisão para métricas dependentes de corte, by default 0.30.

    Returns
    -------
    dict[str, float]
        Dicionário com ROC-AUC, PR-AUC, recall, precisão, F1, acurácia
        balanceada e Brier score.

    Examples
    --------
    >>> metrics = compute_classification_metrics([0, 1, 1, 0], [0.2, 0.7, 0.6, 0.1])
    >>> round(metrics["roc_auc"], 3)
    1.0
    """
    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba)
    y_pred = (y_proba_arr >= threshold).astype(int)

    return {
        m.ROC_AUC: float(roc_auc_score(y_true_arr, y_proba_arr)),
        m.PR_AUC: float(average_precision_score(y_true_arr, y_proba_arr)),
        m.RECALL: float(recall_score(y_true_arr, y_pred, zero_division=_ZERO_DIVISION)),
        m.PRECISION: float(precision_score(y_true_arr, y_pred, zero_division=_ZERO_DIVISION)),
        m.F1: float(f1_score(y_true_arr, y_pred, zero_division=_ZERO_DIVISION)),
        m.BALANCED_ACCURACY: float(balanced_accuracy_score(y_true_arr, y_pred)),
        m.BRIER: float(brier_score_loss(y_true_arr, y_proba_arr)),
    }
