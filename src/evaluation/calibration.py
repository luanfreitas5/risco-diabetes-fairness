"""Análise de calibração de probabilidades.

As probabilidades alimentam a decisão de triagem, então precisam ser
confiáveis: quando o modelo diz "30% de risco", ~30% dos casos devem de fato ser
positivos. Aqui computamos a curva de confiabilidade e o Brier score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


@dataclass(frozen=True)
class CalibrationResult:
    """Dados da curva de calibração.

    Attributes
    ----------
    prob_true : np.ndarray
        Fração observada de positivos por bin.
    prob_pred : np.ndarray
        Probabilidade média prevista por bin.
    brier_score : float
        Brier score (quanto menor, melhor a calibração).
    """

    prob_true: np.ndarray
    prob_pred: np.ndarray
    brier_score: float


def compute_calibration_curve(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    *,
    n_bins: int = 10,
) -> CalibrationResult:
    """Computa a curva de confiabilidade e o Brier score.

    Parameters
    ----------
    y_true : ArrayLike
        Rótulos verdadeiros (0/1).
    y_proba : ArrayLike
        Probabilidades previstas da classe positiva.
    n_bins : int, optional
        Número de bins da curva, by default 10.

    Returns
    -------
    CalibrationResult
        Curva de calibração e Brier score.

    Examples
    --------
    >>> res = compute_calibration_curve([0, 1, 1, 0], [0.2, 0.8, 0.6, 0.1], n_bins=5)
    >>> res.brier_score < 0.2
    True
    """
    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba)
    prob_true, prob_pred = calibration_curve(
        y_true_arr, y_proba_arr, n_bins=n_bins, strategy="quantile"
    )
    brier = float(brier_score_loss(y_true_arr, y_proba_arr))
    return CalibrationResult(prob_true=prob_true, prob_pred=prob_pred, brier_score=brier)
