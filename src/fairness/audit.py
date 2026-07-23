"""Auditoria de justiça por subgrupo com ``fairlearn``.

Para cada atributo sensível, calcula métricas por subgrupo (recall, precisão,
taxa de seleção, TPR/FPR) e as disparidades entre os grupos (diferença e razão).
Disparidades altas indicam que o modelo trata subgrupos de forma desigual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from numpy.typing import ArrayLike
from sklearn.metrics import precision_score, recall_score

from config.logging import get_logger
from config.settings import get_settings
from utils.numbers import to_float

logger = get_logger(__name__)

# O stub do scikit-learn tipa ``zero_division`` como ``str``, mas a implementação
# aceita 0/1/np.nan em tempo de execução. ``Any`` evita ruído de tipagem no ponto
# de uso sem mascarar outros erros reais.
_ZERO_DIVISION: Any = 0

# Métricas calculadas por subgrupo na auditoria.
_METRICS = {
    "recall": lambda yt, yp: recall_score(yt, yp, zero_division=_ZERO_DIVISION),
    "precision": lambda yt, yp: precision_score(yt, yp, zero_division=_ZERO_DIVISION),
    "selection_rate": selection_rate,
    "true_positive_rate": true_positive_rate,
    "false_positive_rate": false_positive_rate,
}


@dataclass
class FairnessReport:
    """Relatório de justiça agregando os resultados por atributo sensível.

    Attributes
    ----------
    by_group : dict[str, pd.DataFrame]
        Para cada atributo sensível, as métricas por subgrupo.
    disparities : dict[str, dict[str, float]]
        Para cada atributo, as disparidades (diferença e razão) por métrica.
    """

    by_group: dict[str, pd.DataFrame] = field(default_factory=dict)
    disparities: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serializa o relatório para dicionário (JSON-friendly).

        Returns
        -------
        dict[str, object]
            Estrutura com métricas por grupo e disparidades.
        """
        return {
            "by_group": {k: v.to_dict(orient="index") for k, v in self.by_group.items()},
            "disparities": self.disparities,
        }


def audit_fairness(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    sensitive: pd.DataFrame,
    *,
    threshold: float | None = None,
) -> FairnessReport:
    """Audita a justiça do modelo por subgrupo para cada atributo sensível.

    Parameters
    ----------
    y_true : ArrayLike
        Rótulos verdadeiros (0/1).
    y_proba : ArrayLike
        Probabilidades previstas da classe positiva.
    sensitive : pd.DataFrame
        DataFrame com uma coluna por atributo sensível (ex.: Sex, AgeBand).
    threshold : float, optional
        Limiar de decisão. Se ``None``, usa ``evaluation.decision_threshold``.

    Returns
    -------
    FairnessReport
        Métricas por subgrupo e disparidades por atributo.

    Examples
    --------
    >>> report = audit_fairness(y_test, y_proba, sensitive_test)  # doctest: +SKIP
    >>> report.by_group["Sex"]  # doctest: +SKIP
    """
    settings = get_settings()
    thr = threshold if threshold is not None else settings.evaluation.decision_threshold
    y_true_arr = np.asarray(y_true)
    y_pred = (np.asarray(y_proba) >= thr).astype(int)

    report = FairnessReport()
    for feature in sensitive.columns:
        frame = MetricFrame(
            metrics=_METRICS,
            y_true=y_true_arr,
            y_pred=y_pred,
            sensitive_features=sensitive[feature],
        )
        report.by_group[feature] = cast(pd.DataFrame, frame.by_group)
        report.disparities[feature] = {
            "recall_difference": to_float(frame.difference()["recall"]),
            "recall_ratio": to_float(frame.ratio()["recall"]),
            "selection_rate_difference": to_float(frame.difference()["selection_rate"]),
            "tpr_difference": to_float(frame.difference()["true_positive_rate"]),
            "fpr_difference": to_float(frame.difference()["false_positive_rate"]),
        }
        logger.info(
            "Fairness [%s] | disparidade de recall=%.4f | disparidade de seleção=%.4f",
            feature,
            report.disparities[feature]["recall_difference"],
            report.disparities[feature]["selection_rate_difference"],
        )

    return report
