"""Avaliação de modelos com quantificação de incerteza.

Um único score de CV não basta (regra do "senior bar" do CLAUDE.md): reportamos
a média com intervalo de confiança de 95% e, no holdout, o conjunto completo de
métricas para um limiar operacional.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from config.logging import get_logger
from config.settings import get_settings
from metrics.classification import compute_classification_metrics

logger = get_logger(__name__)


@dataclass(frozen=True)
class CrossValResult:
    """Resultado de validação cruzada com intervalo de confiança.

    Attributes
    ----------
    metric : str
        Nome da métrica avaliada (ex.: ``roc_auc``).
    scores : list[float]
        Score por fold.
    mean : float
        Média dos folds.
    std : float
        Desvio-padrão dos folds.
    ci95 : float
        Semi-amplitude do intervalo de confiança de 95%.
    """

    metric: str
    scores: list[float]
    mean: float
    std: float
    ci95: float

    def summary(self) -> str:
        """Retorna um resumo textual ``média ± IC95``.

        Returns
        -------
        str
            Ex.: ``"roc_auc: 0.8231 ± 0.0042 (IC 95%)"``.
        """
        return f"{self.metric}: {self.mean:.4f} ± {self.ci95:.4f} (IC 95%)"


def cross_validate_model(
    model: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    *,
    scoring: str | None = None,
    cv_folds: int | None = None,
    verbose: int = 0,
) -> CrossValResult:
    """Executa validação cruzada estratificada e reporta a incerteza.

    Parameters
    ----------
    model : Pipeline
        Pipeline de modelo (não treinado) a avaliar.
    x : pd.DataFrame
        Features de treino.
    y : pd.Series
        Alvo de treino.
    scoring : str, optional
        Métrica do scikit-learn. Se ``None``, usa a métrica-alvo do config.
    cv_folds : int, optional
        Número de folds. Se ``None``, usa ``evaluation.cv_folds`` do config.
    verbose : int, optional
        Verbosidade do scikit-learn. Se ``0``, não imprime nada.

    Returns
    -------
    CrossValResult
        Resultado agregado com média, desvio e IC de 95%.

    Examples
    --------
    >>> result = cross_validate_model(model, x_train, y_train)  # doctest: +SKIP
    >>> print(result.summary())  # doctest: +SKIP
    """
    settings = get_settings()
    metric = scoring or settings.evaluation.primary_metric
    folds = cv_folds or settings.evaluation.cv_folds
    verbose = verbose or settings.evaluation.verbose
    seed = settings.random_seed

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clone(model), x, y, scoring=metric, cv=cv, n_jobs=-1, verbose=verbose)

    mean = float(scores.mean())
    std = float(scores.std())
    ci95 = float(1.96 * std / np.sqrt(len(scores)))

    result = CrossValResult(
        metric=metric,
        scores=[float(s) for s in scores],
        mean=mean,
        std=std,
        ci95=ci95,
    )
    logger.info("Validação cruzada | %s", result.summary())
    return result


def evaluate_holdout(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    threshold: float | None = None,
) -> dict[str, float]:
    """Avalia um modelo treinado no conjunto de teste (holdout).

    Parameters
    ----------
    model : Pipeline
        Pipeline já treinado.
    x_test : pd.DataFrame
        Features de teste.
    y_test : pd.Series
        Alvo de teste.
    threshold : float, optional
        Limiar de decisão. Se ``None``, usa ``evaluation.decision_threshold``.

    Returns
    -------
    dict[str, float]
        Métricas de classificação e calibração no holdout.

    Examples
    --------
    >>> metrics = evaluate_holdout(model, x_test, y_test)  # doctest: +SKIP
    """
    settings = get_settings()
    thr = threshold if threshold is not None else settings.evaluation.decision_threshold
    y_proba = model.predict_proba(x_test)[:, 1]
    metrics = compute_classification_metrics(y_test, y_proba, threshold=thr)
    logger.info(
        "Holdout | ROC-AUC=%.4f | PR-AUC=%.4f | recall@%.2f=%.4f | Brier=%.4f",
        metrics["roc_auc"],
        metrics["pr_auc"],
        thr,
        metrics["recall"],
        metrics["brier_score"],
    )
    return metrics
