"""Interpretabilidade com SHAP para o pipeline de modelo.

Extrai o estimador final do pipeline (desembrulhando a calibração, se houver),
aplica o pré-processamento às features e calcula os valores SHAP. Em datasets
grandes, uma amostra é usada para manter o custo computacional aceitável.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from config.logging import get_logger
from config.settings import get_settings

logger = get_logger(__name__)

_MAX_SAMPLE = 2000


@dataclass(frozen=True)
class ShapResult:
    """Resultado do cálculo de SHAP.

    Attributes
    ----------
    values : np.ndarray
        Valores SHAP (linhas x features) para a classe positiva.
    feature_names : list[str]
        Nomes das features na ordem das colunas de ``values``.
    data : np.ndarray
        Matriz de features (pré-processada) usada no cálculo.
    """

    values: np.ndarray
    feature_names: list[str]
    data: np.ndarray


def _unwrap_estimator(estimator: object) -> object:
    """Desembrulha o estimador base de um ``CalibratedClassifierCV``.

    Parameters
    ----------
    estimator : object
        Estimador final do pipeline (possivelmente calibrado).

    Returns
    -------
    object
        Estimador base treinado, adequado para o SHAP.
    """
    if isinstance(estimator, CalibratedClassifierCV):
        # Usa o estimador base do primeiro fold calibrado (aproximação usual).
        # O stub do scikit-learn tipa ``calibrated_classifiers_`` de forma
        # imprecisa (Generator em vez de list); o cast reflete o tipo real.
        calibrated = cast("list", estimator.calibrated_classifiers_)
        return calibrated[0].estimator
    return estimator


def compute_shap_values(
    model: Pipeline,
    x: pd.DataFrame,
    *,
    max_sample: int = _MAX_SAMPLE,
) -> ShapResult:
    """Calcula os valores SHAP para uma amostra das features.

    Parameters
    ----------
    model : Pipeline
        Pipeline treinado (pré-processador + estimador).
    x : pd.DataFrame
        Features a explicar.
    max_sample : int, optional
        Tamanho máximo da amostra usada no cálculo, by default 2000.

    Returns
    -------
    ShapResult
        Valores SHAP, nomes das features e matriz pré-processada.

    Examples
    --------
    >>> result = compute_shap_values(model, x_test)  # doctest: +SKIP
    """
    seed = get_settings().random_seed
    sample = x.sample(n=min(max_sample, len(x)), random_state=seed) if len(x) > max_sample else x

    preprocessor = model.named_steps["preprocessor"]
    estimator = _unwrap_estimator(model.named_steps["model"])

    x_transformed = preprocessor.transform(sample)
    feature_names = list(preprocessor.get_feature_names_out())

    logger.info("Calculando valores SHAP para %d amostras.", len(sample))
    explainer = shap.Explainer(estimator, x_transformed, feature_names=feature_names)
    # O stub do shap tipa o retorno como ``Explanation | list[Explanation]``,
    # mas para um único explainer o retorno é sempre uma ``Explanation``.
    explanation = cast(shap.Explanation, explainer(x_transformed, check_additivity=False))

    values = np.asarray(explanation.values)
    # Para saída binária (linhas x features x classes), fica com a classe positiva.
    if values.ndim == 3:
        values = values[:, :, 1]

    return ShapResult(values=values, feature_names=feature_names, data=np.asarray(x_transformed))


def compute_global_importance(result: ShapResult) -> pd.Series:
    """Calcula a importância global das features (média do |SHAP|).

    Parameters
    ----------
    result : ShapResult
        Resultado de :func:`compute_shap_values`.

    Returns
    -------
    pd.Series
        Importância média por feature, ordenada de forma decrescente.

    Examples
    --------
    >>> importance = compute_global_importance(result)  # doctest: +SKIP
    """
    mean_abs = np.abs(result.values).mean(axis=0)
    return pd.Series(mean_abs, index=result.feature_names).sort_values(ascending=False)
