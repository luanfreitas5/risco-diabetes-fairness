"""Testes da interpretabilidade SHAP."""

from __future__ import annotations

import polars as pl
import pytest

from constants import columns as col
from explainability.shap_explainer import compute_global_importance, compute_shap_values
from models.factory import build_model


@pytest.fixture(scope="module")
def small_fit(synthetic_raw: pl.DataFrame):
    """Amostra pequena e modelo logístico ajustado para acelerar o cálculo SHAP."""
    pdf = synthetic_raw.to_pandas().head(60)
    x = pdf[col.FEATURES]
    y = pdf[col.TARGET].astype(int)
    model = build_model("logistic", calibrate=False)
    model.fit(x, y)
    return model, x


@pytest.fixture(scope="module")
def shap_result(small_fit):
    """Valores SHAP computados uma única vez e reutilizados entre os testes."""
    model, x = small_fit
    return compute_shap_values(model, x, max_sample=60)


@pytest.mark.smoke
def test_compute_shap_values_shapes_match_features(small_fit, shap_result) -> None:
    """Os valores SHAP têm uma coluna por feature transformada."""
    _, x = small_fit
    assert shap_result.values.shape[0] == len(x)
    assert shap_result.values.shape[1] == len(shap_result.feature_names)
    assert shap_result.data.shape == shap_result.values.shape


def test_compute_shap_values_respects_max_sample(small_fit) -> None:
    """Uma amostra máxima menor do que os dados reduz o número de linhas explicadas."""
    model, x = small_fit
    result = compute_shap_values(model, x, max_sample=10)
    assert result.values.shape[0] == 10


def test_compute_global_importance_orders_descending(shap_result) -> None:
    """A importância global vem ordenada de forma decrescente e não-negativa."""
    importance = compute_global_importance(shap_result)

    assert set(importance.index) == set(shap_result.feature_names)
    assert (importance.to_numpy() >= 0).all()
    assert list(importance) == sorted(importance, reverse=True)


def test_compute_shap_values_unwraps_calibrated_estimator(synthetic_raw: pl.DataFrame) -> None:
    """Com um modelo calibrado, o SHAP é calculado sobre o estimador base."""
    pdf = synthetic_raw.to_pandas().head(60)
    x = pdf[col.FEATURES]
    y = pdf[col.TARGET].astype(int)
    model = build_model("logistic", calibrate=True)
    model.fit(x, y)

    result = compute_shap_values(model, x, max_sample=60)
    assert result.values.shape[0] == len(x)
