"""Testes da avaliação de modelos (validação cruzada e holdout)."""

from __future__ import annotations

import polars as pl
import pytest

from constants import columns as col
from evaluation.evaluator import CrossValResult, cross_validate_model, evaluate_holdout
from models.factory import build_model


@pytest.fixture(scope="module")
def fitted_model(synthetic_raw: pl.DataFrame):
    """Modelo logístico (sem calibração) treinado nos dados sintéticos."""
    pdf = synthetic_raw.to_pandas()
    x = pdf[col.FEATURES]
    y = pdf[col.TARGET].astype(int)
    model = build_model("logistic", calibrate=False)
    model.fit(x, y)
    return model, x, y


@pytest.mark.smoke
def test_cross_validate_model_reports_mean_and_ci(synthetic_raw: pl.DataFrame) -> None:
    """A validação cruzada retorna média, desvio e IC95 consistentes."""
    pdf = synthetic_raw.to_pandas()
    x = pdf[col.FEATURES]
    y = pdf[col.TARGET].astype(int)
    model = build_model("logistic", calibrate=False)

    result = cross_validate_model(model, x, y, scoring="roc_auc", cv_folds=3)

    assert isinstance(result, CrossValResult)
    assert len(result.scores) == 3
    assert result.mean == pytest.approx(sum(result.scores) / 3)
    assert result.ci95 >= 0
    assert "roc_auc" in result.summary()


def test_cross_validate_model_uses_settings_defaults(synthetic_raw: pl.DataFrame) -> None:
    """Sem parâmetros explícitos, usa a métrica e os folds do config."""
    pdf = synthetic_raw.to_pandas()
    x = pdf[col.FEATURES]
    y = pdf[col.TARGET].astype(int)
    model = build_model("logistic", calibrate=False)

    result = cross_validate_model(model, x, y)
    assert result.metric == "roc_auc"
    assert len(result.scores) == 5


def test_evaluate_holdout_returns_expected_keys(fitted_model) -> None:
    """As métricas de holdout contêm as chaves esperadas e ficam em [0, 1]."""
    model, x, y = fitted_model
    metrics = evaluate_holdout(model, x, y)
    for key in ("roc_auc", "pr_auc", "recall", "precision", "f1", "brier_score"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_evaluate_holdout_respects_custom_threshold(fitted_model) -> None:
    """Um limiar mais baixo não pode reduzir o recall (mesma invariante de negócio)."""
    model, x, y = fitted_model
    metrics_low = evaluate_holdout(model, x, y, threshold=0.1)
    metrics_high = evaluate_holdout(model, x, y, threshold=0.9)
    assert metrics_low["recall"] >= metrics_high["recall"]
