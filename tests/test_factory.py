"""Testes da fábrica de pipelines de modelo."""

from __future__ import annotations

import pytest
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from models.factory import build_model, list_available_models


def test_list_available_models() -> None:
    """A fábrica anuncia os dois modelos suportados."""
    assert list_available_models() == ["logistic", "lightgbm"]


@pytest.mark.smoke
def test_build_model_logistic_without_calibration() -> None:
    """O modelo logístico sem calibração expõe o estimador diretamente."""
    model = build_model("logistic", calibrate=False)
    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["model"], LogisticRegression)


def test_build_model_lightgbm_without_calibration() -> None:
    """O LightGBM é construído sem escala (passthrough) e sem calibração."""
    model = build_model("lightgbm", calibrate=False)
    assert isinstance(model.named_steps["model"], LGBMClassifier)


def test_build_model_with_calibration_wraps_estimator() -> None:
    """Com calibração ativada, o estimador final é um CalibratedClassifierCV."""
    model = build_model("logistic", calibrate=True)
    assert isinstance(model.named_steps["model"], CalibratedClassifierCV)


def test_build_model_defaults_to_active_model_from_settings() -> None:
    """Sem nome explícito, usa o modelo ativo configurado (lightgbm)."""
    model = build_model()
    assert isinstance(model, Pipeline)


def test_build_model_unknown_name_raises() -> None:
    """Um nome de modelo desconhecido levanta ValueError."""
    with pytest.raises(ValueError, match="desconhecido"):
        build_model("arvore-magica")  # type: ignore[arg-type]
