"""Testes comportamentais do modelo (expectativas direcionais e de sanidade)."""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from constants import columns as col
from models.factory import build_model


@pytest.fixture(scope="module")
def fitted_logistic(synthetic_raw: pl.DataFrame):
    """Treina um baseline logístico (sem calibração) nos dados sintéticos."""
    pdf = synthetic_raw.to_pandas()
    x = pdf[col.FEATURES]
    y = pdf[col.TARGET].astype(int)
    model = build_model("logistic", calibrate=False)
    model.fit(x, y)
    return model


@pytest.mark.smoke
def test_predict_proba_is_bounded(fitted_logistic, synthetic_raw: pl.DataFrame) -> None:
    """As probabilidades previstas ficam no intervalo [0, 1]."""
    x = synthetic_raw.to_pandas()[col.FEATURES]
    proba = fitted_logistic.predict_proba(x)[:, 1]
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0


@pytest.mark.ml
def test_risk_factors_increase_predicted_risk(fitted_logistic) -> None:
    """Expectativa direcional: mais fatores de risco não reduzem o risco previsto.

    Como os dados sintéticos foram gerados com risco crescente em HighBP,
    HighChol e IMC, um paciente de alto risco deve ter probabilidade maior que
    um de baixo risco.
    """

    low_risk = dict.fromkeys(col.FEATURES, 0.0)
    low_risk.update({"BMI": 20.0, "GenHlth": 1.0, "Age": 3.0, "Education": 6.0, "Income": 8.0})

    high_risk = dict(low_risk)
    high_risk.update({"HighBP": 1.0, "HighChol": 1.0, "BMI": 45.0})

    frame = pd.DataFrame([low_risk, high_risk])[col.FEATURES]
    proba = fitted_logistic.predict_proba(frame)[:, 1]
    assert proba[1] >= proba[0]
