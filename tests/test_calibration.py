"""Testes da análise de calibração."""

from __future__ import annotations

import numpy as np

from evaluation.calibration import compute_calibration_curve


def test_calibration_curve_shapes_match() -> None:
    """As curvas de probabilidade observada e prevista têm o mesmo tamanho."""
    rng = np.random.default_rng(0)
    y_proba = rng.uniform(0, 1, 500)
    y_true = rng.binomial(1, y_proba)
    result = compute_calibration_curve(y_true, y_proba, n_bins=10)
    assert result.prob_true.shape == result.prob_pred.shape


def test_well_calibrated_has_low_brier() -> None:
    """Probabilidades bem calibradas produzem Brier score baixo."""
    rng = np.random.default_rng(1)
    y_proba = rng.uniform(0, 1, 2000)
    y_true = rng.binomial(1, y_proba)
    result = compute_calibration_curve(y_true, y_proba)
    assert result.brier_score < 0.2
