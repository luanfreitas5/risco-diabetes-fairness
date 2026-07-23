"""Testes das métricas de classificação."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from metrics.classification import compute_classification_metrics, recall_at_threshold


@pytest.mark.smoke
def test_perfect_separation_gives_auc_one() -> None:
    """Separação perfeita deve produzir ROC-AUC = 1.0."""
    y_true = [0, 0, 1, 1]
    y_proba = [0.1, 0.2, 0.8, 0.9]
    metrics = compute_classification_metrics(y_true, y_proba, threshold=0.5)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)


def test_recall_at_lower_threshold_is_not_smaller() -> None:
    """Invariante: baixar o limiar não pode reduzir o recall."""
    y_true = [0, 1, 1, 0, 1]
    y_proba = [0.2, 0.55, 0.4, 0.3, 0.7]
    recall_high = recall_at_threshold(y_true, y_proba, 0.6)
    recall_low = recall_at_threshold(y_true, y_proba, 0.3)
    assert recall_low >= recall_high


@given(
    threshold=st.floats(min_value=0.01, max_value=0.99),
)
def test_metrics_are_bounded(threshold: float) -> None:
    """Propriedade: todas as métricas de razão ficam em [0, 1]."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 50)
    # Garante ao menos uma classe de cada para métricas bem definidas.
    y_true[0], y_true[1] = 0, 1
    y_proba = rng.uniform(0, 1, 50)
    metrics = compute_classification_metrics(y_true, y_proba, threshold=threshold)
    for name in ("roc_auc", "pr_auc", "recall", "precision", "f1", "balanced_accuracy"):
        assert 0.0 <= metrics[name] <= 1.0
