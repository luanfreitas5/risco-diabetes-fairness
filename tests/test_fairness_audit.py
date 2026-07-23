"""Testes da auditoria de justiça por subgrupo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fairness.audit import FairnessReport, audit_fairness


@pytest.fixture
def audit_inputs() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Rótulos, probabilidades e atributos sensíveis sintéticos."""
    rng = np.random.default_rng(7)
    n = 200
    y_true = rng.integers(0, 2, n)
    y_proba = rng.uniform(0, 1, n)
    sensitive = pd.DataFrame(
        {
            "Sex": rng.integers(0, 2, n),
            "AgeBand": rng.choice(["jovem", "meia-idade", "idoso"], n),
        }
    )
    return y_true, y_proba, sensitive


@pytest.mark.smoke
def test_audit_fairness_reports_all_sensitive_columns(
    audit_inputs: tuple[np.ndarray, np.ndarray, pd.DataFrame],
) -> None:
    """O relatório traz um resultado por atributo sensível informado."""
    y_true, y_proba, sensitive = audit_inputs
    report = audit_fairness(y_true, y_proba, sensitive)

    assert isinstance(report, FairnessReport)
    assert set(report.by_group.keys()) == {"Sex", "AgeBand"}
    assert set(report.disparities.keys()) == {"Sex", "AgeBand"}


def test_audit_fairness_disparities_are_bounded(
    audit_inputs: tuple[np.ndarray, np.ndarray, pd.DataFrame],
) -> None:
    """As disparidades de recall e seleção ficam em [0, 1] (diferença de taxas)."""
    y_true, y_proba, sensitive = audit_inputs
    report = audit_fairness(y_true, y_proba, sensitive)

    for disparities in report.disparities.values():
        assert 0.0 <= disparities["recall_difference"] <= 1.0
        assert 0.0 <= disparities["selection_rate_difference"] <= 1.0


def test_audit_fairness_respects_custom_threshold(
    audit_inputs: tuple[np.ndarray, np.ndarray, pd.DataFrame],
) -> None:
    """Um limiar mais baixo tende a aumentar a taxa de seleção média."""
    y_true, y_proba, sensitive = audit_inputs
    report_low = audit_fairness(y_true, y_proba, sensitive, threshold=0.1)
    report_high = audit_fairness(y_true, y_proba, sensitive, threshold=0.9)

    selection_low = report_low.by_group["Sex"]["selection_rate"].mean()
    selection_high = report_high.by_group["Sex"]["selection_rate"].mean()
    assert selection_low >= selection_high


def test_fairness_report_to_dict_is_json_friendly(
    audit_inputs: tuple[np.ndarray, np.ndarray, pd.DataFrame],
) -> None:
    """``to_dict`` serializa ``by_group`` como dicionário aninhado por subgrupo."""
    y_true, y_proba, sensitive = audit_inputs
    report = audit_fairness(y_true, y_proba, sensitive)
    payload = report.to_dict()

    assert "by_group" in payload
    assert "disparities" in payload
    assert isinstance(payload["by_group"]["Sex"], dict)
