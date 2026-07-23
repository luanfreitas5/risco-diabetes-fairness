"""Testes das funções de plotagem e do tema visual compartilhado."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from config.paths import ProjectPaths
from evaluation.calibration import compute_calibration_curve
from visualization import plots
from visualization.theme import CATEGORICAL, PALETTE, apply_theme


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(3)


def test_apply_theme_sets_rcparams() -> None:
    """Aplicar o tema define os parâmetros de figura esperados."""
    apply_theme()
    assert plt.rcParams["axes.titleweight"] == "bold"
    assert plt.rcParams["savefig.dpi"] == 300


def test_palette_and_categorical_are_consistent() -> None:
    """A ordem categórica reutiliza exatamente as cores da paleta nomeada."""
    assert set(CATEGORICAL).issubset(set(PALETTE.values()))


def test_plot_roc_curve_returns_figure(rng: np.random.Generator) -> None:
    """A curva ROC produz uma figura válida do matplotlib."""
    y_true = rng.integers(0, 2, 100)
    y_proba = rng.uniform(0, 1, 100)
    fig = plots.plot_roc_curve(y_true, y_proba)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_precision_recall_curve_returns_figure(rng: np.random.Generator) -> None:
    """A curva PR produz uma figura válida do matplotlib."""
    y_true = rng.integers(0, 2, 100)
    y_proba = rng.uniform(0, 1, 100)
    fig = plots.plot_precision_recall_curve(y_true, y_proba)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_calibration_curve_returns_figure(rng: np.random.Generator) -> None:
    """A curva de calibração produz uma figura válida do matplotlib."""
    y_proba = rng.uniform(0, 1, 300)
    y_true = rng.binomial(1, y_proba)
    calibration = compute_calibration_curve(y_true, y_proba)
    fig = plots.plot_calibration_curve(calibration)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_fairness_disparity_returns_figure() -> None:
    """O gráfico de disparidade por subgrupo produz uma figura válida."""
    by_group = pd.DataFrame({"recall": [0.7, 0.5]}, index=pd.Index(["homem", "mulher"], name="Sex"))
    fig = plots.plot_fairness_disparity(by_group, "Sex", metric="recall")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_shap_importance_returns_figure() -> None:
    """A importância SHAP produz um gráfico de barras horizontais válido."""
    importance = pd.Series([0.5, 0.3, 0.1], index=["BMI", "HighBP", "Age"]).sort_values(
        ascending=False
    )
    fig = plots.plot_shap_importance(importance, top_n=2)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_save_figure_writes_png_and_svg(
    fake_paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``save_figure`` grava PNG (300 dpi) e SVG no diretório de figuras."""
    monkeypatch.setattr(plots, "get_paths", lambda: fake_paths)

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    png_path = plots.save_figure(fig, "exemplo")
    plt.close(fig)

    assert png_path.exists()
    assert (fake_paths.figures / "exemplo.svg").exists()
