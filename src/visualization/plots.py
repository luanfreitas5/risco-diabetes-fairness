"""Funções de plotagem para os relatórios de avaliação, justiça e SHAP.

Todas as figuras rotulam eixos, têm título e são salvas em ``reports/figures``
nos formatos ``.png`` (300 dpi) e ``.svg``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from config.paths import get_paths
from evaluation.calibration import CalibrationResult
from visualization.theme import PALETTE, apply_theme


def save_figure(fig: Figure, name: str) -> Path:
    """Salva uma figura em ``reports/figures`` como PNG (300 dpi) e SVG.

    Parameters
    ----------
    fig : Figure
        Figura a salvar.
    name : str
        Nome-base do arquivo (sem extensão).

    Returns
    -------
    Path
        Caminho do PNG gravado.

    Examples
    --------
    >>> save_figure(fig, "roc_curve")  # doctest: +SKIP
    """
    figures_dir = get_paths().figures
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / f"{name}.svg", bbox_inches="tight")
    return png_path


def plot_roc_curve(y_true: ArrayLike, y_proba: ArrayLike) -> Figure:
    """Plota a curva ROC do modelo.

    Parameters
    ----------
    y_true : ArrayLike
        Rótulos verdadeiros (0/1).
    y_proba : ArrayLike
        Probabilidades previstas da classe positiva.

    Returns
    -------
    Figure
        Figura com a curva ROC.
    """
    apply_theme()
    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(
        y_true, y_proba, ax=ax, curve_kwargs={"color": PALETTE["primary"]}
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color=PALETTE["neutral"], label="Aleatório")
    ax.set_title("Curva ROC — Triagem de Risco de Diabetes")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos (recall)")
    ax.legend(loc="lower right")
    return fig


def plot_precision_recall_curve(y_true: ArrayLike, y_proba: ArrayLike) -> Figure:
    """Plota a curva de Precisão-Recall (informativa sob desbalanceamento).

    Parameters
    ----------
    y_true : ArrayLike
        Rótulos verdadeiros (0/1).
    y_proba : ArrayLike
        Probabilidades previstas da classe positiva.

    Returns
    -------
    Figure
        Figura com a curva PR.
    """
    apply_theme()
    fig, ax = plt.subplots(figsize=(7, 6))
    PrecisionRecallDisplay.from_predictions(
        y_true, y_proba, ax=ax, curve_kwargs={"color": PALETTE["accent"]}
    )
    prevalence = float(np.mean(np.asarray(y_true)))
    ax.axhline(
        prevalence, linestyle="--", color=PALETTE["neutral"], label=f"Prevalência={prevalence:.2f}"
    )
    ax.set_title("Curva Precisão-Recall")
    ax.set_xlabel("Recall (sensibilidade)")
    ax.set_ylabel("Precisão")
    ax.legend(loc="upper right")
    return fig


def plot_calibration_curve(result: CalibrationResult) -> Figure:
    """Plota a curva de confiabilidade (calibração de probabilidades).

    Parameters
    ----------
    result : CalibrationResult
        Resultado de :func:`evaluation.calibration.compute_calibration_curve`.

    Returns
    -------
    Figure
        Figura com a curva de calibração.
    """
    apply_theme()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        [0, 1], [0, 1], linestyle="--", color=PALETTE["neutral"], label="Perfeitamente calibrado"
    )
    ax.plot(
        result.prob_pred,
        result.prob_true,
        marker="o",
        color=PALETTE["primary"],
        label=f"Modelo (Brier={result.brier_score:.4f})",
    )
    ax.set_title("Curva de Calibração (Confiabilidade)")
    ax.set_xlabel("Probabilidade média prevista")
    ax.set_ylabel("Fração observada de positivos")
    ax.legend(loc="upper left")
    return fig


def plot_fairness_disparity(
    by_group: pd.DataFrame, feature: str, *, metric: str = "recall"
) -> Figure:
    """Plota uma métrica por subgrupo de um atributo sensível.

    Parameters
    ----------
    by_group : pd.DataFrame
        Métricas por subgrupo (``FairnessReport.by_group[feature]``).
    feature : str
        Nome do atributo sensível (para o título).
    metric : str, optional
        Métrica a plotar, by default ``"recall"``.

    Returns
    -------
    Figure
        Figura de barras da métrica por subgrupo.
    """
    apply_theme()
    fig, ax = plt.subplots(figsize=(8, 5))
    values = by_group[metric]
    ax.bar([str(idx) for idx in values.index], values.to_numpy(), color=PALETTE["primary"])
    ax.set_title(f"{metric} por subgrupo — {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return fig


def plot_shap_importance(importance: pd.Series, *, top_n: int = 15) -> Figure:
    """Plota a importância global das features (média do |SHAP|).

    Parameters
    ----------
    importance : pd.Series
        Importância por feature (``compute_global_importance``).
    top_n : int, optional
        Número de features a exibir, by default 15.

    Returns
    -------
    Figure
        Figura de barras horizontais das features mais importantes.
    """
    apply_theme()
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([str(idx) for idx in top.index], top.to_numpy(), color=PALETTE["accent"])
    ax.set_title("Importância global das features (SHAP)")
    ax.set_xlabel("Média do |valor SHAP|")
    ax.set_ylabel("Feature")
    return fig
