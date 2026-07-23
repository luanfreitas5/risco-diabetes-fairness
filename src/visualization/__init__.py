"""Utilitários de visualização (matplotlib/seaborn) para relatórios.

Módulos
-------
theme
    Paleta de cores e estilo compartilhados do projeto.
plots
    Funções de plotagem (ROC, PR, calibração, disparidade, importância SHAP).
"""

from visualization.plots import (
    plot_calibration_curve,
    plot_fairness_disparity,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_shap_importance,
)
from visualization.theme import PALETTE, apply_theme

__all__ = [
    "PALETTE",
    "apply_theme",
    "plot_calibration_curve",
    "plot_fairness_disparity",
    "plot_precision_recall_curve",
    "plot_roc_curve",
    "plot_shap_importance",
]
