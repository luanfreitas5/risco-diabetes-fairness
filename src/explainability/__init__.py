"""Interpretabilidade do modelo com SHAP.

Módulos
-------
shap_explainer
    Calcula valores SHAP e a importância global das features.
"""

from explainability.shap_explainer import (
    compute_global_importance,
    compute_shap_values,
)

__all__ = ["compute_global_importance", "compute_shap_values"]
