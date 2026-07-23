"""Avaliação rigorosa de modelos: validação cruzada, holdout e calibração.

Módulos
-------
evaluator
    Validação cruzada com intervalo de confiança e avaliação no holdout.
calibration
    Análise de calibração de probabilidades (curva de confiabilidade, Brier).
"""

from evaluation.calibration import compute_calibration_curve
from evaluation.evaluator import (
    CrossValResult,
    cross_validate_model,
    evaluate_holdout,
)

__all__ = [
    "CrossValResult",
    "compute_calibration_curve",
    "cross_validate_model",
    "evaluate_holdout",
]
