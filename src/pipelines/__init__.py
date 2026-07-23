"""Pipelines ponta a ponta que orquestram os módulos do projeto.

Módulos
-------
preprocessing
    Pipeline de preparação: carrega, valida, engenharia de features e persiste.
training
    Pipeline de treino: CV, ajuste no treino e persistência do modelo.
evaluation
    Pipeline de avaliação: métricas, calibração, SHAP e figuras.
fairness
    Pipeline de auditoria de justiça por subgrupo.
"""

from pipelines.evaluation import run_evaluation
from pipelines.fairness import run_fairness_audit
from pipelines.preprocessing import run_preprocessing
from pipelines.training import run_training

__all__ = [
    "run_evaluation",
    "run_fairness_audit",
    "run_preprocessing",
    "run_training",
]
