"""Rastreamento de experimentos com MLflow (backend SQLite).

Módulos
-------
tracker
    ``log_cv_run``: um run pai por modelo, um run filho aninhado por fold,
    tags de linhagem (``git_sha``, ``data_hash``, ``seed``) e modelo com assinatura.
"""

from experiment.tracker import compute_git_sha, log_cv_run

__all__ = ["compute_git_sha", "log_cv_run"]
