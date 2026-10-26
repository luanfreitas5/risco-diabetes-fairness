"""Rastreamento de experimentos com MLflow.

Contrato de cada run: tags de linhagem (``git_sha``, ``data_hash``, ``seed``) no run
pai **e** nos filhos, parâmetros do modelo e da CV, um run filho aninhado por fold,
métricas ``cv_<métrica>_mean``/``_std`` no pai e o modelo (ajustado em todo o treino)
com assinatura e exemplo de entrada. O backend é sempre um banco SQLite, nunca o
*file store* (``./mlruns``), que está em modo de manutenção.
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 - uso restrito a comandos git com argumentos fixos
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn as mlflow_sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from config.logging import get_logger
from config.paths import PROJECT_ROOT
from config.settings import get_settings
from evaluation.evaluator import CrossValResult

logger = get_logger(__name__)

_SQLITE_PREFIX = "sqlite:///"
_EXAMPLE_ROWS = 5
_SIGNATURE_ROWS = 100


def _run_git(*args: str) -> str:
    """Executa um comando Git na raiz do projeto e retorna a saída padrão.

    Parameters
    ----------
    *args : str
        Argumentos do comando Git (ex.: ``"rev-parse", "HEAD"``).

    Returns
    -------
    str
        Saída padrão do comando.
    """
    return subprocess.run(  # noqa: S603  # nosec B603 B607 - argumentos fixos, sem entrada do usuário
        ["git", *args],  # noqa: S607 - git resolvido pelo PATH do ambiente
        capture_output=True,
        text=True,
        check=True,
        cwd=PROJECT_ROOT,
    ).stdout.strip()


def compute_git_sha(*, require_clean_tree: bool = True) -> str:
    """Retorna o SHA do ``HEAD``; recusa árvore suja para não registrar um SHA mentiroso.

    Parameters
    ----------
    require_clean_tree : bool, optional
        Se ``True``, levanta erro quando há alterações não commitadas, by default True.

    Returns
    -------
    str
        SHA completo do commit atual.

    Raises
    ------
    RuntimeError
        Se a árvore Git estiver suja e ``require_clean_tree`` for ``True``.

    Examples
    --------
    >>> compute_git_sha(require_clean_tree=False)  # doctest: +SKIP
    """
    if require_clean_tree and _run_git("status", "--porcelain"):
        raise RuntimeError("Árvore Git suja: faça commit antes de registrar o experimento.")
    return _run_git("rev-parse", "HEAD")


def configure_tracking() -> None:
    """Aponta o MLflow para o banco SQLite configurado e seleciona o experimento.

    Examples
    --------
    >>> configure_tracking()  # doctest: +SKIP
    """
    experiment = get_settings().experiment
    uri = os.environ.get("MLFLOW_TRACKING_URI") or experiment.tracking_uri
    if uri.startswith(_SQLITE_PREFIX):
        db_path = Path(uri.removeprefix(_SQLITE_PREFIX))
        if not db_path.is_absolute():
            db_path = PROJECT_ROOT / db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        uri = f"{_SQLITE_PREFIX}{db_path.as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment.experiment_name)


def log_cv_run(
    name: str,
    model: Pipeline,
    cv_result: CrossValResult,
    *,
    data_hash: str,
    x_example: pd.DataFrame | None = None,
    fitted: bool = False,
) -> str | None:
    """Registra um modelo avaliado por CV: run pai + um run filho por fold.

    Parameters
    ----------
    name : str
        Nome do run pai (ex.: ``"lightgbm"``).
    model : Pipeline
        Pipeline avaliado. Se ``fitted`` for ``True``, é registrado como artefato.
    cv_result : CrossValResult
        Resultado da validação cruzada (scores por fold, média e desvio).
    data_hash : str
        Hash SHA-256 do dataset gold usado (igual ao do manifesto).
    x_example : pd.DataFrame, optional
        Amostra de entrada para inferir a assinatura do modelo.
    fitted : bool, optional
        Se o ``model`` já foi ajustado em todo o treino e deve ser registrado,
        by default False.

    Returns
    -------
    str | None
        ``run_id`` do run pai, ou ``None`` se o rastreamento estiver desabilitado.

    Raises
    ------
    RuntimeError
        Se exigir árvore limpa e houver alterações não commitadas.

    Examples
    --------
    >>> run_id = log_cv_run("lightgbm", model, cv, data_hash="ab12")  # doctest: +SKIP
    """
    settings = get_settings()
    if not settings.experiment.enabled:
        logger.info("Rastreamento MLflow desabilitado; run '%s' não registrado.", name)
        return None

    configure_tracking()
    tags = {
        "git_sha": compute_git_sha(require_clean_tree=settings.experiment.require_clean_tree),
        "data_hash": data_hash,
        "seed": str(settings.random_seed),
    }
    metric = cv_result.metric
    run_id: str | None = None
    with mlflow.start_run(run_name=name, tags=tags) as parent:
        params: dict[str, Any] = {
            **model.get_params(deep=False),
            "cv": "StratifiedKFold",
            "n_splits": len(cv_result.scores),
        }
        mlflow.log_params({key: str(value)[:500] for key, value in params.items()})
        for fold, score in enumerate(cv_result.scores):
            with mlflow.start_run(
                run_name=f"{name}-fold{fold}", nested=True, tags=tags | {"fold": str(fold)}
            ):
                mlflow.log_metric(f"val_{metric}", score)
        mlflow.log_metrics({f"cv_{metric}_mean": cv_result.mean, f"cv_{metric}_std": cv_result.std})
        if fitted and x_example is not None:
            sample = x_example.head(_SIGNATURE_ROWS)
            signature = infer_signature(sample, model.predict_proba(sample)[:, 1])
            mlflow_sklearn.log_model(
                model,
                name="model",
                signature=signature,
                input_example=x_example.head(_EXAMPLE_ROWS),
                serialization_format=mlflow_sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
        run_id = parent.info.run_id
    logger.info(
        "Run MLflow registrado | %s | %s=%.4f ± %.4f | run_id=%s",
        name,
        metric,
        cv_result.mean,
        cv_result.std,
        run_id,
    )
    return run_id
