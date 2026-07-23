"""Persistência de modelos treinados com ``joblib``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from config.logging import get_logger
from exceptions.model import ModelPersistenceError

logger = get_logger(__name__)


def save_model(model: Pipeline, path: Path, *, metadata: dict[str, Any] | None = None) -> Path:
    """Salva um pipeline de modelo em disco (com metadados opcionais).

    Parameters
    ----------
    model : Pipeline
        Pipeline treinado a persistir.
    path : Path
        Caminho de saída (``.joblib``).
    metadata : dict[str, Any], optional
        Metadados de rastreabilidade (hash dos dados, params, métricas).

    Returns
    -------
    Path
        O caminho gravado.

    Raises
    ------
    ModelPersistenceError
        Se ocorrer falha ao salvar.

    Examples
    --------
    >>> save_model(model, Path("models/model.joblib"))  # doctest: +SKIP
    """
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "metadata": metadata or {}}, path)
    except (OSError, ValueError) as exc:
        logger.exception("Falha ao salvar o modelo em %s", path)
        raise ModelPersistenceError(f"Não foi possível salvar o modelo: {path}") from exc
    logger.info("Modelo salvo: %s", path.name)
    return path


def load_model(path: Path) -> tuple[Pipeline, dict[str, Any]]:
    """Carrega um pipeline de modelo do disco.

    Parameters
    ----------
    path : Path
        Caminho do artefato ``.joblib``.

    Returns
    -------
    tuple[Pipeline, dict[str, Any]]
        O pipeline e seus metadados.

    Raises
    ------
    ModelPersistenceError
        Se o arquivo não existir ou não puder ser carregado.

    Examples
    --------
    >>> model, meta = load_model(Path("models/model.joblib"))  # doctest: +SKIP
    """
    path = Path(path)
    if not path.exists():
        raise ModelPersistenceError(f"Modelo não encontrado: {path}")
    try:
        payload = joblib.load(path)
    except (OSError, ValueError) as exc:
        logger.exception("Falha ao carregar o modelo de %s", path)
        raise ModelPersistenceError(f"Não foi possível carregar o modelo: {path}") from exc
    logger.info("Modelo carregado: %s", path.name)
    return payload["model"], payload.get("metadata", {})
