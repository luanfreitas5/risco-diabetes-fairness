"""Fábrica de pipelines de modelo (padrão Factory).

Cada modelo é um ``Pipeline`` do scikit-learn que acopla o pré-processamento
adequado ao estimador. Opcionalmente, envolve o estimador em um
``CalibratedClassifierCV`` para produzir probabilidades confiáveis — essenciais
para uma triagem de risco.
"""

from __future__ import annotations

from typing import Literal

from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config.logging import get_logger
from config.settings import get_settings
from preprocessing.pipeline import build_preprocessor

logger = get_logger(__name__)

ModelName = Literal["logistic", "lightgbm"]


def list_available_models() -> list[str]:
    """Lista os identificadores de modelos disponíveis na fábrica.

    Returns
    -------
    list[str]
        Nomes dos modelos suportados.

    Examples
    --------
    >>> list_available_models()
    ['logistic', 'lightgbm']
    """
    return ["logistic", "lightgbm"]


def _build_estimator(name: ModelName, seed: int) -> tuple[object, bool]:
    """Constrói o estimador base e indica se precisa de escala.

    Parameters
    ----------
    name : {"logistic", "lightgbm"}
        Identificador do modelo.
    seed : int
        Semente de reprodutibilidade.

    Returns
    -------
    tuple[object, bool]
        O estimador e um booleano ``scale`` (True se exige padronização).

    Raises
    ------
    ValueError
        Se o nome do modelo for desconhecido.
    """
    params = get_settings().model
    if name == "logistic":
        estimator = LogisticRegression(
            C=params.logistic.C,
            max_iter=params.logistic.max_iter,
            class_weight=params.logistic.class_weight,
            solver=params.logistic.solver,
            random_state=seed,
        )
        return estimator, True
    if name == "lightgbm":
        estimator = LGBMClassifier(
            n_estimators=params.lightgbm.n_estimators,
            learning_rate=params.lightgbm.learning_rate,
            num_leaves=params.lightgbm.num_leaves,
            max_depth=params.lightgbm.max_depth,
            min_child_samples=params.lightgbm.min_child_samples,
            subsample=params.lightgbm.subsample,
            colsample_bytree=params.lightgbm.colsample_bytree,
            reg_lambda=params.lightgbm.reg_lambda,
            class_weight=params.lightgbm.class_weight,
            n_jobs=params.lightgbm.n_jobs,
            random_state=seed,
            verbose=-1,
        )
        return estimator, False
    raise ValueError(f"Modelo desconhecido: {name!r}. Disponíveis: {list_available_models()}")


def build_model(name: ModelName | None = None, *, calibrate: bool | None = None) -> Pipeline:
    """Constrói o pipeline completo de um modelo (pré-processamento + estimador).

    Parameters
    ----------
    name : {"logistic", "lightgbm"}, optional
        Modelo a construir. Se ``None``, usa ``model.active_model`` do config.
    calibrate : bool, optional
        Se ``True``, envolve o estimador em calibração de probabilidades.
        Se ``None``, usa ``model.calibration.enabled`` do config.

    Returns
    -------
    Pipeline
        Pipeline scikit-learn pronto para ``fit``/``predict``/``predict_proba``.

    Examples
    --------
    >>> model = build_model("logistic", calibrate=False)  # doctest: +SKIP
    """
    settings = get_settings()
    model_name: ModelName = name or settings.model.active_model
    seed = settings.random_seed
    should_calibrate = settings.model.calibration.enabled if calibrate is None else calibrate

    estimator, scale = _build_estimator(model_name, seed)
    preprocessor = build_preprocessor(scale=scale)

    if should_calibrate:
        estimator = CalibratedClassifierCV(
            estimator,
            method=settings.model.calibration.method,
            cv=settings.model.calibration.cv,
        )
        logger.info(
            "Calibração ativada (%s, cv=%d).",
            settings.model.calibration.method,
            settings.model.calibration.cv,
        )

    logger.info("Modelo construído: %s (escala=%s).", model_name, scale)
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])
