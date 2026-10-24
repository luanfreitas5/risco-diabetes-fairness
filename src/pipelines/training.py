"""Pipeline de treino: baseline + modelo ativo, com validação cruzada.

Segue a ordem do "senior bar": baseline primeiro (Regressão Logística), depois o
modelo ativo (LightGBM). Reporta a validação cruzada com intervalo de confiança,
ajusta no conjunto de treino e persiste o modelo com metadados de rastreabilidade.
"""

from __future__ import annotations

from pathlib import Path

from config.logging import get_logger
from config.paths import get_paths
from config.settings import get_settings
from data.loader import load_gold
from data.splitter import DataSplit, split_data
from evaluation.evaluator import cross_validate_model
from experiment.tracker import log_cv_run
from models.factory import build_model
from models.persistence import save_model
from utils.hashing import hash_dataframe
from utils.io import save_json
from utils.seed import seed_everything

logger = get_logger(__name__)

MODEL_FILE = "model.joblib"


def run_training() -> Path:
    """Executa o pipeline de treino e persiste o modelo ativo.

    Treina primeiro o baseline (logístico) como benchmark e, em seguida, o
    modelo ativo definido em ``model.active_model``. Avalia ambos por validação
    cruzada e salva o modelo ativo ajustado no conjunto de treino.

    Returns
    -------
    Path
        Caminho do modelo persistido.

    Examples
    --------
    >>> run_training()  # doctest: +SKIP
    """
    logger.info("== Pipeline de treino iniciado ==")
    settings = get_settings()
    seed_everything(settings.random_seed)
    paths = get_paths()
    paths.ensure_dirs()

    processed_df = load_gold()
    split: DataSplit = split_data(processed_df)
    data_hash = hash_dataframe(processed_df)

    # 1) Baseline (benchmark): Regressão Logística.
    logger.info("Avaliando baseline (Regressão Logística) por validação cruzada.")
    baseline = build_model("logistic")
    baseline_cv = cross_validate_model(baseline, split.x_train, split.y_train)
    log_cv_run("baseline-logistic", baseline, baseline_cv, data_hash=data_hash)

    # 2) Modelo ativo (ex.: LightGBM).
    active_name = settings.model.active_model
    logger.info("Avaliando modelo ativo (%s) por validação cruzada.", active_name)
    model = build_model(active_name)
    active_cv = cross_validate_model(model, split.x_train, split.y_train)

    logger.info("Ajustando o modelo ativo no conjunto de treino.")
    model.fit(split.x_train, split.y_train)
    run_id = log_cv_run(
        active_name, model, active_cv, data_hash=data_hash, x_example=split.x_train, fitted=True
    )

    metadata = {
        "active_model": active_name,
        "random_seed": settings.random_seed,
        "baseline_cv": {"summary": baseline_cv.summary(), "mean": baseline_cv.mean},
        "active_cv": {"summary": active_cv.summary(), "mean": active_cv.mean},
        "n_train": len(split.x_train),
        "data_hash": data_hash,
        "mlflow_run_id": run_id,
    }
    model_path = save_model(model, paths.models / MODEL_FILE, metadata=metadata)
    save_json(metadata, paths.metrics / "training_summary.json")

    logger.info(
        "== Treino concluído | baseline %s vs ativo %s ==",
        f"{baseline_cv.mean:.4f}",
        f"{active_cv.mean:.4f}",
    )
    return model_path
