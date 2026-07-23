"""Pipeline de avaliação: métricas, calibração, curvas e interpretabilidade.

Carrega o modelo persistido, reconstrói o mesmo split (semente fixa) e avalia no
holdout: métricas de classificação, calibração, curvas ROC/PR e importância SHAP.
Salva as figuras em ``reports/figures`` e as métricas em ``reports/metrics``.
"""

from __future__ import annotations

from config.logging import get_logger
from config.paths import get_paths
from data.loader import load_processed
from data.splitter import split_data
from evaluation.calibration import compute_calibration_curve
from evaluation.evaluator import evaluate_holdout
from explainability.shap_explainer import (
    compute_global_importance,
    compute_shap_values,
)
from models.persistence import load_model
from utils.io import save_json
from utils.seed import seed_everything
from visualization.plots import (
    plot_calibration_curve,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_shap_importance,
    save_figure,
)

logger = get_logger(__name__)

MODEL_FILE = "model.joblib"


def run_evaluation() -> dict[str, float]:
    """Executa o pipeline de avaliação e gera figuras e métricas.

    Returns
    -------
    dict[str, float]
        Métricas de holdout do modelo.

    Examples
    --------
    >>> run_evaluation()  # doctest: +SKIP
    """
    logger.info("== Pipeline de avaliação iniciado ==")
    paths = get_paths()
    paths.ensure_dirs()
    seed_everything()

    model, _ = load_model(paths.models / MODEL_FILE)
    df = load_processed()
    split = split_data(df)

    y_proba = model.predict_proba(split.x_test)[:, 1]

    # Métricas de holdout.
    metrics = evaluate_holdout(model, split.x_test, split.y_test)
    save_json(metrics, paths.metrics / "holdout_metrics.json")

    # Curvas ROC e PR.
    save_figure(plot_roc_curve(split.y_test, y_proba), "roc_curve")
    save_figure(plot_precision_recall_curve(split.y_test, y_proba), "precision_recall_curve")

    # Calibração.
    calibration = compute_calibration_curve(split.y_test, y_proba)
    save_figure(plot_calibration_curve(calibration), "calibration_curve")

    # Interpretabilidade (SHAP).
    try:
        shap_result = compute_shap_values(model, split.x_test)
        importance = compute_global_importance(shap_result)
        save_figure(plot_shap_importance(importance), "shap_importance")
        save_json(importance.to_dict(), paths.metrics / "shap_importance.json")
    except (ValueError, TypeError, RuntimeError) as exc:  # SHAP pode falhar em alguns setups
        logger.warning("Cálculo de SHAP ignorado (%s).", exc)

    logger.info("== Avaliação concluída ==")
    return metrics
