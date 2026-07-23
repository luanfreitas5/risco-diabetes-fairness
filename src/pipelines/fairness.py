"""Pipeline de auditoria de justiça (fairness) por subgrupo.

Carrega o modelo, reconstrói o split (semente fixa) e audita o desempenho por
subgrupo sensível no holdout, gerando figuras de disparidade e um relatório JSON.
"""

from __future__ import annotations

from config.logging import get_logger
from config.paths import get_paths
from data.loader import load_processed
from data.splitter import split_data
from fairness.audit import FairnessReport, audit_fairness
from models.persistence import load_model
from utils.io import save_json
from utils.seed import seed_everything
from visualization.plots import plot_fairness_disparity, save_figure

logger = get_logger(__name__)

MODEL_FILE = "model.joblib"


def run_fairness_audit() -> FairnessReport:
    """Executa a auditoria de justiça e gera o relatório e as figuras.

    Returns
    -------
    FairnessReport
        Relatório com métricas por subgrupo e disparidades.

    Examples
    --------
    >>> run_fairness_audit()  # doctest: +SKIP
    """
    logger.info("== Pipeline de auditoria de justiça iniciado ==")
    paths = get_paths()
    paths.ensure_dirs()
    seed_everything()

    model, _ = load_model(paths.models / MODEL_FILE)
    df = load_processed()
    split = split_data(df)

    y_proba = model.predict_proba(split.x_test)[:, 1]
    report = audit_fairness(split.y_test, y_proba, split.sensitive_test)

    for feature, by_group in report.by_group.items():
        save_figure(
            plot_fairness_disparity(by_group, feature, metric="recall"),
            f"fairness_recall_{feature}",
        )

    save_json(report.to_dict(), paths.metrics / "fairness_report.json")
    logger.info("== Auditoria de justiça concluída ==")
    return report
