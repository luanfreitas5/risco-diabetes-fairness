"""Testes de integração dos pipelines de orquestração.

Cada pipeline é testado isoladamente: ``get_paths``/``load_processed``/
``load_model`` são substituídos (monkeypatch) por versões que apontam para um
diretório temporário e para os dados sintéticos da sessão, evitando tocar os
diretórios reais do projeto. Etapas pesadas já cobertas por testes dedicados
(validação cruzada completa, SHAP) são substituídas por versões rápidas para
manter o pipeline de testes ágil; a orquestração em si (ordem das chamadas,
arquivos gravados, valores de retorno) roda de ponta a ponta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from config.paths import ProjectPaths
from constants import columns as col
from evaluation.evaluator import CrossValResult
from explainability.shap_explainer import ShapResult
from features.engineering import engineer_features
from models.factory import build_model
from models.persistence import save_model
from pipelines import evaluation as evaluation_pipeline
from pipelines import fairness as fairness_pipeline
from pipelines import preprocessing as preprocessing_pipeline
from pipelines import training as training_pipeline
from visualization import plots


@pytest.fixture(autouse=True)
def _redirect_save_figure(fake_paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch) -> None:
    """Garante que ``save_figure`` grave em um diretório temporário, nunca no real.

    ``save_figure`` resolve ``get_paths()`` a partir do próprio módulo
    ``visualization.plots``, independente do ``get_paths`` do pipeline chamador.
    """
    monkeypatch.setattr(plots, "get_paths", lambda: fake_paths)


@pytest.fixture
def fitted_model(synthetic_raw: pl.DataFrame):
    """Modelo logístico ajustado no dataset processado sintético (rápido, sem calibração)."""
    processed = engineer_features(synthetic_raw)
    pdf = processed.to_pandas()
    model = build_model("logistic", calibrate=False)
    model.fit(pdf[col.FEATURES], pdf[col.TARGET].astype(int))
    return model, processed


@pytest.mark.integration
def test_run_preprocessing_writes_processed_parquet_and_manifest(
    synthetic_raw: pl.DataFrame, fake_paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O pipeline de pré-processamento valida, persiste e registra o manifesto."""
    monkeypatch.setattr(preprocessing_pipeline, "get_paths", lambda: fake_paths)
    monkeypatch.setattr(preprocessing_pipeline, "load_raw", lambda raw_file=None: synthetic_raw)

    output = preprocessing_pipeline.run_preprocessing()

    assert output == fake_paths.data_processed / preprocessing_pipeline.PROCESSED_FILE
    assert output.exists()
    assert (fake_paths.data_processed / "manifest.json").exists()
    assert "AgeBand" in pl.read_parquet(output).columns


@pytest.mark.integration
def test_run_training_persists_active_model_and_summary(
    synthetic_raw: pl.DataFrame, fake_paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O pipeline de treino avalia baseline e ativo, e persiste o modelo ajustado."""
    processed = engineer_features(synthetic_raw)
    fake_cv = CrossValResult(metric="roc_auc", scores=[0.8, 0.81], mean=0.805, std=0.005, ci95=0.01)

    monkeypatch.setattr(training_pipeline, "get_paths", lambda: fake_paths)
    monkeypatch.setattr(training_pipeline, "load_processed", lambda: processed)
    monkeypatch.setattr(training_pipeline, "cross_validate_model", lambda *a, **k: fake_cv)

    model_path = training_pipeline.run_training()

    assert model_path == fake_paths.models / training_pipeline.MODEL_FILE
    assert model_path.exists()
    assert (fake_paths.metrics / "training_summary.json").exists()


@pytest.mark.integration
def test_run_evaluation_generates_metrics_and_figures(
    fitted_model, fake_paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O pipeline de avaliação gera métricas de holdout e as figuras esperadas."""
    model, processed = fitted_model
    fake_shap = ShapResult(values=np.zeros((1, 1)), feature_names=["BMI"], data=np.zeros((1, 1)))

    monkeypatch.setattr(evaluation_pipeline, "get_paths", lambda: fake_paths)
    monkeypatch.setattr(evaluation_pipeline, "load_processed", lambda: processed)
    monkeypatch.setattr(evaluation_pipeline, "load_model", lambda path: (model, {}))
    monkeypatch.setattr(evaluation_pipeline, "compute_shap_values", lambda model, x: fake_shap)
    monkeypatch.setattr(
        evaluation_pipeline,
        "compute_global_importance",
        lambda result: pd.Series({"BMI": 0.5, "HighBP": 0.3}),
    )

    metrics = evaluation_pipeline.run_evaluation()

    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert (fake_paths.metrics / "holdout_metrics.json").exists()
    assert (fake_paths.metrics / "shap_importance.json").exists()
    for name in ("roc_curve", "precision_recall_curve", "calibration_curve", "shap_importance"):
        assert (fake_paths.figures / f"{name}.png").exists()


@pytest.mark.integration
def test_run_evaluation_survives_shap_failure(
    fitted_model, fake_paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Uma falha no cálculo de SHAP é registrada, mas não interrompe a avaliação."""
    model, processed = fitted_model

    def _raise_value_error(*_args: object, **_kwargs: object) -> None:
        raise ValueError("SHAP indisponível neste setup")

    monkeypatch.setattr(evaluation_pipeline, "get_paths", lambda: fake_paths)
    monkeypatch.setattr(evaluation_pipeline, "load_processed", lambda: processed)
    monkeypatch.setattr(evaluation_pipeline, "load_model", lambda path: (model, {}))
    monkeypatch.setattr(evaluation_pipeline, "compute_shap_values", _raise_value_error)

    metrics = evaluation_pipeline.run_evaluation()

    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert not (fake_paths.metrics / "shap_importance.json").exists()


@pytest.mark.integration
def test_run_fairness_audit_generates_report_and_figures(
    fitted_model, fake_paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O pipeline de auditoria de justiça gera o relatório e as figuras por atributo."""
    model, processed = fitted_model

    monkeypatch.setattr(fairness_pipeline, "get_paths", lambda: fake_paths)
    monkeypatch.setattr(fairness_pipeline, "load_processed", lambda: processed)
    monkeypatch.setattr(fairness_pipeline, "load_model", lambda path: (model, {}))

    report = fairness_pipeline.run_fairness_audit()

    assert len(report.by_group) > 0
    assert (fake_paths.metrics / "fairness_report.json").exists()
    for feature in report.by_group:
        assert (fake_paths.figures / f"fairness_recall_{feature}.png").exists()


@pytest.mark.smoke
def test_model_persisted_for_pipeline_is_reloadable(fake_paths: ProjectPaths) -> None:
    """Sanidade: um modelo persistido no caminho usado pelo pipeline pode ser recarregado."""
    model = build_model("logistic", calibrate=False)
    path = save_model(model, fake_paths.models / training_pipeline.MODEL_FILE)
    assert path.exists()
