"""Testes do rastreamento de experimentos (MLflow com backend SQLite)."""

from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import get_settings
from evaluation.evaluator import CrossValResult
from experiment import tracker

FAKE_CV = CrossValResult(
    metric="roc_auc", scores=[0.80, 0.82, 0.81], mean=0.81, std=0.01, ci95=0.01
)


@pytest.mark.smoke
def test_log_cv_run_is_noop_when_disabled() -> None:
    """Com o rastreamento desabilitado, nenhum run é criado."""
    assert tracker.log_cv_run("modelo", object(), FAKE_CV, data_hash="abc") is None  # type: ignore[arg-type]


def test_compute_git_sha_rejects_dirty_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Árvore suja impede o registro, pois o SHA não descreveria o código."""
    monkeypatch.setattr(tracker, "_run_git", lambda *args: " M src/main.py")
    with pytest.raises(RuntimeError, match="Árvore Git suja"):
        tracker.compute_git_sha()


def test_compute_git_sha_allows_dirty_tree_when_not_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sem a exigência de árvore limpa, devolve o SHA do HEAD."""
    monkeypatch.setattr(tracker, "_run_git", lambda *args: "abc123")
    assert tracker.compute_git_sha(require_clean_tree=False) == "abc123"


@pytest.mark.integration
def test_log_cv_run_creates_parent_and_fold_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Um run pai com tags de linhagem e um run filho por fold são gravados."""
    mlflow = pytest.importorskip("mlflow")
    experiment = get_settings().experiment
    monkeypatch.setattr(experiment, "enabled", True)
    monkeypatch.setattr(
        experiment, "tracking_uri", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    )
    monkeypatch.setattr(tracker, "compute_git_sha", lambda **kwargs: "sha-teste")

    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline

    model = Pipeline([("model", DummyClassifier())])
    run_id = tracker.log_cv_run("dummy", model, FAKE_CV, data_hash="hash-teste")

    assert run_id is not None
    run = mlflow.get_run(run_id)
    assert run.data.tags["git_sha"] == "sha-teste"
    assert run.data.tags["data_hash"] == "hash-teste"
    assert run.data.metrics["cv_roc_auc_mean"] == pytest.approx(0.81)
    children = mlflow.search_runs(
        filter_string=f"tags.mlflow.parentRunId = '{run_id}'", output_format="list"
    )
    assert len(children) == len(FAKE_CV.scores)
