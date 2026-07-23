"""Testes de persistência de modelos (joblib)."""

from __future__ import annotations

from pathlib import Path

import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from exceptions.model import ModelPersistenceError
from models.persistence import load_model, save_model


def _dummy_pipeline() -> Pipeline:
    return Pipeline([("model", LogisticRegression())])


@pytest.mark.smoke
def test_save_and_load_model_roundtrip(tmp_path: Path) -> None:
    """Salvar e carregar um modelo preserva o pipeline e os metadados."""
    model = _dummy_pipeline()
    metadata = {"active_model": "logistic", "n_train": 100}

    path = save_model(model, tmp_path / "model.joblib", metadata=metadata)
    assert path.exists()

    loaded_model, loaded_metadata = load_model(path)
    assert isinstance(loaded_model, Pipeline)
    assert loaded_metadata == metadata


def test_save_model_without_metadata_defaults_to_empty_dict(tmp_path: Path) -> None:
    """Sem metadados explícitos, o dicionário salvo fica vazio."""
    path = save_model(_dummy_pipeline(), tmp_path / "sub" / "model.joblib")
    _, metadata = load_model(path)
    assert metadata == {}


def test_load_model_missing_file_raises(tmp_path: Path) -> None:
    """Carregar um caminho inexistente levanta ModelPersistenceError."""
    with pytest.raises(ModelPersistenceError, match="não encontrado"):
        load_model(tmp_path / "inexistente.joblib")


def test_save_model_wraps_os_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Uma falha do joblib ao salvar é envolvida em ModelPersistenceError."""

    def _raise_os_error(*_args: object, **_kwargs: object) -> None:
        raise OSError("disco cheio")

    monkeypatch.setattr(joblib, "dump", _raise_os_error)
    with pytest.raises(ModelPersistenceError, match="Não foi possível salvar"):
        save_model(_dummy_pipeline(), tmp_path / "model.joblib")


def test_load_model_wraps_os_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Uma falha do joblib ao carregar é envolvida em ModelPersistenceError."""
    path = tmp_path / "model.joblib"
    path.write_bytes(b"not-a-real-joblib-file")

    def _raise_value_error(*_args: object, **_kwargs: object) -> None:
        raise ValueError("payload corrompido")

    monkeypatch.setattr(joblib, "load", _raise_value_error)
    with pytest.raises(ModelPersistenceError, match="Não foi possível carregar"):
        load_model(path)
