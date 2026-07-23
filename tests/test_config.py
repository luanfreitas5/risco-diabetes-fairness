"""Testes de carregamento e validação da configuração."""

from __future__ import annotations

import pytest

from config.paths import get_paths
from config.settings import get_settings


@pytest.mark.smoke
def test_settings_load_and_validate() -> None:
    """A configuração carrega e valida os campos esperados."""
    settings = get_settings()
    assert settings.data.target == "Diabetes_binary"
    assert 0 < settings.evaluation.decision_threshold < 1
    assert settings.model.active_model in {"logistic", "lightgbm"}


def test_sensitive_features_configured() -> None:
    """Os atributos sensíveis incluem sexo e faixa etária derivada."""
    settings = get_settings()
    assert "Sex" in settings.sensitive_features
    assert "AgeBand" in settings.sensitive_features


def test_paths_resolve_under_project_root() -> None:
    """Todos os caminhos são resolvidos sob a raiz do projeto."""
    paths = get_paths()
    assert paths.data_raw.is_absolute()
    assert paths.root in paths.figures.parents
