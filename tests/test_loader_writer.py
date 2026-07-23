"""Testes de carregamento (CSV/parquet) e persistência em parquet."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from config.paths import ProjectPaths
from data import loader
from data.writer import write_parquet
from exceptions.data import DataNotFoundError
from features.engineering import engineer_features


@pytest.mark.smoke
def test_load_raw_reads_and_validates_csv(
    synthetic_raw: pl.DataFrame, fake_paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Carrega um CSV bruto válido e retorna o DataFrame validado."""
    fake_paths.data_raw.mkdir(parents=True, exist_ok=True)
    raw_path = fake_paths.data_raw / "raw.csv"
    synthetic_raw.write_csv(raw_path)
    monkeypatch.setattr(loader, "get_paths", lambda: fake_paths)

    df = loader.load_raw("raw.csv")
    assert df.height == synthetic_raw.height


def test_load_raw_missing_file_raises(
    fake_paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Um arquivo bruto inexistente levanta DataNotFoundError."""
    monkeypatch.setattr(loader, "get_paths", lambda: fake_paths)
    with pytest.raises(DataNotFoundError):
        loader.load_raw("inexistente.csv")


def test_load_raw_uses_configured_file_name_when_none_given(
    synthetic_raw: pl.DataFrame, fake_paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sem nome explícito, usa ``data.raw_file`` da configuração."""
    fake_paths.data_raw.mkdir(parents=True, exist_ok=True)
    settings = loader.get_settings()
    synthetic_raw.write_csv(fake_paths.data_raw / settings.data.raw_file)
    monkeypatch.setattr(loader, "get_paths", lambda: fake_paths)

    df = loader.load_raw()
    assert df.height == synthetic_raw.height


def test_load_raw_skips_validation_when_disabled(
    fake_paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Com ``validate=False``, dados fora do contrato ainda são carregados."""
    fake_paths.data_raw.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"ColunaQualquer": [1, 2, 3]}).write_csv(fake_paths.data_raw / "raw.csv")
    monkeypatch.setattr(loader, "get_paths", lambda: fake_paths)

    df = loader.load_raw("raw.csv", validate=False)
    assert "ColunaQualquer" in df.columns


def test_load_processed_reads_valid_parquet(
    synthetic_raw: pl.DataFrame, fake_paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Carrega um parquet processado válido."""
    processed = engineer_features(synthetic_raw)
    fake_paths.data_processed.mkdir(parents=True, exist_ok=True)
    write_parquet(processed, fake_paths.data_processed / "processed.parquet")
    monkeypatch.setattr(loader, "get_paths", lambda: fake_paths)

    df = loader.load_processed("processed.parquet")
    assert "AgeBand" in df.columns


def test_load_processed_missing_file_raises(
    fake_paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Um parquet processado inexistente levanta DataNotFoundError."""
    monkeypatch.setattr(loader, "get_paths", lambda: fake_paths)
    with pytest.raises(DataNotFoundError):
        loader.load_processed("inexistente.parquet")


def test_resolve_raw_path_uses_configured_file_name(
    fake_paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``resolve_raw_path`` retorna o caminho absoluto do arquivo configurado."""
    monkeypatch.setattr(loader, "get_paths", lambda: fake_paths)
    settings = loader.get_settings()

    path = loader.resolve_raw_path()
    assert path == fake_paths.data_raw / settings.data.raw_file


def test_write_parquet_creates_parent_dir_and_persists(
    synthetic_raw: pl.DataFrame, tmp_path: Path
) -> None:
    """``write_parquet`` cria o diretório-pai e grava o conteúdo corretamente."""
    output = write_parquet(synthetic_raw, tmp_path / "nested" / "raw.parquet")
    assert output.exists()
    assert pl.read_parquet(output).height == synthetic_raw.height
