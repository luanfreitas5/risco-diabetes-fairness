"""Testes dos contratos de dados (Pandera)."""

from __future__ import annotations

import polars as pl
import pytest

from exceptions.data import DataValidationError
from features.engineering import engineer_features
from schemas.dataset import validate_processed, validate_raw


@pytest.mark.smoke
def test_validate_raw_accepts_valid_data(synthetic_raw: pl.DataFrame) -> None:
    """Dados sintéticos válidos passam no contrato bruto."""
    validated = validate_raw(synthetic_raw)
    assert validated.height == synthetic_raw.height


def test_validate_raw_rejects_out_of_range(synthetic_raw: pl.DataFrame) -> None:
    """IMC fora da faixa plausível viola o contrato e levanta erro."""
    corrupted = synthetic_raw.with_columns(pl.lit(500.0).alias("BMI"))
    with pytest.raises(DataValidationError):
        validate_raw(corrupted)


def test_validate_raw_rejects_unexpected_column(synthetic_raw: pl.DataFrame) -> None:
    """Colunas inesperadas são rejeitadas (strict=True)."""
    corrupted = synthetic_raw.with_columns(pl.lit(1).alias("ColunaFantasma"))
    with pytest.raises(DataValidationError):
        validate_raw(corrupted)


def test_validate_processed_accepts_engineered(synthetic_raw: pl.DataFrame) -> None:
    """O dataset processado (com AgeBand) passa no contrato processado."""
    processed = engineer_features(synthetic_raw)
    validated = validate_processed(processed)
    assert "AgeBand" in validated.columns


def test_validate_processed_rejects_unexpected_age_band(synthetic_raw: pl.DataFrame) -> None:
    """Uma categoria de ``AgeBand`` fora do domínio aceito viola o contrato."""
    processed = engineer_features(synthetic_raw)
    corrupted = processed.with_columns(pl.lit("desconhecido").alias("AgeBand"))
    with pytest.raises(DataValidationError):
        validate_processed(corrupted)
