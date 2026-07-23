"""Testes do particionamento de dados."""

from __future__ import annotations

import polars as pl
import pytest

from constants import columns as col
from data.splitter import split_data
from features.engineering import engineer_features


@pytest.fixture(scope="module")
def processed(synthetic_raw: pl.DataFrame) -> pl.DataFrame:
    """Dataset processado (com AgeBand) para os testes de split."""
    return engineer_features(synthetic_raw)


def test_split_partitions_are_disjoint_and_complete(processed: pl.DataFrame) -> None:
    """As partições somam o total e não têm sobreposição de tamanho."""
    split = split_data(processed)
    total = len(split.x_train) + len(split.x_valid) + len(split.x_test)
    assert total == processed.height


def test_split_features_have_expected_columns(processed: pl.DataFrame) -> None:
    """As features das partições contêm exatamente as 21 colunas do modelo."""
    split = split_data(processed)
    assert list(split.x_train.columns) == col.FEATURES


def test_split_sensitive_is_aligned(processed: pl.DataFrame) -> None:
    """Os atributos sensíveis ficam alinhados por tamanho a cada partição."""
    split = split_data(processed)
    assert len(split.sensitive_test) == len(split.x_test)
    assert "AgeBand" in split.sensitive_test.columns


def test_split_raises_on_empty() -> None:
    """Um DataFrame vazio levanta ValueError."""
    empty = pl.DataFrame(schema=dict.fromkeys(col.ALL_COLUMNS, pl.Float64))
    with pytest.raises(ValueError, match="vazio"):
        split_data(empty)
