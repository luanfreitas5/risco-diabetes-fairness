"""Testes de engenharia de features."""

from __future__ import annotations

import polars as pl
import pytest

from constants import columns as col
from features.engineering import add_age_band, engineer_features


@pytest.mark.smoke
def test_add_age_band_maps_boundaries() -> None:
    """Verifica o mapeamento das faixas etárias nos limites das categorias."""
    df = pl.DataFrame({"Age": [1.0, 4.0, 5.0, 8.0, 9.0, 13.0]})
    result = add_age_band(df)
    assert result[col.AGE_BAND].to_list() == [
        "jovem",
        "jovem",
        "meia-idade",
        "meia-idade",
        "idoso",
        "idoso",
    ]


def test_add_age_band_requires_age_column() -> None:
    """Levanta ValueError quando a coluna 'Age' está ausente."""
    with pytest.raises(ValueError, match="Age"):
        add_age_band(pl.DataFrame({"outra": [1.0]}))


def test_engineer_features_casts_target_to_int(synthetic_raw: pl.DataFrame) -> None:
    """A engenharia converte o alvo para inteiro e adiciona AgeBand."""
    result = engineer_features(synthetic_raw)
    assert col.AGE_BAND in result.columns
    assert result[col.TARGET].dtype == pl.Int64
