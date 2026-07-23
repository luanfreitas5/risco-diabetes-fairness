"""Engenharia de features derivadas.

A principal feature derivada é ``AgeBand``: agrega as 13 faixas etárias do BRFSS
em 3 grupos (jovem, meia-idade, idoso), garantindo subgrupos com tamanho
amostral robusto para a auditoria de justiça. ``AgeBand`` **não** entra como
feature do modelo — serve apenas para a análise por subgrupo.
"""

from __future__ import annotations

import polars as pl

from config.logging import get_logger
from constants import columns as col

logger = get_logger(__name__)

# Limites (em código BRFSS de faixa etária, 1-13):
#   1-4  -> 18-39  (jovem)
#   5-8  -> 40-59  (meia-idade)
#   9-13 -> 60+    (idoso)
_YOUNG_MAX = 4
_MIDDLE_MAX = 8


def add_age_band(df: pl.DataFrame) -> pl.DataFrame:
    """Adiciona a coluna ``AgeBand`` derivada da faixa etária ``Age``.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame contendo a coluna ``Age`` (código BRFSS 1-13).

    Returns
    -------
    pl.DataFrame
        DataFrame com a nova coluna categórica ``AgeBand``.

    Raises
    ------
    ValueError
        Se a coluna ``Age`` não estiver presente.

    Examples
    --------
    >>> add_age_band(pl.DataFrame({"Age": [2.0, 6.0, 11.0]}))  # doctest: +SKIP
    """
    if "Age" not in df.columns:
        raise ValueError("Coluna 'Age' ausente: não é possível derivar 'AgeBand'.")

    age_band = (
        pl.when(pl.col("Age") <= _YOUNG_MAX)
        .then(pl.lit("jovem"))
        .when(pl.col("Age") <= _MIDDLE_MAX)
        .then(pl.lit("meia-idade"))
        .otherwise(pl.lit("idoso"))
        .alias(col.AGE_BAND)
    )
    return df.with_columns(age_band)


def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    """Aplica todas as etapas de engenharia de features ao dataset.

    Atualmente adiciona apenas ``AgeBand``, mas centraliza o ponto de extensão
    para novas features derivadas.

    Parameters
    ----------
    df : pl.DataFrame
        Dataset validado (contrato bruto).

    Returns
    -------
    pl.DataFrame
        Dataset com as features derivadas e o alvo convertido para inteiro.

    Examples
    --------
    >>> engineer_features(df)  # doctest: +SKIP
    """
    logger.info("Aplicando engenharia de features (AgeBand).")
    # Converte o alvo para inteiro (contrato processado exige Int).
    return add_age_band(df).with_columns(pl.col(col.TARGET).cast(pl.Int64))
