"""Particionamento estratificado em treino, validação e teste.

O split preserva a prevalência da variável-alvo (estratificação) e mantém, em
paralelo, um DataFrame com os atributos sensíveis alinhado a cada partição,
para a auditoria de justiça.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split

from config.logging import get_logger
from config.settings import get_settings
from constants import columns as col

logger = get_logger(__name__)


def _split_three(
    x: pd.DataFrame,
    y: pd.Series,
    sensitive: pd.DataFrame,
    *,
    test_size: float,
    stratify: pd.Series | None,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Particiona ``x``, ``y`` e ``sensitive`` em conjunto para o ``train_test_split``.

    Isola a chamada de três arrays ao scikit-learn, cuja sobrecarga de tipos só
    anota com precisão o caso de dois arrays — o retorno aqui é anotado
    explicitamente para o restante do módulo.
    """
    x_a, x_b, y_a, y_b, sens_a, sens_b = train_test_split(
        x, y, sensitive, test_size=test_size, stratify=stratify, random_state=random_state
    )
    return cast(
        "tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]",
        (x_a, x_b, y_a, y_b, sens_a, sens_b),
    )


@dataclass(frozen=True)
class DataSplit:
    """Agrupa as partições de treino, validação e teste.

    Attributes
    ----------
    x_train, x_valid, x_test : pd.DataFrame
        Features (as 21 colunas de risco) de cada partição.
    y_train, y_valid, y_test : pd.Series
        Variável-alvo binária de cada partição.
    sensitive_train, sensitive_valid, sensitive_test : pd.DataFrame
        Atributos sensíveis (Sex, AgeBand, Income, Education) alinhados a cada
        partição, usados na auditoria de justiça.
    """

    x_train: pd.DataFrame
    x_valid: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series
    sensitive_train: pd.DataFrame
    sensitive_valid: pd.DataFrame
    sensitive_test: pd.DataFrame


def split_data(
    df: pl.DataFrame,
    *,
    sensitive_features: list[str] | None = None,
) -> DataSplit:
    """Divide o dataset processado em treino/validação/teste estratificados.

    Parameters
    ----------
    df : pl.DataFrame
        Dataset processado (contém as features, o alvo e ``AgeBand``).
    sensitive_features : list[str], optional
        Colunas sensíveis a preservar para auditoria. Se ``None``, usa as de
        ``config.yaml``.

    Returns
    -------
    DataSplit
        Estrutura com as partições e os atributos sensíveis alinhados.

    Raises
    ------
    ValueError
        Se o DataFrame estiver vazio.

    Examples
    --------
    >>> split = split_data(df)  # doctest: +SKIP
    >>> split.x_train.shape  # doctest: +SKIP
    """
    if df.is_empty():
        raise ValueError("DataFrame vazio: não é possível particionar os dados.")

    settings = get_settings()
    sensitive = sensitive_features or settings.sensitive_features
    seed = settings.random_seed
    test_size = settings.data.test_size
    valid_size = settings.data.valid_size
    stratify_flag = settings.data.stratify

    pdf = df.to_pandas()
    y = pdf[col.TARGET].astype(int)
    x = pdf[col.FEATURES].copy()
    sens = pdf[sensitive].copy()

    strat = y if stratify_flag else None
    # 1º split: separa o teste (holdout).
    x_rem, x_test, y_rem, y_test, sens_rem, sens_test = _split_three(
        x, y, sens, test_size=test_size, stratify=strat, random_state=seed
    )

    # 2º split: separa validação do restante, mantendo a proporção configurada.
    valid_fraction = valid_size / (1.0 - test_size)
    strat_rem = y_rem if stratify_flag else None
    x_train, x_valid, y_train, y_valid, sens_train, sens_valid = _split_three(
        x_rem, y_rem, sens_rem, test_size=valid_fraction, stratify=strat_rem, random_state=seed
    )

    logger.info(
        "Split concluído | treino=%d, validação=%d, teste=%d | prevalência treino=%.3f",
        len(x_train),
        len(x_valid),
        len(x_test),
        float(y_train.mean()),
    )

    return DataSplit(
        x_train=x_train.reset_index(drop=True),
        x_valid=x_valid.reset_index(drop=True),
        x_test=x_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_valid=y_valid.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        sensitive_train=sens_train.reset_index(drop=True),
        sensitive_valid=sens_valid.reset_index(drop=True),
        sensitive_test=sens_test.reset_index(drop=True),
    )
