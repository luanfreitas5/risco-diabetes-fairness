"""Carregamento do dataset de risco de diabetes (BRFSS 2015).

Lê o CSV da camada bronze com Polars, valida contra o contrato de dados e registra o hash
do arquivo para rastreabilidade. Também carrega os parquets das camadas silver e gold.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from config.logging import get_logger
from config.paths import get_paths
from config.settings import get_settings
from exceptions.data import DataNotFoundError
from schemas.dataset import validate_processed, validate_raw
from utils.hashing import hash_file

logger = get_logger(__name__)


def load_bronze(bronze_file: str | None = None, *, validate: bool = True) -> pl.DataFrame:
    """Carrega o dataset bronze (CSV) e valida contra o contrato de dados.

    Parameters
    ----------
    bronze_file : str, optional
        Nome do arquivo em ``data/bronze``. Se ``None``, usa o definido em
        ``config.yaml`` (``data.bronze_file``).
    validate : bool, optional
        Se ``True``, valida contra :class:`RawDiabetesSchema`, by default True.

    Returns
    -------
    pl.DataFrame
        Dataset bronze carregado (e validado).

    Raises
    ------
    DataNotFoundError
        Se o arquivo bronze não for encontrado.

    Examples
    --------
    >>> df = load_bronze()  # doctest: +SKIP
    >>> df.shape  # doctest: +SKIP
    (253680, 22)
    """
    settings = get_settings()
    paths = get_paths()
    file_name = bronze_file or settings.data.bronze_file
    path = paths.data_bronze / file_name

    if not path.exists():
        raise DataNotFoundError(
            f"Arquivo bronze não encontrado: {path}. "
            "Baixe o dataset do Kaggle e coloque-o em data/bronze/."
        )

    logger.info("Carregando dataset bronze: %s", path.name)
    raw_df = pl.read_csv(path)
    logger.info(
        "Dataset carregado: %d linhas x %d colunas | hash=%s",
        raw_df.height,
        raw_df.width,
        hash_file(path)[:12],
    )

    if validate:
        raw_df = validate_raw(raw_df)
        logger.info("Contrato de dados (bronze) validado com sucesso.")
    return raw_df


def load_gold(file_name: str = "diabetes_gold.parquet") -> pl.DataFrame:
    """Carrega o dataset gold (pronto para modelagem) (parquet) e valida o contrato processado.

    Parameters
    ----------
    file_name : str, optional
        Nome do parquet em ``data/gold``, by default
        ``"diabetes_gold.parquet"``.

    Returns
    -------
    pl.DataFrame
        Dataset gold validado.

    Raises
    ------
    DataNotFoundError
        Se o arquivo gold não existir (rode o pré-processamento antes).

    Examples
    --------
    >>> df = load_gold()  # doctest: +SKIP
    """
    path = get_paths().data_gold / file_name
    if not path.exists():
        raise DataNotFoundError(
            f"Dataset gold não encontrado: {path}. Rode `make preprocess` antes."
        )
    logger.info("Carregando dataset gold: %s", path.name)
    return validate_processed(pl.read_parquet(path))


def resolve_bronze_path(bronze_file: str | None = None) -> Path:
    """Retorna o caminho absoluto do arquivo bronze configurado.

    Parameters
    ----------
    bronze_file : str, optional
        Nome do arquivo; se ``None``, usa o de ``config.yaml``.

    Returns
    -------
    Path
        Caminho absoluto para o arquivo bronze.
    """
    settings = get_settings()
    return get_paths().data_bronze / (bronze_file or settings.data.bronze_file)


def load_silver(file_name: str = "diabetes_silver.parquet") -> pl.DataFrame:
    """Carrega o dataset silver (limpo e tipado) e valida o contrato bruto.

    Parameters
    ----------
    file_name : str, optional
        Nome do parquet em ``data/silver``, by default
        ``"diabetes_silver.parquet"``.

    Returns
    -------
    pl.DataFrame
        Dataset silver validado.

    Raises
    ------
    DataNotFoundError
        Se o arquivo silver não existir (rode o pré-processamento antes).

    Examples
    --------
    >>> df = load_silver()  # doctest: +SKIP
    """
    path = get_paths().data_silver / file_name
    if not path.exists():
        raise DataNotFoundError(
            f"Dataset silver não encontrado: {path}. Rode `make preprocess` antes."
        )
    logger.info("Carregando dataset silver: %s", path.name)
    return validate_raw(pl.read_parquet(path))
