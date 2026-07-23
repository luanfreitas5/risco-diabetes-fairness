"""Carregamento do dataset de risco de diabetes (BRFSS 2015).

Lê o CSV bruto com Polars, valida contra o contrato de dados e registra o hash
do arquivo para rastreabilidade. Também carrega o parquet processado.
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


def load_raw(raw_file: str | None = None, *, validate: bool = True) -> pl.DataFrame:
    """Carrega o dataset bruto (CSV) e valida contra o contrato de dados.

    Parameters
    ----------
    raw_file : str, optional
        Nome do arquivo em ``data/raw``. Se ``None``, usa o definido em
        ``config.yaml`` (``data.raw_file``).
    validate : bool, optional
        Se ``True``, valida contra :class:`RawDiabetesSchema`, by default True.

    Returns
    -------
    pl.DataFrame
        Dataset bruto carregado (e validado).

    Raises
    ------
    DataNotFoundError
        Se o arquivo bruto não for encontrado.

    Examples
    --------
    >>> df = load_raw()  # doctest: +SKIP
    >>> df.shape  # doctest: +SKIP
    (253680, 22)
    """
    settings = get_settings()
    paths = get_paths()
    file_name = raw_file or settings.data.raw_file
    path = paths.data_raw / file_name

    if not path.exists():
        raise DataNotFoundError(
            f"Arquivo bruto não encontrado: {path}. "
            "Baixe o dataset do Kaggle e coloque-o em data/raw/."
        )

    logger.info("Carregando dataset bruto: %s", path.name)
    df = pl.read_csv(path)
    logger.info(
        "Dataset carregado: %d linhas x %d colunas | hash=%s",
        df.height,
        df.width,
        hash_file(path)[:12],
    )

    if validate:
        df = validate_raw(df)
        logger.info("Contrato de dados (raw) validado com sucesso.")
    return df


def load_processed(file_name: str = "diabetes_processed.parquet") -> pl.DataFrame:
    """Carrega o dataset processado (parquet) e valida o contrato processado.

    Parameters
    ----------
    file_name : str, optional
        Nome do parquet em ``data/processed``, by default
        ``"diabetes_processed.parquet"``.

    Returns
    -------
    pl.DataFrame
        Dataset processado validado.

    Raises
    ------
    DataNotFoundError
        Se o arquivo processado não existir (rode o pré-processamento antes).

    Examples
    --------
    >>> df = load_processed()  # doctest: +SKIP
    """
    path = get_paths().data_processed / file_name
    if not path.exists():
        raise DataNotFoundError(
            f"Dataset processado não encontrado: {path}. Rode `make preprocess` antes."
        )
    logger.info("Carregando dataset processado: %s", path.name)
    return validate_processed(pl.read_parquet(path))


def resolve_raw_path(raw_file: str | None = None) -> Path:
    """Retorna o caminho absoluto do arquivo bruto configurado.

    Parameters
    ----------
    raw_file : str, optional
        Nome do arquivo; se ``None``, usa o de ``config.yaml``.

    Returns
    -------
    Path
        Caminho absoluto para o arquivo bruto.
    """
    settings = get_settings()
    return get_paths().data_raw / (raw_file or settings.data.raw_file)
