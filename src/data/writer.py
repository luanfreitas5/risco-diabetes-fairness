"""Persistência de DataFrames em parquet (armazenamento colunar via PyArrow)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from config.logging import get_logger

logger = get_logger(__name__)


def write_parquet(df: pl.DataFrame, path: Path) -> Path:
    """Grava um DataFrame em parquet, criando o diretório-pai se necessário.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame a persistir.
    path : Path
        Caminho de saída (``.parquet``).

    Returns
    -------
    Path
        O caminho gravado.

    Examples
    --------
    >>> write_parquet(df, Path("data/processed/diabetes_processed.parquet"))  # doctest: +SKIP
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    logger.info("Parquet gravado: %s (%d linhas)", path.name, df.height)
    return path
