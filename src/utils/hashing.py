"""Utilitários de hashing para rastrear a versão dos dados.

Um checksum estável dos dados permite detectar mudanças silenciosas e vincular
cada modelo ao dataset exato que o produziu (reprodutibilidade).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl

_CHUNK_SIZE = 1 << 20  # 1 MiB


def hash_file(path: Path, *, chunk_size: int = _CHUNK_SIZE) -> str:
    """Retorna o hash SHA-256 de um arquivo, lido em blocos.

    Parameters
    ----------
    path : Path
        Caminho do arquivo.
    chunk_size : int, optional
        Tamanho do bloco de leitura em bytes, by default 1 MiB.

    Returns
    -------
    str
        Digest hexadecimal SHA-256.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.

    Examples
    --------
    >>> hash_file(Path("data/raw/dataset.csv"))  # doctest: +SKIP
    '9f86d081...'
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado para hashing: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_dataframe(df: pl.DataFrame) -> str:
    """Retorna um hash SHA-256 determinístico do conteúdo de um DataFrame.

    Serializa o DataFrame de forma estável (colunas ordenadas) antes de aplicar
    o hash, permitindo comparar versões de dados independentemente do disco.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame a ser hasheado.

    Returns
    -------
    str
        Digest hexadecimal SHA-256.

    Examples
    --------
    >>> hash_dataframe(pl.DataFrame({"a": [1, 2]}))  # doctest: +SKIP
    'a1b2c3...'
    """
    payload = df.select(sorted(df.columns)).hash_rows(seed=0).to_numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()
