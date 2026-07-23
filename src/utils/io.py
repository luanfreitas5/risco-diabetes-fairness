"""Utilitários de entrada/saída para artefatos leves (JSON).

Modelos e datasets usam módulos dedicados (joblib/parquet); aqui tratamos de
metadados, métricas e manifestos.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(data: dict[str, Any], path: Path, *, indent: int = 2) -> Path:
    """Salva um dicionário em JSON (UTF-8, legível), criando o diretório-pai.

    Parameters
    ----------
    data : dict[str, Any]
        Conteúdo a ser serializado.
    path : Path
        Caminho de saída (``.json``).
    indent : int, optional
        Indentação do JSON, by default 2.

    Returns
    -------
    Path
        O caminho gravado.

    Examples
    --------
    >>> save_json({"roc_auc": 0.82}, Path("reports/metrics/metrics.json"))  # doctest: +SKIP
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=indent, default=str),
        encoding="utf-8",
    )
    return path


def read_json(path: Path) -> dict[str, Any]:
    """Lê um arquivo JSON e retorna um dicionário.

    Parameters
    ----------
    path : Path
        Caminho do arquivo JSON.

    Returns
    -------
    dict[str, Any]
        Conteúdo desserializado.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.

    Examples
    --------
    >>> read_json(Path("reports/metrics/metrics.json"))  # doctest: +SKIP
    {'roc_auc': 0.82}
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo JSON não encontrado: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
