"""Funções utilitárias de uso geral.

Módulos
-------
seed
    Gestão de sementes de aleatoriedade (reexporta ``seed_everything``).
hashing
    Hash de arquivos e DataFrames para rastrear versões de dados.
io
    Leitura/escrita de artefatos (parquet, json, joblib).
"""

from utils.hashing import hash_dataframe, hash_file
from utils.io import read_json, save_json
from utils.seed import seed_everything

__all__ = [
    "hash_dataframe",
    "hash_file",
    "read_json",
    "save_json",
    "seed_everything",
]
