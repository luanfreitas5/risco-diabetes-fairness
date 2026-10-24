"""Ingestão, carregamento, escrita e particionamento de dados.

Módulos
-------
loader
    Carrega as camadas bronze (CSV), silver e gold (parquet), validando os contratos Pandera.
writer
    Persiste DataFrames em parquet (armazenamento colunar).
splitter
    Divide os dados em treino/validação/teste de forma estratificada.
"""

from data.loader import load_bronze, load_gold, load_silver
from data.splitter import DataSplit, split_data
from data.writer import write_parquet

__all__ = [
    "DataSplit",
    "load_bronze",
    "load_gold",
    "load_silver",
    "split_data",
    "write_parquet",
]
