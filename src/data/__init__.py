"""Ingestão, carregamento, escrita e particionamento de dados.

Módulos
-------
loader
    Carrega o dataset bruto (CSV) validando contra o contrato Pandera.
writer
    Persiste DataFrames em parquet (armazenamento colunar).
splitter
    Divide os dados em treino/validação/teste de forma estratificada.
"""

from data.loader import load_processed, load_raw
from data.splitter import DataSplit, split_data
from data.writer import write_parquet

__all__ = ["DataSplit", "load_processed", "load_raw", "split_data", "write_parquet"]
