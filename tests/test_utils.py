"""Testes dos utilitários de hashing e IO."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from config.environment import seed_everything
from utils.hashing import hash_dataframe, hash_file
from utils.io import read_json, save_json
from utils.numbers import to_float


@pytest.mark.smoke
def test_hash_dataframe_is_deterministic() -> None:
    """O hash do DataFrame é estável para o mesmo conteúdo."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert hash_dataframe(df) == hash_dataframe(df.clone())


def test_hash_dataframe_changes_with_content() -> None:
    """Conteúdos diferentes produzem hashes diferentes."""
    df1 = pl.DataFrame({"a": [1, 2, 3]})
    df2 = pl.DataFrame({"a": [1, 2, 4]})
    assert hash_dataframe(df1) != hash_dataframe(df2)


def test_hash_file_missing_raises(tmp_path: Path) -> None:
    """Hashear um arquivo inexistente levanta FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        hash_file(tmp_path / "inexistente.csv")


def test_save_and_read_json_roundtrip(tmp_path: Path) -> None:
    """Salvar e ler um JSON preserva o conteúdo."""
    payload = {"roc_auc": 0.82, "modelo": "lightgbm"}
    path = save_json(payload, tmp_path / "metrics.json")
    assert read_json(path) == payload


def test_read_json_missing_file_raises(tmp_path: Path) -> None:
    """Ler um JSON inexistente levanta FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_json(tmp_path / "inexistente.json")


def test_hash_file_reads_existing_file_in_chunks(tmp_path: Path) -> None:
    """O hash de um arquivo existente é estável entre chamadas."""
    path = tmp_path / "dados.csv"
    path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    assert hash_file(path) == hash_file(path, chunk_size=4)


def test_seed_everything_fixes_hashseed_and_returns_seed() -> None:
    """``seed_everything`` fixa o ``PYTHONHASHSEED`` e devolve a semente aplicada."""
    assert seed_everything(123) == 123
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_to_float_converts_numpy_scalar() -> None:
    """``to_float`` converte um escalar numpy para ``float`` nativo."""
    assert to_float(np.float64(0.5)) == 0.5
