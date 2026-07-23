"""Pipeline de preparação de dados: raw -> processed.

Carrega e valida o dataset bruto, aplica a engenharia de features, valida o
contrato processado e persiste o resultado em parquet, registrando o hash dos
dados para rastreabilidade.
"""

from __future__ import annotations

from pathlib import Path

from config.logging import get_logger
from config.paths import get_paths
from data.loader import load_raw
from data.writer import write_parquet
from features.engineering import engineer_features
from schemas.dataset import validate_processed
from utils.hashing import hash_dataframe
from utils.io import save_json
from utils.numbers import to_float

logger = get_logger(__name__)

PROCESSED_FILE = "diabetes_processed.parquet"


def run_preprocessing(raw_file: str | None = None) -> Path:
    """Executa o pipeline de preparação e persiste o dataset processado.

    Parameters
    ----------
    raw_file : str, optional
        Nome do arquivo bruto em ``data/raw``. Se ``None``, usa o do config.

    Returns
    -------
    Path
        Caminho do parquet processado gravado.

    Examples
    --------
    >>> run_preprocessing()  # doctest: +SKIP
    """
    logger.info("== Pipeline de pré-processamento iniciado ==")
    paths = get_paths()
    paths.ensure_dirs()

    raw = load_raw(raw_file)
    processed = engineer_features(raw)
    processed = validate_processed(processed)

    output = write_parquet(processed, paths.data_processed / PROCESSED_FILE)

    manifest = {
        "n_rows": processed.height,
        "n_cols": processed.width,
        "target_prevalence": to_float(processed["Diabetes_binary"].mean()),
        "data_hash": hash_dataframe(processed),
    }
    save_json(manifest, paths.data_processed / "manifest.json")
    logger.info("== Pré-processamento concluído | hash=%s ==", manifest["data_hash"][:12])
    return output
