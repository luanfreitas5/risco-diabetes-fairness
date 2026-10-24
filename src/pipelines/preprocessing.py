"""Pipeline de preparação de dados na arquitetura medalhão: bronze -> silver -> gold.

- **bronze:** CSV exatamente como recebido (imutável); apenas lido, com hash
  registrado em ``data/bronze/manifest.json``.
- **silver:** dados limpos, tipados, sem duplicatas e validados pelo contrato
  bruto (Pandera), persistidos em parquet.
- **gold:** features de modelagem (engenharia de atributos), validadas pelo
  contrato processado, com manifesto de hash para rastreabilidade.
"""

from __future__ import annotations

from pathlib import Path

from config.logging import get_logger
from config.paths import get_paths
from data.loader import load_bronze, resolve_bronze_path
from data.writer import write_parquet
from features.engineering import engineer_features
from schemas.dataset import validate_processed, validate_raw
from utils.hashing import hash_dataframe, hash_file
from utils.io import save_json
from utils.numbers import to_float

logger = get_logger(__name__)

SILVER_FILE = "diabetes_silver.parquet"
GOLD_FILE = "diabetes_gold.parquet"


def run_preprocessing(bronze_file: str | None = None) -> Path:
    """Executa bronze -> silver -> gold e persiste o dataset gold.

    Parameters
    ----------
    bronze_file : str, optional
        Nome do arquivo em ``data/bronze``. Se ``None``, usa o do config.

    Returns
    -------
    Path
        Caminho do parquet gold gravado.

    Examples
    --------
    >>> run_preprocessing()  # doctest: +SKIP
    """
    logger.info("== Pipeline de pré-processamento iniciado ==")
    paths = get_paths()
    paths.ensure_dirs()

    bronze = load_bronze(bronze_file)
    bronze_path = resolve_bronze_path(bronze_file)
    if bronze_path.exists():
        save_json(
            {"file": bronze_path.name, "sha256": hash_file(bronze_path)},
            paths.data_bronze / "manifest.json",
        )

    # Silver: remove duplicatas exatas e valida o contrato de dados.
    silver = validate_raw(bronze.unique(maintain_order=True))
    logger.info("Silver: %d linhas (bronze: %d)", silver.height, bronze.height)
    write_parquet(silver, paths.data_silver / SILVER_FILE)

    # Gold: engenharia de features e contrato processado.
    gold = validate_processed(engineer_features(silver))
    output = write_parquet(gold, paths.data_gold / GOLD_FILE)

    manifest = {
        "n_rows": gold.height,
        "n_cols": gold.width,
        "target_prevalence": to_float(gold["Diabetes_binary"].mean()),
        "data_hash": hash_dataframe(gold),
    }
    save_json(manifest, paths.data_gold / "manifest.json")
    logger.info("== Pré-processamento concluído | hash=%s ==", manifest["data_hash"][:12])
    return output
