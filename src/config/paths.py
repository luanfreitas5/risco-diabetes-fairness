"""Caminhos centralizados do projeto, resolvidos a partir de ``configs/paths.yaml``.

Todos os caminhos usam :class:`pathlib.Path` e são resolvidos de forma absoluta
a partir da raiz do projeto — nunca strings hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

# Raiz do projeto: src/config/paths.py -> parents[2] == raiz.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"


@dataclass(frozen=True)
class ProjectPaths:
    """Agrupa os caminhos do projeto como objetos ``Path`` imutáveis.

    Attributes
    ----------
    root : Path
        Raiz do projeto.
    data_raw, data_external, data_interim, data_processed : Path
        Diretórios dos estágios de dados.
    models : Path
        Diretório de modelos persistidos.
    reports, figures, metrics, model_cards, datasheets : Path
        Diretórios de relatórios e artefatos de IA responsável.
    logs : Path
        Diretório de logs.
    """

    root: Path
    data_raw: Path
    data_external: Path
    data_interim: Path
    data_processed: Path
    models: Path
    reports: Path
    figures: Path
    metrics: Path
    model_cards: Path
    datasheets: Path
    logs: Path

    def ensure_dirs(self) -> None:
        """Cria todos os diretórios de saída caso ainda não existam.

        Examples
        --------
        >>> get_paths().ensure_dirs()
        """
        for path in (
            self.data_interim,
            self.data_processed,
            self.models,
            self.figures,
            self.metrics,
            self.model_cards,
            self.datasheets,
            self.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_paths(config_path: Path | None = None) -> ProjectPaths:
    """Carrega e resolve os caminhos do projeto a partir de ``paths.yaml``.

    Parameters
    ----------
    config_path : Path, optional
        Caminho para o ``paths.yaml``. Se ``None``, usa ``configs/paths.yaml``.

    Returns
    -------
    ProjectPaths
        Estrutura imutável com todos os caminhos resolvidos absolutamente.

    Raises
    ------
    FileNotFoundError
        Se o arquivo de configuração de caminhos não for encontrado.

    Examples
    --------
    >>> paths = get_paths()
    >>> paths.data_raw.name
    'raw'
    """
    path = config_path or (CONFIGS_DIR / "paths.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de caminhos não encontrado: {path}")

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

    def resolve(relative: str) -> Path:
        return (PROJECT_ROOT / relative).resolve()

    data = cfg["data"]
    reports = cfg["reports"]
    return ProjectPaths(
        root=PROJECT_ROOT,
        data_raw=resolve(data["raw"]),
        data_external=resolve(data["external"]),
        data_interim=resolve(data["interim"]),
        data_processed=resolve(data["processed"]),
        models=resolve(cfg["models"]),
        reports=resolve(reports["root"]),
        figures=resolve(reports["figures"]),
        metrics=resolve(reports["metrics"]),
        model_cards=resolve(reports["model_cards"]),
        datasheets=resolve(reports["datasheets"]),
        logs=resolve(cfg["logs"]),
    )
