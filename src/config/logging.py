"""Configuração única de logging: console ``rich`` + arquivo diário rotativo.

O logging é configurado **uma vez**, no ponto de entrada (``main()``), a partir de
``configs/logging.yaml`` via :func:`logging.config.dictConfig`. Os módulos apenas
obtêm seu logger com :func:`get_logger` e nunca configuram handlers.

Todas as mensagens de log do projeto são em pt-BR. Nunca registre PII, segredos
ou linhas de dados — apenas metadados (contagens, formas, métricas).
"""

from __future__ import annotations

import logging
import logging.config
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.logging import RichHandler

from config.paths import CONFIGS_DIR, PROJECT_ROOT

# Console compartilhado por RichHandler e rich.progress: barras de progresso e
# logs não se embaralham quando escrevem pelo mesmo console.
CONSOLE = Console(stderr=True)


def create_console_handler() -> RichHandler:
    """Cria o handler de console ligado ao console compartilhado.

    Returns
    -------
    RichHandler
        Handler colorido com tracebacks formatados.
    """
    return RichHandler(console=CONSOLE, rich_tracebacks=True, show_path=True, markup=False)


def _rename_rotated(default_name: str) -> str:
    """Converte ``logs/log.log.2026-09-27`` em ``logs/log_2026-09-27.log``."""
    path = Path(default_name)
    stem, date_str = path.name.split(".log.")
    return str(path.with_name(f"{stem}_{date_str}.log"))


def create_daily_file_handler(
    filename: str, backup_count: int = 30
) -> TimedRotatingFileHandler:
    """Cria o handler de arquivo com rotação diária à meia-noite.

    Parameters
    ----------
    filename : str
        Caminho do arquivo ativo, relativo à raiz do projeto.
    backup_count : int, optional
        Quantidade de arquivos diários mantidos, by default 30.

    Returns
    -------
    TimedRotatingFileHandler
        Handler que rotaciona para ``log_YYYY-MM-DD.log``.

    Examples
    --------
    >>> handler = create_daily_file_handler("logs/log.log")  # doctest: +SKIP
    """
    path = PROJECT_ROOT / filename
    path.parent.mkdir(parents=True, exist_ok=True)  # dictConfig falha sem o diretório
    handler = TimedRotatingFileHandler(
        path, when="midnight", backupCount=backup_count, encoding="utf-8"
    )
    handler.namer = _rename_rotated
    return handler


def configure_logging(level: str | None = None, config_path: Path | None = None) -> None:
    """Aplica a configuração de logging do YAML. Chamar apenas no ponto de entrada.

    Parameters
    ----------
    level : str, optional
        Sobrescreve o nível do root (ex.: ``--log-level DEBUG``), by default None.
    config_path : Path, optional
        YAML no formato ``dictConfig``. Se ``None``, usa ``configs/logging.yaml``.

    Raises
    ------
    FileNotFoundError
        Se o YAML não existir.
    ValueError
        Se o YAML ou o nível forem inválidos.

    Examples
    --------
    >>> configure_logging(level="DEBUG")  # doctest: +SKIP
    """
    path = config_path or (CONFIGS_DIR / "logging.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de logging não encontrado: {path}")
    config: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    if level is not None:
        config["root"]["level"] = level.upper()
    logging.config.dictConfig(config)  # substitui os handlers do root: não duplica
    logging.captureWarnings(True)  # warnings.warn() passa pelos mesmos handlers


def get_logger(name: str) -> logging.Logger:
    """Retorna o logger do módulo, sem configurar nada.

    Parameters
    ----------
    name : str
        Nome do logger (tipicamente ``__name__``).

    Returns
    -------
    logging.Logger
        Logger que herda os handlers do root, configurados em ``main()``.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Mensagem de exemplo")
    """
    return logging.getLogger(name)
