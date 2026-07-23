"""Configuração de logging: console via ``rich`` + arquivo rotativo diário.

Todas as mensagens de log do projeto são em pt-BR. Nunca registre PII ou
segredos (ver seção de Privacidade & LGPD no CLAUDE.md).
"""

from __future__ import annotations

import logging
from datetime import date
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

import yaml
from rich.logging import RichHandler

from config.paths import CONFIGS_DIR, PROJECT_ROOT

_STATE = {"configured": False}


def _load_logging_config(config_path: Path | None = None) -> dict[str, Any]:
    """Carrega ``configs/logging.yaml``, com fallback para valores padrão.

    Parameters
    ----------
    config_path : Path, optional
        Caminho do YAML de logging. Se ``None``, usa ``configs/logging.yaml``.

    Returns
    -------
    dict[str, Any]
        Configuração de logging.
    """
    path = config_path or (CONFIGS_DIR / "logging.yaml")
    if not path.exists():
        return {"level": "INFO", "console": {"enabled": True}, "file": {"enabled": False}}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def configure_logging(config_path: Path | None = None, *, force: bool = False) -> None:
    """Configura o logging global do projeto (idempotente).

    Adiciona um handler de console ``rich`` (colorido, com nome e linha) e um
    handler de arquivo rotativo diário em ``logs/log_YYYY-MM-DD.log``.

    Parameters
    ----------
    config_path : Path, optional
        Caminho do YAML de logging. Se ``None``, usa ``configs/logging.yaml``.
    force : bool, optional
        Reconfigura mesmo que o logging já tenha sido inicializado, by default
        False.

    Examples
    --------
    >>> configure_logging()
    >>> logging.getLogger(__name__).info("Pipeline iniciado")
    """
    if _STATE["configured"] and not force:
        return

    cfg = _load_logging_config(config_path)
    level = getattr(logging, str(cfg.get("level", "INFO")).upper(), logging.INFO)

    handlers: list[logging.Handler] = []

    console_cfg = cfg.get("console", {})
    if console_cfg.get("enabled", True):
        handlers.append(
            RichHandler(
                rich_tracebacks=console_cfg.get("rich_tracebacks", True),
                show_path=console_cfg.get("show_path", True),
                markup=True,
            )
        )

    file_cfg = cfg.get("file", {})
    if file_cfg.get("enabled", True):
        log_dir = PROJECT_ROOT / file_cfg.get("dir", "logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"log_{date.today().isoformat()}.log"
        file_handler = TimedRotatingFileHandler(
            log_file,
            when=file_cfg.get("when", "midnight"),
            backupCount=file_cfg.get("backup_count", 14),
            encoding="utf-8",
        )
        file_handler.setFormatter(
            logging.Formatter(
                file_cfg.get("format", "%(asctime)s \t %(levelname)s \t %(name)s \t %(message)s"),
                datefmt=cfg.get("date_format", "%Y-%m-%d %H:%M:%S"),
            )
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        # datefmt="[%X]",
        handlers=handlers,
        force=True,
    )
    _STATE["configured"] = True


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger configurado para o módulo indicado.

    Garante que o logging global esteja configurado antes de devolver o logger.

    Parameters
    ----------
    name : str
        Nome do logger (tipicamente ``__name__``).

    Returns
    -------
    logging.Logger
        Logger pronto para uso.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Mensagem de exemplo")
    """
    configure_logging()
    return logging.getLogger(name)
