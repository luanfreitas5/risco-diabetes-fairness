"""Orquestração principal do pipeline ponta a ponta.

Executa, em ordem, todas as etapas do projeto: pré-processamento, treino,
avaliação e auditoria de justiça. Cada etapa também pode ser executada de forma
isolada via CLI (ver ``src/cli``) ou pelos alvos do ``Makefile``.

Examples
--------
Rodar o pipeline completo::

    uv run python -m main
    uv run python -m main --only train
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from warnings import filterwarnings

from config.logging import get_logger
from config.settings import get_settings
from pipelines.evaluation import run_evaluation
from pipelines.fairness import run_fairness_audit
from pipelines.preprocessing import run_preprocessing
from pipelines.training import run_training
from utils.seed import seed_everything

filterwarnings("ignore")

logger = get_logger(__name__)

Stage = str
STAGES: tuple[Stage, ...] = ("preprocess", "train", "evaluate", "audit", "all")

# Ordem canônica das etapas do pipeline (independe da ordem informada pelo usuário).
_STAGE_RUNNERS: dict[str, Callable[[], object]] = {
    "preprocess": run_preprocessing,
    "train": run_training,
    "evaluate": run_evaluation,
    "audit": run_fairness_audit,
}


def run_pipeline(stages: tuple[Stage, ...] = STAGES) -> None:
    """Executa as etapas do pipeline na ordem canônica.

    Parameters
    ----------
    stages : tuple[str, ...], optional
        Subconjunto de etapas a executar, by default todas as etapas.

    Examples
    --------
    >>> run_pipeline(("preprocess", "train"))  # doctest: +SKIP
    """
    seed_everything(get_settings().random_seed)
    logger.info("== Pipeline completo iniciado | etapas=%s ==", ", ".join(stages))

    run_all = "all" in stages
    for name, runner in _STAGE_RUNNERS.items():
        if run_all or name in stages:
            runner()

    logger.info("== Pipeline completo concluído ==")


def main() -> None:
    """Ponto de entrada da orquestração principal (via ``argparse``).

    Examples
    --------
    >>> main()  # doctest: +SKIP
    """
    parser = argparse.ArgumentParser(description="Pipeline de risco de diabetes com fairness.")
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default=None,
        help="Executa apenas uma etapa específica (padrão: todas).",
    )
    args = parser.parse_args()
    stages = (args.stage,) if args.stage else STAGES
    run_pipeline(stages)


if __name__ == "__main__":
    main()
