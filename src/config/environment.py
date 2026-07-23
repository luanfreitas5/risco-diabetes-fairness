"""Reprodutibilidade: fixa todas as fontes de aleatoriedade do ambiente.

``random_state`` isolado não é reprodutibilidade. Aqui fixamos ``random``,
``numpy``, ``PYTHONHASHSEED`` e (se presente) frameworks adicionais.
"""

from __future__ import annotations

import contextlib
import os
import random

import numpy as np

from config.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED) -> int:
    """Fixa todas as sementes de aleatoriedade para garantir reprodutibilidade.

    Parameters
    ----------
    seed : int, optional
        Semente a ser aplicada, by default ``DEFAULT_SEED`` (42).

    Returns
    -------
    int
        A semente aplicada (útil para logging e rastreabilidade).

    Examples
    --------
    >>> seed_everything(42)
    42
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # Cria um gerador de números aleatórios determinístico
    rng = np.random.default_rng(seed)
    rng.normal()

    # opcional: apenas se lightgbm estiver disponível (usa a semente no fit)
    with contextlib.suppress(ImportError):
        import lightgbm  # type: ignore # noqa: F401, PLC0415

    logger.info("Sementes de aleatoriedade fixadas (seed=%d).", seed)
    return seed
