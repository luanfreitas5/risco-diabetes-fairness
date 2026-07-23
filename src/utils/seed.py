"""Gestão de sementes de aleatoriedade.

Reexporta :func:`config.environment.seed_everything` para manter um ponto de
acesso conveniente em ``utils``.
"""

from __future__ import annotations

from config.environment import DEFAULT_SEED, seed_everything

__all__ = ["DEFAULT_SEED", "seed_everything"]
