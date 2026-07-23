"""Tema e paleta de cores compartilhados do projeto.

Reusar a mesma paleta em todas as figuras dá consistência visual aos relatórios
e ao dashboard.
"""

from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt
import seaborn as sns

# Paleta acessível (tons distinguíveis, amigáveis a daltonismo).
PALETTE: Final[dict[str, str]] = {
    "primary": "#2C6E9B",  # azul — classe/negativo
    "accent": "#D1495B",  # vermelho — positivo/risco
    "neutral": "#6C757D",  # cinza — referência
    "positive": "#3E8E7E",  # verde — bom desempenho
    "warning": "#E9A03B",  # âmbar — atenção
}

# Ordem categórica padrão para gráficos com múltiplas séries.
CATEGORICAL: Final[list[str]] = [
    PALETTE["primary"],
    PALETTE["accent"],
    PALETTE["positive"],
    PALETTE["warning"],
    PALETTE["neutral"],
]


def apply_theme() -> None:
    """Aplica o tema visual padrão do projeto ao matplotlib/seaborn.

    Examples
    --------
    >>> apply_theme()
    """
    sns.set_theme(style="whitegrid", context="notebook")
    sns.set_palette(CATEGORICAL)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.titleweight"] = "bold"
