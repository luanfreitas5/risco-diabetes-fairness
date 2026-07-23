"""Rótulos da variável-alvo e mapeamentos dos subgrupos sensíveis.

Os mapeamentos seguem o codebook do BRFSS 2015 e são usados para tornar os
relatórios de justiça legíveis (ex.: ``Sex=0`` -> "Mulher").
"""

from __future__ import annotations

from typing import Final

# --- Variável-alvo -----------------------------------------------------------
TARGET_LABELS: Final[dict[int, str]] = {
    0: "Sem diabetes",
    1: "Diabetes ou pré-diabetes",
}

# --- Sexo (BRFSS: 0 = mulher, 1 = homem) -------------------------------------
SEX_LABELS: Final[dict[int, str]] = {
    0: "Mulher",
    1: "Homem",
}

# --- Faixas etárias BRFSS (código 1-13) --------------------------------------
AGE_LABELS: Final[dict[int, str]] = {
    1: "18-24",
    2: "25-29",
    3: "30-34",
    4: "35-39",
    5: "40-44",
    6: "45-49",
    7: "50-54",
    8: "55-59",
    9: "60-64",
    10: "65-69",
    11: "70-74",
    12: "75-79",
    13: "80+",
}

# --- Faixas etárias agregadas (para auditoria de justiça) --------------------
# Reduz 13 categorias a 3 grupos para ter subgrupos com tamanho amostral robusto.
AGE_BAND_LABELS: Final[dict[str, str]] = {
    "jovem": "18-39",
    "meia-idade": "40-59",
    "idoso": "60+",
}

# --- Escolaridade (BRFSS código 1-6) -----------------------------------------
EDUCATION_LABELS: Final[dict[int, str]] = {
    1: "Nunca frequentou / até jardim",
    2: "Fundamental (1-8)",
    3: "Médio incompleto (9-11)",
    4: "Médio completo",
    5: "Superior incompleto",
    6: "Superior completo",
}

# --- Renda (BRFSS código 1-8; faixas anuais em USD) --------------------------
INCOME_LABELS: Final[dict[int, str]] = {
    1: "< 10k",
    2: "10-15k",
    3: "15-20k",
    4: "20-25k",
    5: "25-35k",
    6: "35-50k",
    7: "50-75k",
    8: "75k+",
}
