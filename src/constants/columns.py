"""Nomes e agrupamentos das colunas do dataset BRFSS 2015 (versão binária).

O dataset traz 21 features de fatores de risco + a variável-alvo. Todas as
colunas são numéricas; alguns campos são binários, outros ordinais/categóricos
e três são contínuos.
"""

from __future__ import annotations

from typing import Final

# --- Variável-alvo -----------------------------------------------------------
TARGET: Final[str] = "Diabetes_binary"

# --- Features binárias (0/1) -------------------------------------------------
BINARY_FEATURES: Final[list[str]] = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "DiffWalk",
    "Sex",
]

# --- Features ordinais/categóricas (inteiros com ordem natural) --------------
# GenHlth: 1 (excelente) a 5 (ruim); Age: 1-13 (faixas); Education: 1-6;
# Income: 1-8.
ORDINAL_FEATURES: Final[list[str]] = [
    "GenHlth",
    "Age",
    "Education",
    "Income",
]

# --- Features contínuas ------------------------------------------------------
# BMI: índice de massa corporal; MentHlth/PhysHlth: nº de dias (0-30).
CONTINUOUS_FEATURES: Final[list[str]] = [
    "BMI",
    "MentHlth",
    "PhysHlth",
]

# --- Conjunto completo de features -------------------------------------------
FEATURES: Final[list[str]] = BINARY_FEATURES + ORDINAL_FEATURES + CONTINUOUS_FEATURES

# --- Todas as colunas do arquivo bruto ---------------------------------------
ALL_COLUMNS: Final[list[str]] = [TARGET, *FEATURES]

# --- Atributos sensíveis (auditoria de justiça) ------------------------------
# ``AgeBand`` é derivada de ``Age`` em src/features/engineering.py.
SENSITIVE_RAW: Final[list[str]] = ["Sex", "Age", "Income", "Education"]
AGE_BAND: Final[str] = "AgeBand"
