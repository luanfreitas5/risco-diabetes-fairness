"""Fixtures compartilhadas dos testes.

Gera pequenos DataFrames sintéticos e válidos (nunca dados de produção),
com a variável-alvo correlacionada a fatores de risco para permitir testes
comportamentais do modelo.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
import pytest

matplotlib.use("Agg")  # backend headless: evita abrir janelas/travar em CI sem display.

from config.paths import ProjectPaths
from constants import columns as col

RANDOM_SEED = 42
N_ROWS = 400


@pytest.fixture
def fake_paths(tmp_path: Path) -> ProjectPaths:
    """Estrutura de :class:`ProjectPaths` isolada em um diretório temporário.

    Usada para testar pipelines e funções de IO sem tocar os diretórios reais
    do projeto (``data/``, ``models/``, ``reports/``).
    """
    return ProjectPaths(
        root=tmp_path,
        data_raw=tmp_path / "data" / "raw",
        data_external=tmp_path / "data" / "external",
        data_interim=tmp_path / "data" / "interim",
        data_processed=tmp_path / "data" / "processed",
        models=tmp_path / "models",
        reports=tmp_path / "reports",
        figures=tmp_path / "reports" / "figures",
        metrics=tmp_path / "reports" / "metrics",
        model_cards=tmp_path / "reports" / "model_cards",
        datasheets=tmp_path / "reports" / "datasheets",
        logs=tmp_path / "logs",
    )


@pytest.fixture(scope="session")
def synthetic_raw() -> pl.DataFrame:
    """DataFrame sintético válido no formato do dataset bruto BRFSS.

    Returns
    -------
    pl.DataFrame
        Dados sintéticos com todas as 22 colunas do contrato bruto.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    high_bp = rng.integers(0, 2, N_ROWS)
    high_chol = rng.integers(0, 2, N_ROWS)
    bmi = rng.uniform(15, 55, N_ROWS)

    # Risco latente correlacionado com HighBP, HighChol e IMC -> alvo.
    logit = -2.0 + 1.5 * high_bp + 1.0 * high_chol + 0.04 * (bmi - 28)
    prob = 1 / (1 + np.exp(-logit))
    target = rng.binomial(1, prob)

    data = {
        col.TARGET: target.astype(float),
        "HighBP": high_bp.astype(float),
        "HighChol": high_chol.astype(float),
        "CholCheck": rng.integers(0, 2, N_ROWS).astype(float),
        "BMI": bmi,
        "Smoker": rng.integers(0, 2, N_ROWS).astype(float),
        "Stroke": rng.integers(0, 2, N_ROWS).astype(float),
        "HeartDiseaseorAttack": rng.integers(0, 2, N_ROWS).astype(float),
        "PhysActivity": rng.integers(0, 2, N_ROWS).astype(float),
        "Fruits": rng.integers(0, 2, N_ROWS).astype(float),
        "Veggies": rng.integers(0, 2, N_ROWS).astype(float),
        "HvyAlcoholConsump": rng.integers(0, 2, N_ROWS).astype(float),
        "AnyHealthcare": rng.integers(0, 2, N_ROWS).astype(float),
        "NoDocbcCost": rng.integers(0, 2, N_ROWS).astype(float),
        "GenHlth": rng.integers(1, 6, N_ROWS).astype(float),
        "MentHlth": rng.integers(0, 31, N_ROWS).astype(float),
        "PhysHlth": rng.integers(0, 31, N_ROWS).astype(float),
        "DiffWalk": rng.integers(0, 2, N_ROWS).astype(float),
        "Sex": rng.integers(0, 2, N_ROWS).astype(float),
        "Age": rng.integers(1, 14, N_ROWS).astype(float),
        "Education": rng.integers(1, 7, N_ROWS).astype(float),
        "Income": rng.integers(1, 9, N_ROWS).astype(float),
    }
    return pl.DataFrame(data).select(col.ALL_COLUMNS)
