"""Testes do pré-processamento."""

from __future__ import annotations

import pandas as pd

from constants import columns as col
from preprocessing.pipeline import build_preprocessor


def _sample_features() -> pd.DataFrame:
    """Cria um DataFrame mínimo com todas as features."""
    return pd.DataFrame([dict.fromkeys(col.FEATURES, 1.0)])


def test_preprocessor_scale_has_scaler() -> None:
    """Com scale=True, o transformador inclui a etapa de padronização."""
    pre = build_preprocessor(scale=True)
    assert pre.transformers[0][0] == "scale"


def test_preprocessor_passthrough_preserves_feature_count() -> None:
    """Com scale=False, todas as features são preservadas (passthrough)."""
    pre = build_preprocessor(scale=False)
    transformed = pre.fit_transform(_sample_features())
    assert transformed.shape[1] == len(col.FEATURES)


def test_preprocessor_scale_preserves_feature_count() -> None:
    """A padronização não altera o número de colunas."""
    pre = build_preprocessor(scale=True)
    transformed = pre.fit_transform(_sample_features())
    assert transformed.shape[1] == len(col.FEATURES)
