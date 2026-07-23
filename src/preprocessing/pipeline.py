"""Construção do pré-processamento (``ColumnTransformer``) por tipo de modelo.

O dataset BRFSS já vem limpo (sem nulos, todo numérico), então o
pré-processamento é enxuto:

- **Modelos lineares** (Regressão Logística): padronizam as features contínuas
  e ordinais (StandardScaler) e mantêm as binárias.
- **Modelos de árvore** (LightGBM): não exigem escala; as features passam
  diretamente (passthrough).
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from constants import columns as col

# Colunas que se beneficiam de padronização em modelos lineares.
_SCALE_COLUMNS = col.CONTINUOUS_FEATURES + col.ORDINAL_FEATURES


def build_preprocessor(*, scale: bool) -> ColumnTransformer:
    """Constrói o ``ColumnTransformer`` de pré-processamento.

    Parameters
    ----------
    scale : bool
        Se ``True``, padroniza contínuas/ordinais (para modelos lineares);
        se ``False``, aplica passthrough em tudo (para modelos de árvore).

    Returns
    -------
    ColumnTransformer
        Transformador pronto para compor um ``Pipeline`` do scikit-learn.

    Examples
    --------
    >>> pre = build_preprocessor(scale=True)
    >>> pre.transformers[0][0]
    'scale'
    """
    if scale:
        return ColumnTransformer(
            transformers=[("scale", StandardScaler(), _SCALE_COLUMNS)],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
    return ColumnTransformer(
        transformers=[("passthrough", "passthrough", col.FEATURES)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
