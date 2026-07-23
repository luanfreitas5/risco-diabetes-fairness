"""Contratos de dados (Pandera) do dataset BRFSS 2015 de risco de diabetes.

Define os schemas dos estágios *raw* e *processed*, com tipos, faixas e
categorias aceitas segundo o codebook do BRFSS. A validação falha cedo com
:class:`DataValidationError` caso os dados violem o contrato.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.errors import SchemaError, SchemaErrors
from pandera.typing.polars import Series

from config.logging import get_logger
from exceptions.data import DataValidationError

logger = get_logger(__name__)

# Faixa fisiologicamente plausível de IMC para triagem (evita outliers grosseiros).
_BMI_MIN, _BMI_MAX = 10.0, 100.0


class RawDiabetesSchema(pa.DataFrameModel):
    """Contrato do dataset bruto (``diabetes_binary_health_indicators``).

    Todas as colunas são numéricas; binárias em {0, 1} e ordinais nos intervalos
    do codebook BRFSS. ``strict = True`` rejeita colunas inesperadas.
    """

    Diabetes_binary: Series[float] = pa.Field(isin=[0.0, 1.0])
    HighBP: Series[float] = pa.Field(isin=[0.0, 1.0])
    HighChol: Series[float] = pa.Field(isin=[0.0, 1.0])
    CholCheck: Series[float] = pa.Field(isin=[0.0, 1.0])
    BMI: Series[float] = pa.Field(ge=_BMI_MIN, le=_BMI_MAX)
    Smoker: Series[float] = pa.Field(isin=[0.0, 1.0])
    Stroke: Series[float] = pa.Field(isin=[0.0, 1.0])
    HeartDiseaseorAttack: Series[float] = pa.Field(isin=[0.0, 1.0])
    PhysActivity: Series[float] = pa.Field(isin=[0.0, 1.0])
    Fruits: Series[float] = pa.Field(isin=[0.0, 1.0])
    Veggies: Series[float] = pa.Field(isin=[0.0, 1.0])
    HvyAlcoholConsump: Series[float] = pa.Field(isin=[0.0, 1.0])
    AnyHealthcare: Series[float] = pa.Field(isin=[0.0, 1.0])
    NoDocbcCost: Series[float] = pa.Field(isin=[0.0, 1.0])
    GenHlth: Series[float] = pa.Field(ge=1.0, le=5.0)
    MentHlth: Series[float] = pa.Field(ge=0.0, le=30.0)
    PhysHlth: Series[float] = pa.Field(ge=0.0, le=30.0)
    DiffWalk: Series[float] = pa.Field(isin=[0.0, 1.0])
    Sex: Series[float] = pa.Field(isin=[0.0, 1.0])
    Age: Series[float] = pa.Field(ge=1.0, le=13.0)
    Education: Series[float] = pa.Field(ge=1.0, le=6.0)
    Income: Series[float] = pa.Field(ge=1.0, le=8.0)

    class Config(pa.DataFrameModel.Config):
        """Configuração do schema."""

        strict = True
        coerce = True


class ProcessedDiabetesSchema(pa.DataFrameModel):
    """Contrato do dataset processado, com a feature derivada ``AgeBand``.

    Herdaria de :class:`RawDiabetesSchema`, mas o Pandera-polars valida melhor
    com o schema explícito. Acrescenta a coluna categórica de faixa etária
    agregada, usada na auditoria de justiça.
    """

    Diabetes_binary: Series[int] = pa.Field(isin=[0, 1])
    HighBP: Series[float] = pa.Field(isin=[0.0, 1.0])
    HighChol: Series[float] = pa.Field(isin=[0.0, 1.0])
    CholCheck: Series[float] = pa.Field(isin=[0.0, 1.0])
    BMI: Series[float] = pa.Field(ge=_BMI_MIN, le=_BMI_MAX)
    Smoker: Series[float] = pa.Field(isin=[0.0, 1.0])
    Stroke: Series[float] = pa.Field(isin=[0.0, 1.0])
    HeartDiseaseorAttack: Series[float] = pa.Field(isin=[0.0, 1.0])
    PhysActivity: Series[float] = pa.Field(isin=[0.0, 1.0])
    Fruits: Series[float] = pa.Field(isin=[0.0, 1.0])
    Veggies: Series[float] = pa.Field(isin=[0.0, 1.0])
    HvyAlcoholConsump: Series[float] = pa.Field(isin=[0.0, 1.0])
    AnyHealthcare: Series[float] = pa.Field(isin=[0.0, 1.0])
    NoDocbcCost: Series[float] = pa.Field(isin=[0.0, 1.0])
    GenHlth: Series[float] = pa.Field(ge=1.0, le=5.0)
    MentHlth: Series[float] = pa.Field(ge=0.0, le=30.0)
    PhysHlth: Series[float] = pa.Field(ge=0.0, le=30.0)
    DiffWalk: Series[float] = pa.Field(isin=[0.0, 1.0])
    Sex: Series[float] = pa.Field(isin=[0.0, 1.0])
    Age: Series[float] = pa.Field(ge=1.0, le=13.0)
    Education: Series[float] = pa.Field(ge=1.0, le=6.0)
    Income: Series[float] = pa.Field(ge=1.0, le=8.0)
    AgeBand: Series[str] = pa.Field(isin=["jovem", "meia-idade", "idoso"])

    class Config(pa.DataFrameModel.Config):
        """Configuração do schema."""

        strict = True
        coerce = True


def validate_raw(df: pl.DataFrame) -> pl.DataFrame:
    """Valida um DataFrame contra o contrato do dataset bruto.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame bruto a validar.

    Returns
    -------
    pl.DataFrame
        O mesmo DataFrame, validado (e coagido aos tipos do schema).

    Raises
    ------
    DataValidationError
        Se os dados violarem o contrato.

    Examples
    --------
    >>> validate_raw(df)  # doctest: +SKIP
    """
    try:
        return RawDiabetesSchema.validate(df, lazy=True)
    except (SchemaError, SchemaErrors) as exc:
        logger.exception("Falha na validação do dataset bruto")
        raise DataValidationError("Dataset bruto não atende ao contrato de dados.") from exc


def validate_processed(df: pl.DataFrame) -> pl.DataFrame:
    """Valida um DataFrame contra o contrato do dataset processado.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame processado a validar.

    Returns
    -------
    pl.DataFrame
        O mesmo DataFrame, validado.

    Raises
    ------
    DataValidationError
        Se os dados violarem o contrato.

    Examples
    --------
    >>> validate_processed(df)  # doctest: +SKIP
    """
    try:
        return ProcessedDiabetesSchema.validate(df, lazy=True)
    except (SchemaError, SchemaErrors) as exc:
        logger.exception("Falha na validação do dataset processado")
        raise DataValidationError("Dataset processado não atende ao contrato de dados.") from exc
