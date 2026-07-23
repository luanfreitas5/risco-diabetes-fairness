"""Contratos de dados (schemas Pandera) para validação em fronteiras do pipeline.

Validar dados — e não só código — em cada estágio (raw -> processed, e
entrada/saída do modelo) evita corrupção silenciosa e falha cedo com erro claro.

Módulos
-------
dataset
    Schemas do dataset bruto e processado (BRFSS 2015).
"""

from schemas.dataset import (
    ProcessedDiabetesSchema,
    RawDiabetesSchema,
    validate_processed,
    validate_raw,
)

__all__ = [
    "ProcessedDiabetesSchema",
    "RawDiabetesSchema",
    "validate_processed",
    "validate_raw",
]
