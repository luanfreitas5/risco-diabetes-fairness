"""Carrega e valida a configuração do projeto com Pydantic.

Uma configuração inválida falha já na inicialização, com erro tipado e claro,
em vez de quebrar no meio da execução do pipeline.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.paths import CONFIGS_DIR


class DataConfig(BaseModel):
    """Parâmetros de dados e particionamento."""

    raw_file: str
    target: str = "Diabetes_binary"
    test_size: float = Field(gt=0, lt=1, default=0.20)
    valid_size: float = Field(gt=0, lt=1, default=0.20)
    stratify: bool = True


class EvaluationConfig(BaseModel):
    """Parâmetros de avaliação e seleção de modelo."""

    primary_metric: str = "roc_auc"
    cv_folds: int = Field(gt=1, default=5)
    decision_threshold: float = Field(gt=0, lt=1, default=0.30)
    fairness_metric: str = "true_positive_rate"
    verbose: int = Field(ge=0, default=0)


class ProjectConfig(BaseModel):
    """Identificação do projeto."""

    name: str
    version: str


class LogisticParams(BaseModel):
    """Hiperparâmetros da Regressão Logística (baseline)."""

    C: float = Field(gt=0, default=1.0)
    max_iter: int = Field(gt=0, default=1000)
    class_weight: str | None = "balanced"
    solver: str = "lbfgs"


class LightGBMParams(BaseModel):
    """Hiperparâmetros do LightGBM."""

    n_estimators: int = Field(gt=0, default=600)
    learning_rate: float = Field(gt=0, le=1, default=0.03)
    num_leaves: int = Field(gt=1, default=31)
    max_depth: int = -1
    min_child_samples: int = Field(gt=0, default=40)
    subsample: float = Field(gt=0, le=1, default=0.8)
    colsample_bytree: float = Field(gt=0, le=1, default=0.8)
    reg_lambda: float = Field(ge=0, default=1.0)
    class_weight: str | None = "balanced"
    n_jobs: int = -1


class CalibrationParams(BaseModel):
    """Parâmetros de calibração de probabilidades."""

    enabled: bool = True
    method: Literal["isotonic", "sigmoid"] = "isotonic"
    cv: int = Field(gt=1, default=5)


class ModelParams(BaseModel):
    """Agrupa os hiperparâmetros de todos os modelos e da calibração."""

    active_model: Literal["logistic", "lightgbm"] = "lightgbm"
    logistic: LogisticParams = LogisticParams()
    lightgbm: LightGBMParams = LightGBMParams()
    calibration: CalibrationParams = CalibrationParams()


class Settings(BaseSettings):
    """Configuração global do projeto, validada por Pydantic.

    Combina ``config.yaml`` e ``model_params.yaml`` (via :func:`get_settings`)
    e variáveis de ambiente do ``.env`` (segredos, nunca commitados).

    Attributes
    ----------
    project : ProjectConfig
        Nome e versão do projeto.
    random_seed : int
        Semente global de reprodutibilidade.
    data : DataConfig
        Parâmetros de dados e splits.
    sensitive_features : list[str]
        Atributos sensíveis para auditoria de justiça.
    evaluation : EvaluationConfig
        Parâmetros de avaliação.
    model : ModelParams
        Hiperparâmetros de modelagem.
    """

    # ``protected_namespaces=()`` evita o aviso do Pydantic por causa do campo
    # ``model`` (que colide com o namespace protegido ``model_``).
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
        protected_namespaces=(),
    )

    project: ProjectConfig
    random_seed: int = 42
    data: DataConfig
    sensitive_features: list[str] = Field(default_factory=list)
    evaluation: EvaluationConfig = EvaluationConfig()
    model: ModelParams = ModelParams()


def _load_yaml(path: Path) -> dict[str, Any]:
    """Lê um arquivo YAML e retorna um dicionário.

    Parameters
    ----------
    path : Path
        Caminho do arquivo YAML.

    Returns
    -------
    dict[str, Any]
        Conteúdo do YAML.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.
    """
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@lru_cache(maxsize=1)
def get_settings(configs_dir: Path | None = None) -> Settings:
    """Carrega e valida a configuração completa do projeto.

    Combina ``config.yaml`` (geral) e ``model_params.yaml`` (modelagem) em um
    único objeto :class:`Settings` validado.

    Parameters
    ----------
    configs_dir : Path, optional
        Diretório dos arquivos de configuração. Se ``None``, usa ``configs/``.

    Returns
    -------
    Settings
        Configuração validada do projeto.

    Examples
    --------
    >>> settings = get_settings()
    >>> settings.data.target
    'Diabetes_binary'
    """
    base = configs_dir or CONFIGS_DIR
    config = _load_yaml(base / "config.yaml")
    model_params = _load_yaml(base / "model_params.yaml")
    config["model"] = model_params
    return Settings(**config)
