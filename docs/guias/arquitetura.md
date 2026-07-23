# Arquitetura

O projeto segue organização por funcionalidade (não por camadas técnicas),
baixo acoplamento e alta coesão. Os módulos em `src/` são importados como
top-level (ex.: `from data.loader import load_raw`).

## Fluxo do pipeline

```
data/raw (CSV)
   │  load_raw + validate_raw (Pandera)          [data, schemas]
   ▼
engineer_features (AgeBand)                       [features]
   │  validate_processed
   ▼
data/processed (parquet + manifest)               [data.writer, utils.hashing]
   │  split_data (estratificado)                  [data.splitter]
   ▼
build_model (baseline + LightGBM, calibrado)      [models, preprocessing]
   │  cross_validate_model (IC 95%)               [evaluation]
   ▼
models/model.joblib (+ metadados)                 [models.persistence]
   │
   ├── evaluate: métricas, calibração, ROC/PR, SHAP   [evaluation, metrics, explainability, visualization]
   └── audit: justiça por subgrupo (fairlearn)        [fairness]
```

## Camadas

| Pacote | Responsabilidade |
|---|---|
| `config` | Configuração, caminhos, logging, reprodutibilidade |
| `constants` | Nomes de colunas, rótulos e métricas |
| `schemas` | Contratos de dados (Pandera) |
| `data` | Carregamento, escrita e particionamento |
| `features` | Engenharia de features derivadas |
| `preprocessing` | Transformações (ColumnTransformer) |
| `models` | Fábrica e persistência de modelos |
| `evaluation` / `metrics` | Avaliação, calibração e métricas |
| `fairness` | Auditoria de justiça por subgrupo |
| `explainability` | SHAP |
| `visualization` | Tema e figuras |
| `pipelines` | Orquestração de cada etapa |
| `cli` / `main` | Interface de linha de comando |

## Decisões (defensáveis em entrevista)

- **Dataset desbalanceado (versão real, ~14%)** em vez do 50/50: evita treinar
  numa distribuição artificial; o desbalanceamento é tratado com
  `class_weight="balanced"` e métricas adequadas (PR-AUC, recall).
- **Calibração** (`CalibratedClassifierCV`): as probabilidades embasam a decisão
  de triagem, então precisam ser confiáveis (Brier score reportado).
- **AgeBand** só para auditoria (não é feature do modelo): evita reduzir a
  granularidade preditiva e garante subgrupos com tamanho amostral robusto.
