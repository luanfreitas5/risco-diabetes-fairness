# 🩺 Risco de Diabetes com Auditoria de Justiça

[![CI](https://github.com/luanfreitas5/risco-diabetes-fairness/actions/workflows/ci.yml/badge.svg)](https://github.com/luanfreitas5/risco-diabetes-fairness/actions/workflows/ci.yml)
[![Tests](https://github.com/luanfreitas5/risco-diabetes-fairness/actions/workflows/tests.yml/badge.svg)](https://github.com/luanfreitas5/risco-diabetes-fairness/actions/workflows/tests.yml)
[![Coverage](https://img.shields.io/badge/coverage-%E2%89%A580%25-brightgreen)](https://github.com/luanfreitas5/risco-diabetes-fairness)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/style-ruff-000000)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Modelo de **triagem de risco de diabetes** a partir de fatores de risco
autorreportados (survey **BRFSS 2015**, ~253 mil respostas), com ênfase em
**IA responsável**: auditoria de justiça por subgrupos, calibração de
probabilidades, interpretabilidade com SHAP e um **Model Card** documentando
limitações e uso previsto.

> ⚠️ **Aviso:** ferramenta **educacional / de portfólio**. Não substitui
> avaliação médica profissional e **não** constitui diagnóstico.

---

## ✨ Destaques (senior bar)

- 🧪 **Contratos de dados** (Pandera) validados em cada fronteira do pipeline.
- 📊 **Avaliação rigorosa:** ROC-AUC com **intervalo de confiança**, PR-AUC (sob
  prevalência ~14%), **recall** no limiar operacional e **Brier score**.
- ⚖️ **Justiça (fairness):** métricas por subgrupo — `Sex`, `AgeBand`, `Income`,
  `Education` — com `fairlearn`.
- 🎯 **Calibração** de probabilidades (`CalibratedClassifierCV`), essencial para
  triagem.
- 🔍 **Interpretabilidade** com **SHAP** (importância global das features).
- 🖥️ **Dashboard Streamlit** (Plotly) para triagem interativa.
- 🔁 **Reprodutibilidade:** seeds fixas, hash do dataset e lock file commitado.

---

## 🧱 Stack

`Polars` · `scikit-learn` · `LightGBM` · `SHAP` · `Fairlearn` · `Pandera` ·
`Pydantic` · `Streamlit` · `Plotly` · `uv` · `ruff` · `pytest`

---

## 📂 Estrutura

```
├── configs/            # YAML validados por Pydantic
├── data/{raw,processed}
├── src/
│   ├── config/         # settings, paths, logging, reprodutibilidade
│   ├── constants/      # colunas, rótulos, métricas
│   ├── schemas/        # contratos Pandera
│   ├── data/           # loader, writer, splitter
│   ├── features/       # engenharia (AgeBand)
│   ├── preprocessing/  # ColumnTransformer
│   ├── models/         # fábrica + persistência
│   ├── evaluation/     # CV com IC, calibração
│   ├── metrics/        # métricas de classificação
│   ├── fairness/       # auditoria por subgrupo
│   ├── explainability/ # SHAP
│   ├── visualization/  # tema + figuras
│   ├── pipelines/      # orquestração por etapa
│   ├── main.py  # interface de linha de comando
├── app/                # dashboard Streamlit
├── tests/              # unit, property-based, comportamentais
├── reports/{figures,metrics,model_cards,datasheets}
└── docs/               # MkDocs Material
```

---

## 🚀 Início rápido

### 1. Dados

Baixe o [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
e coloque `diabetes_binary_health_indicators_BRFSS2015.csv` em `data/raw/`.

### 2. Ambiente

```bash
uv sync --dev          # runtime + dev
uv sync --extra app    # (opcional) dashboard Streamlit
make hooks             # pre-commit
```

### 3. Pipeline

```bash
make preprocess   # valida e gera data/processed
make train        # baseline + LightGBM, persiste o melhor
make evaluate     # métricas, calibração, curvas, SHAP -> reports/
make audit        # auditoria de justiça -> reports/
# ou tudo de uma vez:
make pipeline
```

### 4. Dashboard

```bash
make app
```

---

## 📈 Metodologia

| Decisão | Justificativa |
|---|---|
| **Dataset desbalanceado (real, ~14%)** | Evita treinar em distribuição artificial; desbalanceamento tratado com `class_weight` + métricas adequadas. |
| **Métrica-alvo: ROC-AUC** | Independente de limiar para seleção de modelo; PR-AUC e recall como co-métricas. |
| **Limiar 0.30** | Triagem prioriza **recall** — falsos negativos são o erro mais custoso. |
| **Calibração** | Probabilidades embasam a triagem, então precisam ser confiáveis (Brier). |
| **AgeBand só para auditoria** | Preserva granularidade preditiva e garante subgrupos robustos. |

Detalhes no [Model Card](reports/model_cards/model_card.md) e no
[Datasheet](reports/datasheets/datasheet.md).

---

## 🧰 Qualidade

```bash
make quality   # lint, type check, segurança, complexidade, docstrings
make test      # pytest com cobertura (>= 80%)
```

---

## 📚 Documentação

```bash
make docs-serve   # http://127.0.0.1:8000
```

---

## 📝 Licença

Distribuído sob a licença [MIT](LICENSE).
