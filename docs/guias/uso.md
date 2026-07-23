# Instalação e uso

## Pré-requisitos

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) instalado
- Dataset do [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
  em `data/raw/` (o arquivo `diabetes_binary_health_indicators_BRFSS2015.csv`).

## Instalação

```bash
uv sync --dev            # dependências de runtime + dev
uv sync --extra app      # (opcional) dependências do dashboard Streamlit
make hooks               # instala os hooks do pre-commit
```

## Pipeline

```bash
make preprocess   # valida os dados e gera data/processed
make train        # treina baseline + LightGBM e persiste o melhor modelo
make evaluate     # métricas, calibração, curvas e SHAP -> reports/
make audit        # auditoria de justiça por subgrupo -> reports/
make pipeline     # executa tudo ponta a ponta (src/main.py)
```

Ou uma etapa isolada:

```bash
uv run python -m main --only train
```

## Dashboard

```bash
make app          # uv run streamlit run app/app.py
```

## Qualidade e testes

```bash
make quality      # lint, type check, segurança, complexidade, docstrings
make test         # pytest com cobertura (meta >= 80%)
```
