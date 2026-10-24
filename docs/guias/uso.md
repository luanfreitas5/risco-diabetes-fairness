# Instalação e uso

## Pré-requisitos

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) instalado
- Dataset do [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
  em `data/bronze/` (o arquivo `diabetes_binary_health_indicators_BRFSS2015.csv`).

## Instalação

```bash
make install            # uv sync --all-groups + hooks do pre-commit
make install-app        # (opcional) inclui o dashboard Streamlit
```

## Pipeline

```bash
make preprocess   # bronze -> silver -> gold (data/gold)
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
