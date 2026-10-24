# syntax=docker/dockerfile:1.7
# Servidor MLflow (tracking + registry) com backend SQLite: uso local de uma pessoa só.
# Para uso compartilhado, troque por Postgres (driver psycopg2-binary + serviço db).
# A imagem oficial roda como root; aqui o usuário é não-root e /mlflow nasce com o dono
# certo ANTES do volume nomeado, que herda a permissão.
# Mantenha a tag igual à versão do mlflow no uv.lock (cliente e servidor compatíveis):
# após `uv lock`, confira com `uv pip show mlflow` e ajuste esta tag ao patch correspondente.
FROM ghcr.io/mlflow/mlflow:v3.1.1
RUN groupadd --system --gid 10001 mlflow \
    && useradd --system --uid 10001 --gid mlflow --no-create-home mlflow \
    && mkdir -p /mlflow \
    && chown mlflow:mlflow /mlflow
USER mlflow:mlflow
EXPOSE 5000
