# =============================================================================
# Dockerfile — build multi-stage com uv. Imagem final enxuta e não-root.
# Serve o dashboard Streamlit por padrão.
# =============================================================================

# --- Stage 1: builder -------------------------------------------------------
FROM python:3.12-slim AS builder

# uv oficial (binário estático).
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Instala dependências primeiro (camada cacheável).
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-install-project --extra app || uv sync --no-install-project --extra app

# Copia o código e instala o projeto.
COPY . .
RUN uv sync --frozen --extra app || uv sync --extra app

# --- Stage 2: runtime -------------------------------------------------------
FROM python:3.12-slim AS runtime

# Usuário não-root.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# Copia o ambiente virtual e o código do builder.
COPY --from=builder --chown=appuser:appuser /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8501

# Sobe o dashboard Streamlit.
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
