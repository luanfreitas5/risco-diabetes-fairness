# --- Configuração ----------------------------------------------------------
# Atalhos reprodutíveis. `make check` replica o CI localmente.
UV := uv
RUN := $(UV) run python src/main.py    # 'src' vira raiz do path ao rodar o script
IMAGE := risco-diabetes-fairness

.DEFAULT_GOAL := help
.PHONY: help install install-app hooks pre-commit update-hooks update lock \
	lint format typecheck security quality smoke test check \
	preprocess train evaluate audit pipeline reproduce app \
	mlflow-ui docker-build docker-up docker-down \
	docs docs-serve docs-deploy release profile \
	clean clean-data clean-reports clean-outputs clean-notebooks

help:  ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --- Ambiente ---------------------------------------------------------------
install:  ## Instala todas as dependências travadas (todos os grupos) e os hooks
	$(UV) sync --all-groups --no-group profiling
	$(UV) run pre-commit install --hook-type pre-commit --hook-type commit-msg

install-app:  ## Instala também o extra do dashboard Streamlit
	$(UV) sync --all-groups --no-group profiling --extra app

hooks:  ## Instala os hooks do pre-commit (pre-commit + commit-msg)
	$(UV) run pre-commit install --hook-type pre-commit --hook-type commit-msg
	$(UV) run detect-secrets scan > .secrets.baseline

pre-commit:  ## Roda todos os hooks do pre-commit em todos os arquivos
	$(UV) run pre-commit run --all-files

update-hooks:  ## Atualiza as versões fixadas dos hooks
	$(UV) run pre-commit autoupdate

lock:  ## Atualiza o uv.lock (commite o resultado)
	$(UV) lock

update:  ## Atualiza todas as dependências e sincroniza
	$(UV) lock --upgrade
	$(UV) sync --all-groups --no-group profiling

# --- Qualidade --------------------------------------------------------------
lint:  ## Lint + verificação de formatação (ruff)
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

format:  ## Corrige lint e formata o código (ruff)
	$(UV) run ruff check --fix src tests
	$(UV) run ruff format src tests

typecheck:  ## Type checking estático (basedPyright)
	$(UV) run basedpyright

security:  ## Segurança do código e das dependências (bandit + pip-audit)
	$(UV) run bandit -c pyproject.toml -r src
	$(UV) run pip-audit

quality:  ## Código morto, modernização, complexidade e docstrings
	$(UV) run vulture
	$(UV) run refurb src
	$(UV) run xenon --max-absolute B --max-modules A --max-average A src
	$(UV) run interrogate -v src

smoke:  ## Smoke tests rápidos (os mesmos do pre-commit)
	$(UV) run pytest -m smoke -q --no-cov

test:  ## Suíte completa com cobertura (>= 80%)
	$(UV) run pytest -m "not slow" --cov --cov-report=term-missing --cov-report=xml

check: lint typecheck security quality test  ## Roda a suíte completa do CI localmente

# --- Pipeline de ML (bronze -> silver -> gold -> modelo -> relatórios) -------
preprocess:  ## Valida o bronze e gera silver e gold em data/
	$(RUN) --stage preprocess

train:  ## Treina baseline + LightGBM, registra no MLflow e persiste o modelo
	$(RUN) --stage train

evaluate:  ## Métricas de holdout, calibração, curvas e SHAP -> reports/
	$(RUN) --stage evaluate

audit:  ## Auditoria de justiça por subgrupos sensíveis -> reports/
	$(RUN) --stage audit

pipeline:  ## Executa o pipeline completo ponta a ponta (src/main.py)
	$(RUN) --stage all

reproduce: pipeline  ## Reproduz todos os resultados a partir do bronze (sem DVC neste projeto)

app:  ## Sobe o dashboard Streamlit (requer: make install-app)
	$(UV) run streamlit run app/app.py

# --- MLflow e containers ----------------------------------------------------
mlflow-ui:  ## UI do MLflow sobre o SQLite local (http://127.0.0.1:5000)
	$(UV) run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --host 127.0.0.1 --port 5000

docker-build:  ## Constrói a imagem do dashboard
	docker build -t $(IMAGE):dev .

docker-up:  ## Sobe o dashboard e, com o perfil tracking, o servidor MLflow
	docker compose --profile tracking up --build -d

docker-down:  ## Derruba os serviços do docker compose
	docker compose --profile tracking down

# --- Documentação e release -------------------------------------------------
docs:  ## Constrói a documentação (modo estrito)
	$(UV) run mkdocs build --strict

docs-serve:  ## Servidor local da documentação
	$(UV) run mkdocs serve

docs-deploy:  ## Publica a documentação no GitHub Pages
	$(UV) run mkdocs gh-deploy --force

release:  ## Nova versão (SemVer) + CHANGELOG + tag via commitizen
	$(UV) run cz bump --changelog --yes

profile:  ## Profiling com scalene (uv sync --group profiling)
	$(UV) run scalene src/main.py

# --- Limpeza ----------------------------------------------------------------
clean:  ## Remove caches e artefatos temporários
	rm -rf .pytest_cache .ruff_cache .mypy_cache .hypothesis .coverage htmlcov coverage.xml site
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +

clean-data:  ## Remove silver e gold (o bronze é imutável e nunca é apagado)
	rm -f data/silver/*.parquet data/gold/*.parquet data/gold/manifest.json

clean-reports:  ## Remove figuras e métricas geradas
	rm -f reports/figures/*.png reports/figures/*.svg reports/metrics/*.json

clean-outputs: clean-data clean-reports  ## Remove todas as saídas do pipeline

clean-notebooks:  ## Remove os outputs dos notebooks
	$(UV) run nbstripout notebooks/*.ipynb
