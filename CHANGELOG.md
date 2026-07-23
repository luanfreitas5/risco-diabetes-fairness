# Changelog

Todas as mudanças notáveis deste projeto são documentadas aqui.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/)
e o versionamento segue [Semantic Versioning](https://semver.org/lang/pt-BR/).

## [Não lançado]

### Adicionado
- Estrutura inicial do projeto (configs, `src/`, testes, docs, CI).
- Contratos de dados (Pandera) para os estágios bruto e processado.
- Pipeline de pré-processamento com engenharia de `AgeBand` e hash do dataset.
- Fábrica de modelos: baseline (Regressão Logística) e LightGBM, com calibração.
- Avaliação com validação cruzada (IC 95%), calibração, ROC/PR e SHAP.
- Auditoria de justiça por subgrupo (`Sex`, `AgeBand`, `Income`, `Education`)
  com `fairlearn`.
- Dashboard Streamlit de triagem interativa.
- Model Card e Datasheet.

## [0.1.0] - 2026-07-06

### Adicionado
- Versão inicial do esqueleto do projeto.
