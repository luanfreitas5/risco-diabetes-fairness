# Risco de Diabetes com Auditoria de Justiça

Modelo de **triagem de risco de diabetes** (survey BRFSS 2015) com foco em
**IA responsável**: auditoria de justiça por subgrupos, calibração de
probabilidades, interpretabilidade com SHAP e um Model Card documentando
limitações e uso previsto.

## Destaques

- **Contratos de dados** (Pandera) validados em cada fronteira do pipeline.
- **Avaliação rigorosa:** ROC-AUC com intervalo de confiança, PR-AUC, recall no
  limiar operacional e Brier score (calibração).
- **Justiça (fairness):** métricas por subgrupo (sexo, faixa etária, renda,
  escolaridade) com `fairlearn`.
- **Interpretabilidade:** importância global via SHAP.
- **Dashboard Streamlit** para triagem interativa.

## Navegação

- [Instalação e uso](guias/uso.md)
- [Arquitetura](guias/arquitetura.md)
- [Referência da API](referencia.md)

!!! warning "Aviso"
    Ferramenta educacional de portfólio. **Não** substitui avaliação médica e
    **não** constitui diagnóstico.
