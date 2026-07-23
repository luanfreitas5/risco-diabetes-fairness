# Model Card — Triagem de Risco de Diabetes

> Template de Model Card (Mitchell et al., 2019). Preencha os campos marcados com
> `⟨...⟩` após rodar `make train`, `make evaluate` e `make audit`. Valores
> numéricos são gerados em `reports/metrics/`.

## 1. Detalhes do modelo

- **Nome:** risco-diabetes-fairness
- **Versão:** 0.1.0
- **Tipo:** Classificador binário (LightGBM calibrado; baseline: Regressão Logística)
- **Objetivo:** Estimar o risco de diabetes/pré-diabetes a partir de fatores de
  risco autorreportados (survey BRFSS 2015).
- **Autor:** Luan Freitas
- **Licença:** MIT

## 2. Uso pretendido

- **Uso previsto:** Ferramenta **educacional/de portfólio** de triagem de risco,
  para demonstrar boas práticas de IA responsável (justiça, calibração, SHAP).
- **Usuários previstos:** Estudo, revisão técnica e demonstração de portfólio.
- **Fora de escopo:** ❌ Não é dispositivo médico. ❌ Não fornece diagnóstico.
  ❌ Não deve embasar decisões clínicas, de seguro ou de emprego.

## 3. Fatores

- **Atributos sensíveis auditados:** `Sex`, `AgeBand` (faixa etária agregada),
  `Income`, `Education`.
- **Grupos relevantes:** ver `reports/metrics/fairness_report.json`.

## 4. Métricas

- **Métrica-alvo (seleção de modelo):** ROC-AUC (independente de limiar).
- **Co-métricas:** PR-AUC (sob prevalência ~14%), Recall no limiar operacional
  (`decision_threshold = 0.30`), Brier score (calibração).
- **Justificativa do limiar:** em triagem, falsos negativos (não sinalizar quem
  está em risco) são mais custosos que falsos positivos; o limiar favorece o
  recall.

| Métrica | Valor (holdout) |
|---|---|
| ROC-AUC | ⟨preencher⟩ |
| PR-AUC | ⟨preencher⟩ |
| Recall @ 0.30 | ⟨preencher⟩ |
| Precisão @ 0.30 | ⟨preencher⟩ |
| Brier score | ⟨preencher⟩ |

## 5. Dados de treino e avaliação

- **Fonte:** Diabetes Health Indicators (BRFSS 2015), ~253.680 respostas.
- **Split:** treino/validação/teste estratificados pela prevalência.
- Ver o Datasheet em `reports/datasheets/datasheet.md`.

## 6. Avaliação por subgrupo (justiça)

⟨Resumir as disparidades de recall e taxa de seleção por `Sex`, `AgeBand`,
`Income` e `Education`, a partir de `reports/metrics/fairness_report.json` e das
figuras `reports/figures/fairness_recall_*.png`.⟩

## 7. Análise de erros

⟨Descrever padrões dos piores erros (falsos negativos), com foco em subgrupos.⟩

## 8. Considerações éticas e limitações

- **Viés de dados:** o BRFSS é autorreportado e pode carregar vieses de
  cobertura e de resposta; a rotulagem de diabetes é por autorrelato.
- **LGPD:** o projeto usa dados públicos e agregados, sem PII; o dashboard não
  coleta nem armazena dados pessoais identificáveis.
- **Limitações:** não generaliza para populações fora do desenho do BRFSS; não
  substitui exames laboratoriais.

## 9. Interpretabilidade

- Importância global via SHAP: `reports/figures/shap_importance.png`.
