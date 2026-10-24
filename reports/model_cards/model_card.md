# Model Card — Risco de Diabetes (LightGBM calibrado) v0.4.0

**Status:** DRAFT — apenas para fins educacionais e de portfólio.
**Bloqueios pendentes:**
- IC 95% das métricas de holdout (bootstrap) — bloqueia qualquer afirmação de desempenho comparativo.
- Teste de significância baseline × LightGBM (t reamostrado corrigido) — bloqueia "melhor que o baseline".
- Run do MLflow e Git SHA — o modelo atual foi treinado antes da integração com o MLflow; bloqueia rastreabilidade completa.
- Validação externa e avaliação do modelo por subgrupo com `n` e IC — bloqueiam qualquer uso fora do portfólio.

## 1. Detalhes do modelo

- **Responsável / contato:** Luan Freitas — [github.com/luanfreitas5](https://github.com/luanfreitas5)
- **Data:** 2026-10-24
- **Algoritmo:** pipeline scikit-learn (pré-processamento `ColumnTransformer` + LightGBM, `class_weight="balanced"`, 600 árvores, `learning_rate=0.03`) com calibração isotônica (`CalibratedClassifierCV`, cv=5). Baseline: Regressão Logística (`class_weight="balanced"`).
- **Rastreabilidade:** Git SHA `TODO(owner): registrar no próximo treino (make train grava git_sha como tag)` · MLflow run `TODO(owner): run_id gravado em reports/metrics/training_summary.json após o próximo make train` · hash dos dados (gold) `228123a0fb50021c33cc3bafb8026f1d0a78b293d96bca1924db7fc515dccd6c`
- **Semente:** 42 (todas as operações estocásticas).
- **Licença:** MIT

## 2. Uso pretendido

- **Uso primário:** demonstrar boas práticas de IA responsável (contrato de dados, avaliação com incerteza, calibração, fairness e SHAP) em uma triagem de risco de diabetes/pré-diabetes a partir de fatores autorreportados.
- **Usuários pretendidos:** revisores técnicos, recrutadores e estudantes de ciência de dados.
- **Humano no circuito:** sim — o dashboard exibe um aviso de que o resultado não é diagnóstico e sugere avaliação clínica.

## 3. Usos fora do escopo

- ❌ Diagnóstico ou qualquer decisão clínica individual (não é dispositivo médico).
- ❌ Decisões de seguro, crédito, emprego ou acesso a serviços de saúde.
- ❌ Priorização de pacientes por renda ou escolaridade: o modelo detecta menos casos nesses grupos (ver seção 8).
- ❌ Populações fora do desenho do BRFSS 2015 (adultos dos EUA): não há validação externa nem para o Brasil.

## 4. Fatores

- **Grupos relevantes:** `Sex`, `AgeBand` (jovem 18–39, meia-idade 40–59, idoso 60+), `Income` (códigos BRFSS 1–8) e `Education` (códigos BRFSS 1–6).
- **Ambientes:** inquérito telefônico BRFSS 2015, respondentes adultos dos EUA; nenhum outro ambiente foi testado.

## 5. Métricas

- **Métrica principal:** ROC-AUC — independente de limiar, usada para seleção de modelo. Co-métricas: PR-AUC (prevalência ~14%), recall e precisão no limiar operacional, Brier score.
- **Limiar de decisão:** 0,30 — escolhido por julgamento (`configs/config.yaml`), priorizando recall: em triagem, o falso negativo (não sinalizar quem tem diabetes) é mais custoso que o falso positivo (encaminhar para exame confirmatório). Não foi otimizado em validação por custo formal: `TODO(owner): calibrar o limiar na partição de validação com custos explícitos de FP/FN`.
- **Calibração:** Brier 0,0973 no holdout; curva de confiabilidade em `reports/figures/calibration_curve.png`.

## 6. Dados de avaliação

Holdout estratificado de 20% do BRFSS 2015 (≈ 50,7 mil respostas), mesma distribuição do treino (split aleatório, sem separação temporal ou geográfica) e prevalência de ~13,9% de positivos. O split de validação (20%) é reservado para ajustes; o treino usa 60% (152.208 linhas).

## 7. Dados de treino

BRFSS 2015 (253.680 respostas, 21 preditores, alvo `Diabetes_binary`). Detalhes, composição e LGPD no [Datasheet](../datasheets/datasheet.md).

## 8. Análise quantitativa

**Geral (holdout)** — fonte: `reports/metrics/holdout_metrics.json`.

| Recorte | n | Métrica | Valor | IC 95% | Baseline |
|---|---|---|---|---|---|
| Holdout | ≈ 50,7 mil | ROC-AUC | 0,8272 | `TODO(owner): bootstrap` | `TODO(owner)`: o baseline só tem CV |
| Holdout | ≈ 50,7 mil | PR-AUC | 0,4235 | `TODO(owner): bootstrap` | `TODO(owner)` |
| Holdout (limiar 0,30) | ≈ 50,7 mil | Recall | 0,4692 | `TODO(owner): bootstrap` | `TODO(owner)` |
| Holdout (limiar 0,30) | ≈ 50,7 mil | Precisão | 0,4320 | `TODO(owner): bootstrap` | `TODO(owner)` |
| Holdout (limiar 0,30) | ≈ 50,7 mil | Brier | 0,0973 | `TODO(owner): bootstrap` | `TODO(owner)` |

**Validação cruzada (5 folds, treino)** — fonte: `reports/metrics/training_summary.json`.

| Modelo | ROC-AUC (média ± IC 95%) |
|---|---|
| Baseline: Regressão Logística | 0,8240 ± 0,0018 |
| LightGBM (ativo) | 0,8314 ± 0,0021 |

Os intervalos não se sobrepõem e a diferença é de ~0,007 de AUC, porém **não foi feito teste formal de significância** (`TODO(owner): t reamostrado corrigido`). O ganho é pequeno: o baseline linear já captura a maior parte do sinal.

**Por subgrupo — recall no limiar 0,30** (fonte: `reports/metrics/fairness_report.json`; `n` por grupo e IC: `TODO(owner): incluir n e bootstrap por grupo`).

| Atributo | Recall por grupo | Diferença máx. de recall | Razão min/máx |
|---|---|---|---|
| `Sex` (0 = feminino, 1 = masculino) | 0,473 · 0,465 | 0,008 | 0,98 |
| `AgeBand` | jovem 0,159 · meia-idade 0,403 · idoso 0,512 | 0,353 | 0,31 |
| `Income` (1 → 8) | 0,644 · 0,657 · 0,595 · 0,580 · 0,512 · 0,418 · 0,385 · 0,285 | 0,372 | 0,43 |
| `Education` (1 → 6) | 0,857\* · 0,706 · 0,605 · 0,511 · 0,485 · 0,349 | 0,509 | 0,41 |

\* Educação nível 1 tem provavelmente poucas observações positivas; o valor é instável (`TODO(owner): confirmar n`).

**Pior segmento:** jovens (18–39) com recall de apenas 0,159, ou seja, o modelo deixa passar ~84% dos jovens com diabetes nesse limiar; renda e escolaridade mais altas também têm recall menor (0,285 e 0,349).

**Importância global (SHAP, média |valor|):** `GenHlth` 0,64 · `Age` 0,46 · `BMI` 0,45 · `HighBP` 0,37 · `HighChol` 0,31 (`reports/metrics/shap_importance.json`).

**Análise de erros:** `TODO(owner): inspecionar os piores falsos negativos (jovens, renda alta) e documentar padrões.`

## 9. Considerações éticas e fairness

- **Atributos sensíveis auditados:** `Sex`, `AgeBand`, `Income`, `Education` (somente avaliação; `AgeBand` não é feature do modelo).
- **Disparidades encontradas:** paridade de recall adequada para `Sex` (diferença 0,008); **grande disparidade** em `AgeBand` (0,353), `Income` (0,372) e `Education` (0,509). A taxa de seleção também varia muito (diferença de 0,23 em idade, 0,32 em renda e 0,36 em escolaridade). Parte da diferença decorre de prevalências distintas e de um limiar único; não é prova de causalidade.
- **Mitigação aplicada:** nenhuma. `TODO(owner): avaliar limiares por grupo ou ExponentiatedGradient (fairlearn) e discutir o trade-off com a calibração.`
- **Base legal LGPD:** dados públicos e anonimizados, sem identificadores diretos (ver Datasheet). O dashboard não coleta nem armazena dados pessoais.

## 10. Limitações, riscos e monitoramento

- **Limitações conhecidas:** rótulo autorreportado (erro de classificação e viés de resposta); diabetes e pré-diabetes agrupados; ausência de validação externa; recall baixo (0,47) no limiar 0,30, ou seja, mais da metade dos casos não é sinalizada; sem IC no holdout.
- **Mudanças de distribuição previstas:** dados de 2015 (EUA); mudanças de perfil populacional, de critérios diagnósticos e de hábitos após 2015 não estão refletidas.
- **Monitoramento:** não há modelo em produção; se houver implantação, monitorar drift de dados/predição (Evidently) e recall por subgrupo mensalmente, com alerta para queda > 5 p.p. `TODO(owner)`.
- **Critério de retreino / descontinuação:** retreinar com nova onda do BRFSS; descontinuar se a disparidade de recall entre subgrupos não for tratada antes de qualquer uso real.
