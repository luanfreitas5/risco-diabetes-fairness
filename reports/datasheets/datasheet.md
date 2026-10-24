# Datasheet — Diabetes Health Indicators (BRFSS 2015)

## 1. Motivação

- **Por que foi criado? Para qual tarefa?** O BRFSS (*Behavioral Risk Factor Surveillance System*) é o inquérito telefônico anual de saúde do CDC (EUA). A versão "Diabetes Health Indicators" recorta o BRFSS 2015 em 21 indicadores e um alvo de diabetes para estudo de fatores de risco e classificação.
- **Quem criou e quem financiou?** Coleta: CDC/BRFSS (financiamento público). Recorte para o Kaggle: Alex Teboul. Neste projeto o dataset é usado apenas para fins educacionais e de portfólio, sem financiamento. `TODO(owner): confirmar o nome do autor do recorte na página do Kaggle`.

## 2. Composição

- **O que cada instância representa?** Uma resposta de survey de um adulto residente nos EUA em 2015.
- **Número de instâncias e período coberto:** 253.680 respostas (versão `diabetes_binary_health_indicators`); ano de 2015. Duas variantes ficam em `data/bronze/` e não são usadas: `diabetes_012_*` (alvo com 3 classes) e `diabetes_binary_5050split_*` (balanceada artificialmente).
- **Campos, tipos e contrato:** 21 preditores + alvo, todos numéricos (binários, ordinais e `BMI`/`MentHlth`/`PhysHlth` contínuos). Contratos Pandera: `src/schemas/dataset.py` (`validate_raw` para bronze/silver e `validate_processed` para gold, que acrescenta `AgeBand`).
- **Rótulo:** `Diabetes_binary` — 0 = sem diabetes; 1 = diabetes ou pré-diabetes, **por autorrelato** ("um médico já disse que você tem diabetes?"). Sem janela temporal: é uma condição no momento da entrevista. Prevalência de positivos: 13,93% (`data/gold/manifest.json`).
- **Dados ausentes:** a versão distribuída não tem nulos; ausências do survey original foram removidas pelo recorte do Kaggle (`TODO(owner): documentar o critério de remoção, que não é conhecido`).
- **Campos sensíveis / PII e tratamento:** sem identificadores diretos. Atributos sensíveis presentes: `Sex`, `Age`, `Income`, `Education`; são usados apenas na auditoria de fairness (`AgeBand` é derivada de `Age`).
- **É amostra de uma população maior?** Sim: amostragem complexa do BRFSS (telefonia fixa e celular). O recorte **não** inclui os pesos amostrais, então as estimativas não representam a população dos EUA.

## 3. Processo de coleta

- **Fonte(s) e sistema(s) de origem:** BRFSS 2015 (CDC) → recorte no Kaggle: <https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset>.
- **Período e frequência de coleta:** ano de 2015, coleta contínua ao longo do ano.
- **Consentimento / aviso aos titulares:** participação voluntária em entrevista telefônica, sob as normas do CDC; sem contato com os titulares neste projeto.

## 4. Pré-processamento

Arquitetura medalhão (`make preprocess`):

| Camada | Local | Conteúdo |
|---|---|---|
| bronze | `data/bronze/` | CSVs exatamente como recebidos (imutáveis) + `manifest.json` com o SHA-256 do arquivo lido |
| silver | `data/silver/diabetes_silver.parquet` | linhas duplicadas removidas, tipos e faixas validados pelo contrato bruto |
| gold | `data/gold/diabetes_gold.parquet` | silver + `AgeBand` (jovem 18–39, meia-idade 40–59, idoso 60+), validado e com `manifest.json` |

- **Limpeza, filtragem e linhas removidas por regra:** apenas duplicatas exatas na transição bronze → silver; `TODO(owner): registrar a contagem de linhas removidas (o pipeline loga bronze × silver)`.
- **Anonimização / pseudonimização:** não aplicável (dados públicos sem identificadores diretos).
- **Hash SHA-256 dos dados brutos:** `TODO(owner): rodar make preprocess e copiar o sha256 de data/bronze/manifest.json` · hash do gold (253.680 linhas × 23 colunas): `228123a0fb50021c33cc3bafb8026f1d0a78b293d96bca1924db7fc515dccd6c`.

## 5. Usos

- **Usos pretendidos:** pesquisa e demonstração de classificação de risco, calibração, fairness e interpretabilidade.
- **Usos desaconselhados:** decisões clínicas, de seguro, de crédito ou de emprego; inferência sobre a população brasileira ou sobre indivíduos; estimativa de prevalência (não há pesos amostrais).
- **Riscos de viés conhecidos:** rótulo autorreportado (subnotificação de diabetes não diagnosticado); viés de cobertura e de não resposta da telefonia; grupos pequenos em `Education` = 1 e em `Income` baixa; o modelo treinado detecta menos casos em jovens, renda e escolaridade altas (ver Model Card).

## 6. Distribuição

- **Licença e restrições:** o BRFSS é de domínio público (CDC). A licença do recorte no Kaggle deve ser conferida em sua página antes de redistribuir — por isso os dados não são commitados.
- **Onde está armazenado:** localmente em `data/` (ignorado pelo Git); o download é manual. O projeto ainda não usa DVC; ao adotá-lo, versionar `data/bronze/` com `dvc add`.

## 7. Manutenção

- **Responsável:** Luan Freitas — [github.com/luanfreitas5](https://github.com/luanfreitas5).
- **Frequência de atualização:** nenhuma prevista; dataset estático de 2015. Novas ondas do BRFSS exigem novo bronze e novo hash.
- **Como atender pedidos de exclusão de titulares:** não há dados pessoais nem como localizar um titular nos registros (sem identificador); eventual pedido deve ser encaminhado ao CDC (fonte primária).

## 8. LGPD

- **Base legal (art. 7 / art. 11):** dados públicos e anonimizados; a LGPD não se aplica a dados anonimizados (art. 12), e o tratamento é para estudo/pesquisa (art. 7º, IV; art. 11, II, c). `TODO(owner): validar com o encarregado/DPO se o projeto passar a uso institucional`.
- **Quase-identificadores e risco de reidentificação:** demografia agregada em códigos (idade em faixas, renda e escolaridade ordinais); risco baixo, mas não nulo ao cruzar com outras bases.
- **PII em saídas:** ausente em logs, figuras, métricas e commits (os logs registram apenas contagens, formas e hashes).
- **Prazo de retenção:** enquanto o projeto existir; sem dados do projeto em repositórios remotos.
- **Relatório de impacto (RIPD) necessário?** Não, para este uso educacional com dados públicos anonimizados. `TODO(owner): reavaliar se houver coleta de dados reais via dashboard`.
