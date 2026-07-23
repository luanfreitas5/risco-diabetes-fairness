# Datasheet — Diabetes Health Indicators (BRFSS 2015)

> Template de Datasheet for Datasets (Gebru et al., 2021).

## Motivação

- **Para que o dataset foi criado?** Consolidar indicadores de saúde do
  *Behavioral Risk Factor Surveillance System* (BRFSS) 2015 do CDC (EUA) para
  estudo de fatores de risco de diabetes.
- **Uso neste projeto:** treinar e auditar um classificador de risco de diabetes
  com foco em justiça, calibração e interpretabilidade.

## Composição

- **Instâncias:** ~253.680 respostas de survey (versão binária desbalanceada).
  Versões alternativas: `diabetes_012_*` (multiclasse 0/1/2) e
  `diabetes_binary_5050split_*` (balanceada).
- **Variável-alvo:** `Diabetes_binary` (0 = sem diabetes; 1 = diabetes ou
  pré-diabetes). Prevalência de positivos ~14% na versão principal.
- **Features (21):** demografia (`Sex`, `Age`, `Education`, `Income`), condições
  (`HighBP`, `HighChol`, `Stroke`, `HeartDiseaseorAttack`), hábitos (`Smoker`,
  `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`), IMC (`BMI`) e
  saúde percebida (`GenHlth`, `MentHlth`, `PhysHlth`, `DiffWalk`).
- **Valores ausentes:** o dataset distribuído já vem limpo (sem nulos).

## Processo de coleta

- **Como foi coletado?** Entrevistas telefônicas do BRFSS 2015 (autorrelato).
- **Amostragem:** desenho amostral do BRFSS; representatividade sujeita ao
  desenho original e a vieses de resposta.

## Pré-processamento

- Engenharia de `AgeBand` (agregação das 13 faixas etárias em 3 grupos) para a
  auditoria de justiça — ver `src/features/engineering.py`.
- Validação de contrato (Pandera) em `src/schemas/dataset.py`.
- Sem modificação dos dados brutos: leitura de `data/raw`, escrita em
  `data/processed`.

## Distribuição e licença

- **Fonte:** [Kaggle — Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset).
- **Origem primária:** CDC BRFSS 2015 (domínio público, EUA).
- **Licença:** verificar a licença da página do Kaggle antes de redistribuir.

## Privacidade / LGPD

- **Base legal:** dados públicos, agregados e anonimizados; sem identificadores
  diretos.
- **Quase-identificadores:** demografia agregada; risco de reidentificação baixo.
- **PII:** ausente nas saídas, logs, figuras e commits.

## Manutenção

- **Manifesto de versão:** hash do dataset processado em
  `data/processed/manifest.json` (gerado por `make preprocess`).
