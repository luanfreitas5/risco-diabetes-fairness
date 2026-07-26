## v0.4.0 (2026-07-26)

### Feat

- atualiza changelog e ajusta workflow de release

## v0.3.0 (2026-07-26)

### Feat

- adiciona anotação de tipo em curvas ROC e precisão-recall
- ajusta exibição de curvas ROC e precisão-recall
- atualiza versão do projeto para 0.3.0 e ajusta changelog
- renomeia variáveis para maior clareza e adiciona método to_numpy na classe ShapResult
- atualiza IDs das células nos notebooks de EDA e modelagem
- torna obrigatório o tipo de bump no release e atualiza a matriz de versões do Python nos testes
- atualiza versão do projeto e ajusta configuração do Makefile
- atualiza versões de dependências no pre-commit
- atualiza a configuração do baseline de segredos
- adiciona workflows do GitHub Actions e configurações de dependências
- adiciona configuração inicial do projeto com Docker e Makefile
- adiciona configuração do ambiente e documentação inicial
- adiciona arquivos de configuração do projeto
- adiciona arquivos de configuração e documentação inicial
- adiciona notebooks de EDA e modelagem Adiciona o notebook de Análise Exploratória (EDA) e o notebook de Modelagem e Avaliação, incluindo importações e estrutura inicial para análise de risco de diabetes.
- adiciona dashboard de triagem de risco de diabetes Implementa uma ferramenta educacional com coleta de fatores de risco, cálculo de probabilidade e visualização com medidor Plotly.
- adiciona pipelines de pré-processamento, treino, avaliação e auditoria de justiça

### Fix

- bootstrap release workflow when repository has no tags
- define valor padrão para incremento como "MAJOR"

### Refactor

- simplifica o workflow de release
