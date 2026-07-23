"""Gerenciamento de configuração, caminhos, logging e reprodutibilidade.

Este pacote centraliza toda a configuração do projeto, carregada de arquivos
YAML em ``configs/`` e do ``.env``, e validada em tempo de execução com Pydantic.

Módulos
-------
paths
    Caminhos do projeto (``pathlib.Path``) resolvidos a partir de ``paths.yaml``.
settings
    Carrega e valida ``config.yaml`` e ``model_params.yaml`` com Pydantic.
logging
    Configura o logging (console ``rich`` + arquivo rotativo diário).
environment
    Fixa sementes de aleatoriedade para garantir reprodutibilidade.
"""

from config.paths import ProjectPaths, get_paths
from config.settings import Settings, get_settings

__all__ = ["ProjectPaths", "Settings", "get_paths", "get_settings"]
