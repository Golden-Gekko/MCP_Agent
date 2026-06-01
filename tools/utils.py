import os
from pathlib import Path

from loguru import logger


def create_common_env() -> dict:
    common_env = {'PYTHONUTF8': '1', 'NODE_NO_WARNINGS': '1'}
    return {**common_env, 'LC_ALL': 'en_US.UTF-8'} if os.name != 'nt' else common_env


def check_workspace(workspace: Path | str) -> Path:
    try:
        workspace = Path(workspace)
        if not workspace.exists():
            raise FileNotFoundError(f'Директория не создана: {workspace}')
        if not workspace.is_dir():
            raise NotADirectoryError(f'Передана не директория: {workspace}')
    except Exception as e:
        logger.error(f'Ошибка валидации "Workspace": {e}')
        raise
    logger.info(f'Установлена рабочая директория "{str(workspace)}"')
    return workspace.absolute()


def is_git_repo_init(workspace: Path | str) -> bool:
    git_dir = workspace / '.git'
    if not git_dir.exists():
        logger.warning(f'Директория не является git-репозиторием: {workspace}')
        logger.warning('Git-инструменты будут отключены')
        return False
    else:
        logger.success('Git-репозиторий найден')
        return True
