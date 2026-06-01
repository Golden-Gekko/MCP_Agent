from pathlib import Path
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient
from loguru import logger

from core import settings

from .utils import check_workspace, create_common_env, is_git_repo_init


async def init_tools():
    common_env = create_common_env()
    workspace = check_workspace(settings.service.workspace)

    mcp_config = {
        'filesystem': {
            'command': 'npx',
            'args': ['-y', '@modelcontextprotocol/server-filesystem', str(workspace)],
            'env': common_env,
            'cwd': str(workspace),
            'transport': 'stdio',
        },
        'context7': {
            'command': 'npx',
            'args': ['-y', '@upstash/context7-mcp'],
            'env': {**common_env, 'CONTEXT7_API_KEY': settings.service.context7_api_key},
            'transport': 'stdio',
        },
        'coder': {
            'command': sys.executable,
            'args': [str(Path(__file__).parent / 'coder_mcp.py')],
            'env': {**common_env, 'PYTHONPATH': str(Path(__file__).parent.parent)},
            'transport': 'stdio',
        }
    }
    if is_git_repo_init(workspace):
        mcp_config['git'] = {
            'command': 'uvx',
            'args': ['mcp-server-git', '--repository', str(workspace)],
            'env': {
                **common_env,
                'UV_INDEX_URL': 'https://mirrors.cloud.tencent.com/pypi/simple/'
            },
            'cwd': str(workspace),
            'transport': 'stdio',
        }

    try:
        mcp_client = MultiServerMCPClient(mcp_config)
        tools = await mcp_client.get_tools()
        logger.success(f'Загружено {len(tools)} инструментов')
    except Exception as e:
        logger.error(f'Ошибка создания MultiServerMCPClient: {e}')
        logger.debug(f'Конфигурация: {mcp_config}')
        raise

    return tools
