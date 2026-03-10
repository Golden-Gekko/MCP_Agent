from pathlib import Path
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient
from loguru import logger

from core import settings


async def init_tools():
    client = MultiServerMCPClient(
        {
            'filesystem': {
                'command': 'npx',
                'args': ['-y', '@modelcontextprotocol/server-filesystem', settings.service.workspace],
                'transport': 'stdio',
            },
            'git': {
                'command': 'uvx',
                'args': ['mcp-server-git', '--repository', settings.service.workspace],
                'transport': 'stdio',
            },
            'context7': {
                'command': 'npx',
                'args': ['-y', '@upstash/context7-mcp'],
                'env': {'CONTEXT7_API_KEY': settings.service.context7_api_key},
                'transport': 'stdio',
            },
            'coder': {
                'command': sys.executable,
                'args': [str(Path(__file__).parent / 'coder_mcp.py')],
                'env': {'PYTHONPATH': str(Path(__file__).parent.parent)},
                'transport': 'stdio',
            }
        }
    )
    tools = await client.get_tools()
    logger.success('MultiServerMCPClient создан')
    return tools
