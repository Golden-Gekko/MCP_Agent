import json
import re
from typing import Dict

from langchain_core.messages import SystemMessage
from loguru import logger

from core import settings

from .state import AgentState


def planner_node(state: AgentState) -> Dict:
    prompt = (
        settings.langfuse.client.get_prompt(name='mcp_agent_planer_prompt')
        .compile(user_request=state['user_request']))
    response = settings.llm.chat.llm.invoke([SystemMessage(content=prompt)])

    template = {
        'plan': [],
        'current_step': 0,
        'step_results': {},
        'retry_count': 0
    }

    match = re.search(r'\[.*\]', response.content.strip(), re.DOTALL)
    if match:
        try:
            plan = json.loads(match.group())
            if isinstance(plan, list) and len(plan) > 0:
                template['plan'] = plan
                return template
        except json.JSONDecodeError:
            pass

    logger.warning(f'Планировщик не вернул валидный JSON: {response.content}')
    template['plan'] = [f"Выполнить: {state['user_request']}"]
    return template
