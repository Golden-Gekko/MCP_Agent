import json
import re

from langchain.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import PromptTemplate
from loguru import logger

from utils.langfuse import load_prompt_from_langfuse

from .state import AgentState


class PlanerNode:
    def __init__(
            self,
            llm: BaseChatModel,
            prompt_name: str = 'mcp_agent_planer_prompt',
            prompt_label: str = 'production'
    ):
        self.llm = llm
        self.prompt_template = PromptTemplate.from_template(
            load_prompt_from_langfuse(prompt_name=prompt_name, prompt_label=prompt_label))
        logger.debug(f'PlanerNode prompt_template: {self.prompt_template}')

    async def node(self, state: AgentState) -> dict:
        prompt_value = self.prompt_template.format(
            user_request=state['user_request'])
        logger.debug(f'PlanerNode prompt_value: {prompt_value}')

        response = await self.llm.ainvoke([SystemMessage(content=prompt_value)])

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
