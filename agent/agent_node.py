import json
import re

from json_repair import repair_json
from langchain.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from loguru import logger

from core import settings
from utils.langfuse import load_prompt_from_langfuse

from .state import AgentState


class AgentNode:
    def __init__(
            self,
            llm: BaseChatModel,
            prompt_name: str = 'mcp_agent_prompt',
            prompt_label: str = 'production'
    ):
        self.llm = llm
        self.prompt = load_prompt_from_langfuse(prompt_name=prompt_name, prompt_label=prompt_label)
        logger.debug(f'AgentNode prompt: {self.prompt}')

    async def node(self, state: AgentState) -> dict:
        if state.get('step_iteration', 0) > settings.service.max_step_iterations:
            msg = 'КРИТИЧЕСКАЯ ОШИБКА! Агент зациклился и был сброшен в начальное состояние!'
            logger.error(msg)
            return {
                'messages': [RemoveMessage(id=REMOVE_ALL_MESSAGES), AIMessage(msg)],
                'history': [],
                'current_step': 0,
            }

        messages = [SystemMessage(self.prompt)] + state['messages']
        if state['user_input']:
            messages += [HumanMessage(state['user_input'])]
        response = await self.llm.ainvoke(messages)

        message = self._validate_and_parse_tool_calls(response)
        if not message.tool_calls:
            step_text = state['plan'][state['current_step']]['description']
            message = AIMessage(
                f'Шаг "{step_text}" выполнен. '
                'Подтвердите выполнение словом **"Продолжить"** либо внесите корректировки:\n\n'
                f'ОТВЕТ АГЕНТА:\n{response.content}')

        return {
            'messages': [HumanMessage(state['user_request']), message] if state['user_input'] else [message],
            'step_iteration': state.get('step_iteration', 0) + 1,
            'phase': 'executing',
            'is_approved': False,
        }

    @staticmethod
    def _validate_and_parse_tool_calls(message: AIMessage) -> AIMessage:
        if message.tool_calls or message.response_metadata.get('finish_reason') != 'tool_calls':
            return message

        json_match = re.search(r'\{.*\}', message.content, re.DOTALL)
        if not json_match:
            return message

        raw_json_str = json_match.group()
        try:
            raw_data = json.loads(raw_json_str)
        except json.JSONDecodeError as je:
            logger.warning(f'Ошибка парсинга: {je}. Попытка восстановить')
            try:
                raw_data = json.loads(repair_json(raw_json_str, skip_json_loads=True))
            except Exception as e:
                logger.error(f'Не удалось восстановить JSON через json-repair: {e}')
                return message
        if isinstance(raw_data, dict) and 'name' in raw_data and 'arguments' in raw_data:
            logger.info(f"Успешно извлечен tool_call: {raw_data['name']}")
            logger.success(raw_data)
            message.tool_calls = [{
                'name': raw_data['name'],
                'args': raw_data['arguments'],
                'id': f"call_{hash(raw_data['name'])}"
            }]
            message.content = ''
        return message
