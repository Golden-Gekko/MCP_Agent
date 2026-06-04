import json
import re

from json_repair import repair_json
from langchain.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
                'messages': [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
                'history': [],
                'current_step': 0,
                'error_log': state.get('error_log', []) + [msg]
            }

        step_text = state['plan'][state['current_step']]
        prompt_with_history = ChatPromptTemplate.from_messages([
            ('system', self.prompt),
            ('human', f"Запрос пользователя: {state['user_request']}"),
            ('human', f"ТЕКУЩИЙ ШАГ, КОТОРЫЙ НУЖНО ВЫПОЛНИТЬ: {step_text}"),
            MessagesPlaceholder(variable_name='messages'),
        ])
        chain = prompt_with_history | self.llm
        response = chain.invoke({'messages': state['messages']})

        retry = 0

        # while not self._validate_and_parse_tool_calls(response) and retry < settings.service.max_retries:
        #     messages = state['messages'] + [
        #         response,
        #         HumanMessage(load_prompt_from_langfuse('mcp_agent_error_tool_call_prompt'))
        #     ]
        #     response = chain.invoke({'messages': messages})
        #     retry += 1
        return {'messages': [response], 'step_iteration': state.get('step_iteration', 0) + 1}

    @staticmethod
    def _validate_and_parse_tool_calls(message: AIMessage) -> bool:
        if message.tool_calls:
            return True
        if message.response_metadata.get('finish_reason') == 'tool_calls':
            json_match = re.search(r'\{.*\}', message.content, re.DOTALL)

            if json_match:
                raw_json_str = json_match.group()
                try:
                    raw_data = json.loads(raw_json_str)
                except json.JSONDecodeError as je:
                    logger.warning(f'Ошибка парсинга: {je}. Попытка восстановить')
                    try:
                        raw_data = json.loads(repair_json(raw_json_str, skip_json_loads=True))
                    except Exception as e:
                        logger.error(f'Не удалось восстановить JSON через json-repair: {e}')
                        return False
                if isinstance(raw_data, dict) and 'name' in raw_data and 'arguments' in raw_data:
                    logger.info(f"Успешно извлечен tool_call: {raw_data['name']}")
                    message.tool_calls = [{
                        'name': raw_data['name'],
                        'args': raw_data['arguments'],
                        'id': f"call_{hash(raw_data['name'])}"
                    }]
                    message.content = ""
                    return True
        return False
