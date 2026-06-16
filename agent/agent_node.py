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
                'step_iteration': 0,
                'current_step': 0,
            }

        messages = [SystemMessage(self.prompt)] + state['messages']
        if state['user_input']:
            messages += [HumanMessage(state['user_input'])]
        response = await self.llm.ainvoke(messages)

        return {
            'messages': [HumanMessage(state['user_request']), response] if state['user_input'] else [response],
            'step_iteration': state.get('step_iteration', 0) + 1,
            'phase': 'executing',
            'is_approved': False,
        }
