from langchain.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from loguru import logger

from utils.langfuse import load_prompt_from_langfuse

from .state import AgentState


class FinalizerNode:
    def __init__(
            self,
            llm: BaseChatModel,
            prompt_name: str = 'mcp_agent_finalize_prompt',
            prompt_label: str = 'production'
    ):
        self.llm = llm

        self.prompt = load_prompt_from_langfuse(prompt_name=prompt_name, prompt_label=prompt_label)
        logger.debug(f'FinalizerNode prompt: {self.prompt}')

    async def node(self, state: AgentState) -> dict:
        plan = '\n'.join(f"{i}. {s}" for i, s in enumerate(state.get('plan', []), start=1))
        steps_summary = '\n\n'.join(state.get('history', []))

        msg = (
            f'Запрос пользователя: {state["user_request"]}\n\n'
            f'План: \n{plan}\n\n'
            f'Результаты выполнения:\n{steps_summary}\n\n'
            f'Сформируй итоговый ответ для пользователя.')

        logger.debug(f'FinalizerNode message: {msg[:500]}')

        response = await self.llm.ainvoke([
            SystemMessage(content=self.prompt),
            HumanMessage(content=msg),
        ])
        return {'messages': [AIMessage(content=response.content.strip())]}
