from langchain.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from loguru import logger

from utils.langfuse import load_prompt_from_langfuse

from .state import AgentState, Workflow


class PlanerNode:
    def __init__(
            self,
            llm: BaseChatModel,
            prompt_name: str = 'mcp_agent_planer_prompt',
            prompt_label: str = 'production'
    ):
        self.llm = llm.with_structured_output(Workflow)
        self.prompt = load_prompt_from_langfuse(prompt_name=prompt_name, prompt_label=prompt_label)
        logger.debug(f'PlanerNode prompt: {self.prompt}')

    async def node(self, state: AgentState) -> dict:
        logger.info(f'PlanerNode state: {state}')
        response: Workflow = await self.llm.ainvoke(
            [SystemMessage(content=self.prompt)] +
            state['messages'] +
            [HumanMessage(content=state.get('user_request', ''))]
        )

        plan = [{**item.model_dump(), 'done': False} for item in response.plan]
        message = 'Проверьте план действий. Подтвердите план словом "Продолжить" либо внесите корректировки'
        for i, item in enumerate(plan, start=1):
            message += f"\n* Шаг {i}: {item.get('description', 'Ошибка получения шага')}"

        return {
            'messages': [
                HumanMessage(content=state.get('user_request', '')),
                AIMessage(content=message)],
            'plan': plan,
            'current_step': 0,
            'phase': 'planning',
        }
