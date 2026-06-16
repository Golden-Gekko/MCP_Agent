from langchain.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
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
        self.parser = PydanticOutputParser(pydantic_object=Workflow)
        self.llm = llm
        self.prompt = load_prompt_from_langfuse(
            prompt_name=prompt_name, prompt_label=prompt_label
        ).format(format_instructions=self.parser.get_format_instructions())
        logger.debug(f'PlanerNode prompt: {self.prompt}')

    async def node(self, state: AgentState) -> dict:
        logger.debug(f'PlanerNode state: {state}')
        user_input = state.get('user_input', '')
        messages = [SystemMessage(content=self.prompt)] + state['messages']
        if user_input:
            messages += [HumanMessage(content=user_input)]
        response = await self.llm.ainvoke(messages)

        plan = self.parser.parse(response.content).plan
        message = 'Проверьте план действий. Подтвердите план словом **"Продолжить"** либо внесите корректировки'
        for i, item in enumerate(plan, start=1):
            message += f'\n* Шаг {i}: {item}'

        return {
            'messages': [
                HumanMessage(content=user_input), AIMessage(content=message)
            ] if user_input else [AIMessage(content=message)],
            'plan': plan,
            'current_step': 0,
            'phase': 'planning',
            'is_approved': False,
        }
