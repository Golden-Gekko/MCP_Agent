from langchain.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

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
        step_text = state['plan'][state['current_step']]
        messages = state['messages'] + [
            SystemMessage(content=self.prompt), HumanMessage(content=f'ТЕКУШИЙ ШАГ: {step_text}')]

        response = await self.llm.ainvoke(messages)
        return {'messages': [response]}
