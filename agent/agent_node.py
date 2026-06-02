from langchain.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
        prompt_with_history = ChatPromptTemplate.from_messages([
            ('system', self.prompt),
            ('human', f'ТЕКУШИЙ ШАГ: {step_text}'),
            MessagesPlaceholder(variable_name='messages'),
        ])
        chain = prompt_with_history | self.llm
        response = chain.invoke({'messages': state['messages']})
        return {'messages': [response]}
