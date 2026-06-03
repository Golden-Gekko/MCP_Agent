from langchain.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from loguru import logger
from pyexpat.errors import messages

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
        while not self._validate_and_parse_tool_calls(response):
            messages = state['messages'] + [
                response,
                HumanMessage(load_prompt_from_langfuse('mcp_agent_error_tool_call_prompt'))
            ]
            response = chain.invoke({'messages': messages})
        return {'messages': [response]}

    @staticmethod
    def _validate_and_parse_tool_calls(message: AIMessage) -> bool:
        if message.response_metadata.get('finish_reason') == 'tool_calls' and not message.tool_calls:
            content = message.content
            if '{' in content and 'name' in content:
                logger.warning(f'Модель сгенерировала JSON в content, но tool_calls пуст: {content}')
                return False
        return True
