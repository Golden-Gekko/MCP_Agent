from langchain.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
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
        self.prompt_template = PromptTemplate.from_template(
            load_prompt_from_langfuse(prompt_name=prompt_name, prompt_label=prompt_label))
        logger.debug(f'FinalizerNode prompt_template: {self.prompt_template}')

    async def node(self, state: AgentState) -> dict:
        plan = '\n'.join(state.get('plan', []))

        steps_summary = []
        for i, res in enumerate(state['messages'], start=1):
            if res.type == 'ai':
                steps_summary.append(f'* Шаг {i}: {res.content}')

        prompt_value = self.prompt_template.format(
            user_request=state.get(key='user_request', default='Без запроса'),
            plan=plan,
            steps_summary='\n'.join(steps_summary)
        )
        logger.debug(f'FinalizerNode prompt_value: {prompt_value}')

        response = await self.llm.ainvoke([SystemMessage(content=prompt_value)])
        return {'messages': [AIMessage(content=response.content.strip())]}
