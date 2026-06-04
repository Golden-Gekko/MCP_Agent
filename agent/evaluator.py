from langchain.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import PromptTemplate
from loguru import logger

from utils.langfuse import load_prompt_from_langfuse

from .state import AgentState


class EvaluatorNode:
    def __init__(
            self,
            llm: BaseChatModel,
            prompt_name: str = 'mcp_agent_evaluator_prompt',
            prompt_label: str = 'production'
    ):
        self.llm = llm
        self.prompt_template = PromptTemplate.from_template(
            load_prompt_from_langfuse(prompt_name=prompt_name, prompt_label=prompt_label))
        logger.debug(f'EvaluatorNode prompt_template: {self.prompt_template}')

    async def node(self, state: AgentState) -> dict:
        step_result = state.get('step_content', '')
        prompt_value = self.prompt_template.format(
            step_text=state['plan'][state['current_step']],
            step_result=step_result
        )
        logger.debug(f'EvaluatorNode prompt_value: {prompt_value}')

        response = await self.llm.ainvoke([SystemMessage(content=prompt_value)])
        eval_res = response.content.strip().lower()

        if any(keyword in eval_res for keyword in ['pass', 'retry', 'fail']):
            return {'evaluation': next(kw for kw in ['pass', 'retry', 'fail'] if kw in eval_res)}
        return {'evaluation': 'pass', 'error_log': [f'Evaluator вернул: {eval_res}']}
