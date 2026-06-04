from langchain.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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
        self.prompt = load_prompt_from_langfuse(prompt_name=prompt_name, prompt_label=prompt_label)
        logger.debug(f'EvaluatorNode prompt: {self.prompt}')

    async def node(self, state: AgentState) -> dict:
        response = await self.llm.ainvoke([
            SystemMessage(content=self.prompt),
            HumanMessage(content=f"ЗАДАНИЕ: {state['plan'][state['current_step']]}"),
            AIMessage(content=state.get('step_content', 'РЕЗУЛЬТАТА НЕТ'))
        ])
        eval_res = response.content.strip().lower()

        if any(keyword in eval_res for keyword in ['pass', 'retry', 'fail']):
            return {'evaluation': next(kw for kw in ['pass', 'retry', 'fail'] if kw in eval_res)}
        return {'evaluation': 'pass', 'error_log': [f'Evaluator вернул: {eval_res}']}
