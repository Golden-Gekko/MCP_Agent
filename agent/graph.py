from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langfuse.langchain import CallbackHandler
from loguru import logger

from core import settings
from tools import init_tools

from .agent_node import AgentNode
from .evaluator import EvaluatorNode
from .finalizer import FinalizerNode
from .planer import PlanerNode
from .state import AgentState


class MCPAgent:
    def __init__(self):
        self.llm_with_tools: ChatOpenAI | None = None
        self.tools: list[BaseTool] | None = None
        self.graph = None
        self.lf_handler = CallbackHandler(public_key=settings.langfuse.public_key)

    async def init_graph(self):
        try:
            self.tools = await init_tools()
        except Exception as e:
            logger.error(f'Ошибка инициализации инструментов: {e}')
            raise
        self.llm_with_tools = settings.llm.chat.llm.bind_tools(self.tools, parallel_tool_calls=False)
        self.graph = self._compile_graph()

    @staticmethod
    def step_router(state: AgentState) -> str:
        last_msg = state['messages'][-1]
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            return 'tools'
        return 'evaluate'

    @staticmethod
    def eval_router(state: AgentState) -> str:
        ev = state.get('evaluation', 'pass')
        retries = state.get('retry_count', 0)
        if ev == 'retry' and retries < settings.service.max_retries:
            return 'retry_step'
        idx = state['current_step'] + 1
        if ev == 'pass' and idx < len(state['plan']):
            return 'increment_step'
        return 'finalize'

    @staticmethod
    def increment_step(state: AgentState) -> dict[str, Any]:
        return {
            'current_step': state['current_step'] + 1,
            'retry_count': 0}

    @staticmethod
    def retry_step(state: AgentState) -> dict[str, Any]:
        return {'retry_count': state.get('retry_count', 0) + 1}

    def _compile_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node('agent_node', AgentNode(llm=self.llm_with_tools).node)
        workflow.add_node('evaluator', EvaluatorNode(llm=settings.llm.chat.llm).node)
        workflow.add_node('finalizer', FinalizerNode(llm=settings.llm.chat.llm).node)
        workflow.add_node('planer', PlanerNode(llm=settings.llm.chat.llm).node)

        workflow.add_node('increment_step', self.increment_step)
        workflow.add_node('retry_step', self.retry_step)
        workflow.add_node('tools', ToolNode(self.tools))

        workflow.set_entry_point('planer')
        workflow.add_edge('planer', 'agent_node')
        workflow.add_conditional_edges(
            'agent_node', self.step_router,
            {'tools': 'tools', 'evaluate': 'evaluator'})
        workflow.add_edge('tools', 'agent_node')
        workflow.add_conditional_edges(
            'evaluator', self.eval_router,
            {
                'increment_step': 'increment_step',
                'retry_step': 'retry_step',
                'finalize': 'finalizer'
            })
        workflow.add_edge('increment_step', 'agent_node')
        workflow.add_edge('retry_step', 'agent_node')
        workflow.set_finish_point('finalizer')
        graph = workflow.compile()
        return graph

    async def run(self, input_messages: dict[str, Any]) -> list[BaseMessage]:
        if self.graph is None:
            raise RuntimeError(
                'Агент не инициализирован. Запустите `initialize()`.')
        return (
            await self.graph.ainvoke(input_messages, config={'callbacks': [self.lf_handler]})
        )['messages']
