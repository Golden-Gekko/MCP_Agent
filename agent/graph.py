from datetime import datetime
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, RemoveMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt import ToolNode
from langfuse.langchain import CallbackHandler
from loguru import logger

from core import settings
from tools import init_tools

from .agent_node import AgentNode
from .compressor import ContextCompressorNode
from .evaluator import EvaluatorNode
from .finalizer import FinalizerNode
from .planer import PlanerNode
from .state import AgentState


class MCPAgent:
    def __init__(self):
        self.llm_with_tools: ChatOpenAI | None = None
        self.tools: list[BaseTool] | None = None
        self.graph = None
        self.checkpointer = InMemorySaver()
        self.lf_handler = CallbackHandler(public_key=settings.langfuse.public_key)
        self.init_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    async def init_graph(self):
        try:
            self.tools = await init_tools()
        except Exception as e:
            logger.error(f'Ошибка инициализации инструментов: {e}')
            raise
        self.llm_with_tools = settings.llm.chat.llm.bind_tools(self.tools, parallel_tool_calls=False)
        self.graph = self._compile_graph()

    @staticmethod
    def need_adjust_plan(state: AgentState) -> Literal['agent_node', 'planer']:
        logger.info(state['user_request'])
        logger.info(state['user_request'].lower())
        logger.info('продолжить' in state['user_request'].lower())

        if 'продолжить' in state['user_request'].lower():
            logger.info('THIS IS IFFFF')

            return 'agent_node'
        logger.info('NOT IF')
        return 'planer'

    @staticmethod
    def step_router(state: AgentState) -> str:
        if len(state['messages']) == 0:
            return 'error'
        last_msg = state['messages'][-1]
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            return 'tools'
        return 'compress'

    @staticmethod
    def eval_router(state: AgentState) -> str:
        ev = state.get('evaluation', 'pass')
        retries = state.get('retry_count', 0)
        if ev == 'retry' and retries < settings.service.max_retries:
            return 'retry_step'
        idx = state['current_step'] + 1
        if idx < len(state['plan']):
            return 'increment_step'
        return 'finalize'

    @staticmethod
    def increment_step(state: AgentState) -> dict[str, Any]:
        state['history'] = state.get('history', []) + [state.get('step_content', '')]
        messages = [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *[AIMessage(content=h) for h in state['history']]
        ]
        return {
            'history': state['history'],
            'messages': messages,
            'current_step': state['current_step'] + 1,
            'retry_count': 0
        }

    @staticmethod
    def retry_step(state: AgentState) -> dict[str, Any]:
        return {'retry_count': state.get('retry_count', 0) + 1}

    def _compile_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node('agent_node', AgentNode(llm=self.llm_with_tools).node)
        workflow.add_node('compressor', ContextCompressorNode().node)
        workflow.add_node('evaluator', EvaluatorNode(llm=settings.llm.chat.llm).node)
        workflow.add_node('finalizer', FinalizerNode(llm=settings.llm.chat.llm).node)
        workflow.add_node('planer', PlanerNode(llm=settings.llm.chat.llm).node)

        workflow.add_node('increment_step', self.increment_step)
        workflow.add_node('retry_step', self.retry_step)
        workflow.add_node('tools', ToolNode(self.tools))

        workflow.set_entry_point('planer')
        workflow.add_conditional_edges(source='planer', path=self.need_adjust_plan)

        workflow.set_finish_point('agent_node')
        # workflow.add_edge('planer', 'agent_node')
        # workflow.add_conditional_edges(
        #     'agent_node', self.step_router,
        #     {
        #         'tools': 'tools',
        #         'compress': 'compressor',
        #         'error': 'finalizer',
        #     })
        # workflow.add_edge('tools', 'agent_node')
        # workflow.add_edge('compressor', 'evaluator')
        # workflow.add_conditional_edges(
        #     'evaluator', self.eval_router,
        #     {
        #         'increment_step': 'increment_step',
        #         'retry_step': 'retry_step',
        #         'finalize': 'finalizer'
        #     })
        # workflow.add_edge('increment_step', 'agent_node')
        # workflow.add_edge('retry_step', 'agent_node')
        # workflow.set_finish_point('finalizer')
        graph = workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_after=['planer']
        )
        return graph

    async def run(
            self,
            input_messages: dict[str, Any],
            config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if self.graph is None:
            raise RuntimeError(
                'Агент не инициализирован. Запустите `initialize()`.')
        (config or {}).update({
            'callbacks': [self.lf_handler],
            'metadata': {'langfuse_session_id': f'docker_session_{self.init_time}'}
        })
        result = await self.graph.ainvoke(input_messages, config=config)
        return {
            'messages': result['messages'],
            'phase': result['phase']
        }
