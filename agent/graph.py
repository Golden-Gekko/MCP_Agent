from typing import Any, Literal

from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler
from langfuse.types import TraceContext
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger

from core import settings
from tools import init_tools
from utils.checkpointer import close_checkpointer, create_checkpointer

from .agent_node import AgentNode
from .compressor import ContextCompressorNode
from .finalizer import FinalizerNode
from .planer import PlanerNode
from .solver import AgentSolverNode, PlanSolverNode
from .state import AgentState
from .step_injector import StepInjectorNode


class MCPAgent:
    def __init__(self):
        self.llm_with_tools: ChatOpenAI | None = None
        self.tools: list[BaseTool] | None = None
        self.graph = None
        self.checkpointer: AsyncPostgresSaver | None = None

    async def init_graph(self):
        try:
            self.tools = await init_tools()
        except Exception as e:
            logger.error(f'Ошибка инициализации инструментов: {e}')
            raise
        self.checkpointer = await create_checkpointer(settings.service.checkpoint_db_url)
        self.llm_with_tools = settings.llm.chat.llm.bind_tools(self.tools, parallel_tool_calls=False)
        self.graph = self._compile_graph()

    async def close(self):
        if self.checkpointer is not None:
            await close_checkpointer(self.checkpointer)
            self.checkpointer = None

    @staticmethod
    def resolve_thread_id(username: str | None, request_id: str) -> str:
        return f'{username or "anonymous"}:{request_id}'

    @staticmethod
    def need_adjust_plan_router(state: AgentState) -> Literal['injector', 'planer']:
        if state.get('is_approved', False):
            return 'injector'
        return 'planer'

    @staticmethod
    def need_modify_step_router(state: AgentState) -> Literal['agent_node', 'compressor']:
        if state.get('is_approved', False):
            return 'compressor'
        return 'agent_node'

    @staticmethod
    def agent_router(state: AgentState) -> Literal['agent_solver', 'tools']:
        last_msg = state['messages'][-1]
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            return 'tools'
        return 'agent_solver'

    @staticmethod
    def next_step_router(state: AgentState) -> Literal['injector', 'finalizer']:
        if state['current_step'] < len(state['plan']):
            return 'injector'
        return 'finalizer'

    def _compile_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node('agent_node', AgentNode(llm=self.llm_with_tools).node)
        workflow.add_node('compressor', ContextCompressorNode().node)
        workflow.add_node('finalizer', FinalizerNode(llm=settings.llm.chat.llm).node)
        workflow.add_node('planer', PlanerNode(llm=settings.llm.chat.llm).node)
        workflow.add_node('plan_solver', PlanSolverNode().node)
        workflow.add_node('agent_solver', AgentSolverNode().node)
        workflow.add_node('injector', StepInjectorNode().node)
        workflow.add_node('tools', ToolNode(self.tools))

        workflow.set_entry_point(key='planer')
        workflow.add_edge(start_key='planer', end_key='plan_solver')
        workflow.add_conditional_edges(source='plan_solver', path=self.need_adjust_plan_router)
        workflow.add_edge(start_key='injector', end_key='agent_node')
        workflow.add_conditional_edges(source='agent_node', path=self.agent_router)
        workflow.add_edge(start_key='tools', end_key='agent_node')
        workflow.add_conditional_edges(source='agent_solver', path=self.need_modify_step_router)
        workflow.add_conditional_edges(source='compressor', path=self.next_step_router)
        workflow.set_finish_point(key='finalizer')
        graph = workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_after=['planer'],
            interrupt_before=['agent_solver'],
        )
        return graph

    async def run(self, user_messages: str, request_id: str, username: str | None = None) -> list:
        self._check_graph_available()
        trace_id = Langfuse.create_trace_id()
        logger.debug(f'Трейс при старте графа: {trace_id}')
        return await self._ainvoke_with_tracing(
            data={'user_request': user_messages, 'user_input': user_messages, 'trace_id': trace_id},
            request_id=request_id,
            trace_id=trace_id,
            span_name='agent_run',
            username=username,
            trace_input=user_messages,
        )

    async def resume(self, user_messages: str, request_id: str, username: str | None = None) -> list:
        self._check_graph_available()
        trace_id = await self._get_trace_id(request_id, username)
        logger.debug(f'Трейс при возвращении в граф: {trace_id}')
        return await self._ainvoke_with_tracing(
            data=Command(update={'user_input': user_messages}),
            request_id=request_id,
            trace_id=trace_id,
            span_name='agent_resume',
            username=username,
            trace_input=user_messages,
        )

    async def get_phase(self, request_id: str, username: str | None = None):
        self._check_graph_available()

        state = await self.graph.aget_state(
            {'configurable': {'thread_id': self.resolve_thread_id(username, request_id)}})
        return state.values.get('phase', None)

    def _check_graph_available(self):
        if self.graph is None:
            raise RuntimeError(
                'Агент не инициализирован. Запустите `initialize()`.')

    def _create_config(self, request_id: str, username: str | None, trace_id: str) -> dict[str, Any]:
        lf_handler = CallbackHandler(
            public_key=settings.langfuse.public_key,
            trace_context=TraceContext(trace_id=trace_id)
        )
        return {
            'callbacks': [lf_handler],
            'metadata': {
                'langfuse_session_id': f'{username or "anonymous"}_{request_id[:8]}',
            },
            'configurable': {'thread_id': self.resolve_thread_id(username, request_id)},
        }

    async def _get_trace_id(self, request_id: str, username: str | None = None) -> str | None:
        snapshot = await self.graph.aget_state(
            {'configurable': {'thread_id': self.resolve_thread_id(username, request_id)}}
        )
        return snapshot.values.get('trace_id', None)

    async def _ainvoke_with_tracing(
            self,
            data: dict[str, Any] | Command,
            request_id: str,
            trace_id: str,
            span_name: str,
            username: str | None = None,
            trace_input: str = '',
    ) -> list:
        user = username or 'anonymous'
        session_id = f'{user}_{request_id[:8]}'

        with settings.langfuse.client.start_as_current_observation(
                as_type='span',
                name=span_name,
                trace_context={'trace_id': trace_id},
        ) as span:
            with propagate_attributes(
                user_id=user,
                session_id=session_id,
                trace_name='MCP Agent',
                tags=['langgraph', 'mcp-agent'],
                metadata={'request_id': request_id, 'username': user},
            ):
                span.update(input={'message': trace_input})
                result = await self.graph.ainvoke(
                    data,
                    config=self._create_config(request_id, username, trace_id),
                )
                span.update(output=str(result['messages'][-1].content))
                return result['messages']
