from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langfuse.langchain import CallbackHandler
from loguru import logger

from core import settings
from tools import init_tools

from .state import AgentState


class MCPAgent:
    def __init__(self):
        self.llm: ChatOpenAI = settings.llm.chat.llm
        self.llm_with_tools = None
        self.tools: list[BaseTool] | None = None
        self.tool_node: ToolNode | None = None
        self.graph = None
        self.lf_handler = CallbackHandler(public_key=settings.langfuse.public_key)

    async def init_graph(self):
        try:
            self.tools = await init_tools()
        except Exception as e:
            logger.error(f'Ошибка инициализации инструментов: {e}')
            raise
        self.llm_with_tools = self.llm.bind_tools(self.tools, parallel_tool_calls=False)
        self.tool_node = ToolNode(tools=self.tools)
        self.graph = self._compile_graph()

    def agent_node(self, state: AgentState) -> dict[str, Any]:
        prompt = settings.langfuse.client.get_prompt(name='mcp_agent_prompt').compile()
        step_text = state['plan'][state['current_step']]
        step_instruction = HumanMessage(
            content=f"Текущий шаг: {step_text}\n"
                    f"Контекст предыдущих шагов: {list(state['step_results'].values())[-2:]}"
        )
        messages = [SystemMessage(content=prompt), step_instruction]
        response = self.llm_with_tools.ainvoke(messages)
        return {'messages': [response]}

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
        if ev == 'pass':
            return 'increment_step'
        elif ev == 'retry' and retries < 2:
            return 'retry_step'
        return 'finalize'

    @staticmethod
    def increment_step(state: AgentState) -> dict[str, Any]:
        idx = state['current_step'] + 1
        return {
            'current_step': idx,
            'retry_count': 0,
            'messages': []}

    @staticmethod
    def retry_step(state: AgentState) -> dict[str, Any]:
        return {
            'retry_count': state.get('retry_count', 0) + 1,
            'messages': []}

    def _compile_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node('agent', self.agent_node)
        workflow.add_node('tools', self.tool_node)

        workflow.set_entry_point('agent')
        workflow.add_conditional_edges(
            'agent', tools_condition,
            {'tools': 'tools', '__end__': '__end__'})
        workflow.add_edge('tools', 'agent')

        graph = workflow.compile()
        return graph

    async def run(self, input_messages: dict[str, Any]) -> list[BaseMessage]:
        if self.graph is None:
            raise RuntimeError(
                'Агент не инициализирован. Запустите `initialize()`.')
        return (
            await self.graph.ainvoke(input_messages, config={'callbacks': [self.lf_handler]})
        )['messages']
