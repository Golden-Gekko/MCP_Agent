from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langfuse.langchain import CallbackHandler

from core import settings
from tools import init_tools


class MCPAgent:
    def __init__(self):
        self.llm: ChatOpenAI = settings.llm.chat.llm
        self.llm_with_tools = None
        self.tools: list[BaseTool] | None = None
        self.graph = None
        self.lf_handler = CallbackHandler(public_key=settings.langfuse.public_key)

    async def init_graph(self):
        self.tools = await init_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools, parallel_tool_calls=False)
        self.graph = self._compile_graph()

    def agent_node(self, state: MessagesState) -> dict[str, Any]:
        prompt = settings.langfuse.client.get_prompt(name='mcp_agent_prompt').compile()
        prompt_with_history = ChatPromptTemplate.from_messages([
            ('system', prompt),
            MessagesPlaceholder(variable_name='messages'),
        ])
        chain = prompt_with_history | self.llm_with_tools
        response = chain.invoke({'messages': state['messages']})
        return {'messages': [response]}

    def _compile_graph(self):
        workflow = StateGraph(MessagesState)
        workflow.add_node('agent', self.agent_node)
        workflow.add_node('tools', ToolNode(tools=self.tools))

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
