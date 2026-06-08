import asyncio

import gradio as gr
from langchain_core.messages import HumanMessage
from loguru import logger

from agent import MCPAgent
from core import settings


class MCPCodingAgentApp:
    def __init__(self):
        self.demo = None
        self.agent = None

    async def init_agent(self):
        self.agent = MCPAgent()
        await self.agent.init_graph()
        logger.success('MCP агент инициализирован')

    def build_interface(self):
        with gr.Blocks(title='MCP Coding Agent', fill_height=True) as self.demo:
            gr.Markdown('# MCP Coding Agent')
            gr.Markdown(
                'Помощник разработчика с доступом к файлам, Git и документации')
            chatbot = gr.ChatInterface(
                fn=self.respond,
                title='Чат с агентом',
                examples=[
                    'Напиши функцию вычисления факториала на Python',
                    'прочитай и выведи содержимое файла test.py',
                    'Напиши функцию факториала, сохрани в файл fact.py',
                ],
            )

    async def respond(self, message: str, _: list[dict[str, str]], request: gr.Request) -> str:
        if self.agent is None:
            logger.error('Агент не инициализирован')
            return 'Агент не инициализирован'

        request_id = request.session_hash
        phase = self.agent.get_phase(request_id)
        if phase is None or phase == 'done':
            result = await self.agent.run(user_messages=message, request_id=request_id)
        else:
            result = await self.agent.resume(user_messages=message, request_id=request_id)

        return result[-1].content

    async def initialize_and_launch(self):
        await self.init_agent()
        self.build_interface()
        self.demo.launch(
            server_name=settings.service.host,
            server_port=settings.service.port,
            show_error=True,
        )

    def run(self):
        asyncio.run(self.initialize_and_launch())
