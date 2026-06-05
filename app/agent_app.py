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

        config = {'configurable': {'thread_id': request.session_hash}}
        result = await self.agent.run({'user_request': message}, config=config,)

        if result.get('phase', 'done') == 'done':
            return result['messages'][-1].content

        if result['phase'] == 'planning':
            message: str = result['messages'][-1].content
            if 'шаг' not in message.lower():
                return 'Произошла ошибка планирования. Повторите запрос.'
            return message

        if result['phase'] == 'executing':
            message: str = result['messages'][-1].content
            return message

        return 'Неизвесная ошибка. Повторите запрос'

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
