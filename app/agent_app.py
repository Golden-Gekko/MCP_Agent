import asyncio
import uuid

import gradio as gr
from loguru import logger

from agent import MCPAgent
from core import settings
from utils.auth import get_gradio_auth


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
            gr.Markdown('Помощник разработчика с доступом к файлам, Git и документации')

            chatbot = gr.Chatbot(label='Чат с агентом', height=700)
            request_id_state = gr.State('')

            with gr.Row():
                msg = gr.Textbox(
                    label='Ваше сообщение',
                    placeholder='Введите сообщение или нажмите "Продолжить"',
                    scale=8,
                    container=False
                )
                submit_btn = gr.Button('Отправить', variant='primary')
                continue_btn = gr.Button('▶ Продолжить', variant='secondary')

            gr.Examples(
                examples=[
                    'Напиши функцию вычисления факториала на Python',
                    'Прочитай и выведи содержимое файла test.py',
                    'Напиши функцию ряда Фибоначчи, сохрани в файл fibc.py',
                ],
                inputs=msg,
                label='Примеры запросов'
            )

            submit_btn.click(
                fn=self._respond,
                inputs=[msg, chatbot, request_id_state],
                outputs=[chatbot, msg, request_id_state]
            )
            msg.submit(
                fn=self._respond,
                inputs=[msg, chatbot, request_id_state],
                outputs=[chatbot, msg, request_id_state]
            )
            continue_btn.click(
                fn=self._continue,
                inputs=[chatbot, request_id_state],
                outputs=[chatbot, request_id_state]
            )

    async def _respond(self, message: str, history: list, request_id: str, request: gr.Request):
        if not request_id:
            request_id = str(uuid.uuid4())

        if not message or not message.strip():
            return history, '', request_id
        phase = await self.agent.get_phase(request_id, username=request.username)

        if phase is None or phase == 'done':
            result = await self.agent.run(
                user_messages=message, request_id=request_id, username=request.username)
        else:
            result = await self.agent.resume(
                user_messages=message, request_id=request_id, username=request.username)

        history.append({'role': 'user', 'content': message})
        agent_response = result[-1].content if result else 'Нет ответа'
        history.append({'role': 'assistant', 'content': agent_response})
        return history, '', request_id

    async def _continue(self, history: list, request_id: str, request: gr.Request):
        phase = await self.agent.get_phase(request_id, username=request.username)

        if phase is None or phase == 'done':
            history.append({
                'role': 'assistant',
                'content': 'Нет активных задач для продолжения. Задайте новый вопрос.'
            })
            return history, request_id

        result = await self.agent.resume(
            user_messages='Продолжить', request_id=request_id, username=request.username)

        history.append({'role': 'user', 'content': '▶ Продолжить'})
        agent_response = result[-1].content if result else 'Нет ответа'
        history.append({'role': 'assistant', 'content': agent_response})
        return history, request_id

    async def initialize_and_launch(self):
        await self.init_agent()
        self.build_interface()
        self.demo.launch(
            server_name=settings.service.host,
            server_port=settings.service.port,
            show_error=True,
            auth=get_gradio_auth(),
            auth_message='<p>Введите выданный Вам логин и пароль</p>'
        )

    def run(self):
        asyncio.run(self.initialize_and_launch())
