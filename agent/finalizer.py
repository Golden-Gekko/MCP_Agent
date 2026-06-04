from langchain.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from loguru import logger

from utils.langfuse import load_prompt_from_langfuse

from .state import AgentState


class FinalizerNode:
    def __init__(
            self,
            llm: BaseChatModel,
            prompt_name: str = 'mcp_agent_finalize_prompt',
            prompt_label: str = 'production'
    ):
        self.llm = llm

        self.prompt = load_prompt_from_langfuse(prompt_name=prompt_name, prompt_label=prompt_label)
        logger.debug(f'FinalizerNode prompt: {self.prompt}')

    @staticmethod
    def _extract_text(content) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'text':
                    parts.append(block.get('text', ''))
                elif isinstance(block, str):
                    parts.append(block)
            return '\n'.join(p for p in parts if p).strip()
        return str(content).strip()

    def _build_steps_summary(self, messages: list) -> str:
        tool_index = {
            m.tool_call_id: m
            for m in messages
            if getattr(m, 'type', None) == 'tool' and getattr(m, 'tool_call_id', None)
        }

        lines = []
        for msg in messages:
            if getattr(msg, 'type', None) != 'ai':
                continue

            text = self._extract_text(msg.content)
            if text:
                lines.append(f'[ОТВЕТ АГЕНТА]: {text}')
                continue

            tool_calls = getattr(msg, 'tool_calls', None) or []
            for tc in tool_calls:
                tc_id = tc.get('id')
                tool_name = tc.get('name', 'unknown')

                tool_msg = tool_index.get(tc_id)
                if tool_msg is not None:
                    result = self._extract_text(tool_msg.content)
                    if len(result) > 1500:
                        result = result[:1500] + '... [данные обрезаны]'
                    lines.append(f'[ИНСТРУМЕНТ] "{tool_name}". Результат: {result}')
        return '\n\n'.join(lines) if lines else '(история выполнения пуста)'

    async def node(self, state: AgentState) -> dict:
        plan = '\n'.join(f"{i}. {s}" for i, s in enumerate(state.get('plan', []), start=1))
        steps_summary = self._build_steps_summary(state['messages'])

        msg = (
            f'Запрос пользователя: {state["user_request"]}\n\n'
            f'План: {plan}\n\n'
            f'Результаты выполнения: {steps_summary}\n\n'
            f'Сформируй итоговый ответ для пользователя.')

        logger.info(f'FinalizerNode message: {msg[:500]}')

        response = await self.llm.ainvoke([
            SystemMessage(content=self.prompt),
            HumanMessage(content=msg),
        ])
        return {'messages': [AIMessage(content=response.content.strip())]}
