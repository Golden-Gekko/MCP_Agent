from .state import AgentState


class ContextCompressorNode:
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
                tool_msg = tool_index.get(tc.get('id'))
                if tool_msg is not None:
                    result = self._extract_text(tool_msg.content)
                    lines.append(f'[ИНСТРУМЕНТ] "{tc.get("name", "unknown")}". Результат: {result}')
        return '\n\n'.join(lines) if lines else '(история выполнения пуста)'

    def node(self, state: AgentState) -> dict:
        steps_summary = self._build_steps_summary(state['messages'])
        step = state['current_step']
        history = state.get('history', []) + [
            f"[РЕЗУЛЬТАТ ВЫПОЛНЕНИЯ ШАГА {step + 1}] {state['plan'][step]}.\n\n{steps_summary}"]
        return {'history': history}
