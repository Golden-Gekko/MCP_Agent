from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from .state import AgentState


class StepInjectorNode:
    def node(self, state: AgentState) -> dict:
        step_text = state['plan'][state['current_step']]
        messages: list[BaseMessage] = [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            HumanMessage(f'Запрос пользователя: {state["user_request"]}')
        ]
        for item in state.get('history', []):
            messages.append(AIMessage(content=item))
        messages.append(HumanMessage(f'ТЕКУЩИЙ ШАГ, КОТОРЫЙ НУЖНО ВЫПОЛНИТЬ: {step_text}'))
        return {'messages': messages,}
