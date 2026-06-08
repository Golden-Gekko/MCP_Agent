from langchain_core.messages import HumanMessage

from .state import AgentState


class StepInjectorNode:
    def node(self, state: AgentState) -> dict:
        step_text = state['plan'][state['current_step']]['description']
        return {
            'messages': [HumanMessage(f'ТЕКУЩИЙ ШАГ, КОТОРЫЙ НУЖНО ВЫПОЛНИТЬ: {step_text}')],
        }
