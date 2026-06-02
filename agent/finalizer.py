from typing import Any

from langchain_core.messages import AIMessage, SystemMessage

from core import settings

from .state import AgentState


def finalizer_node(state: AgentState) -> dict[str, Any]:
    plan = state.get('plan', [])
    results = state.get('step_results', {})

    steps_summary = []
    for i, res in sorted(results.items()):
        step = f"* Шаг {i + 1}: {plan[i] if i < len(plan) else 'Неизвестно'}.\n"
        steps_summary.append(f'{step}{res}')
    prompt = (
        settings.langfuse.client.get_prompt(name='mcp_agent_finalize_prompt')
        .compile(
            user_request=state.get(key='user_request', default='Без запроса'),
            steps_summary='\n\n'.join(steps_summary)
        ))
    response = settings.llm.chat.llm.invoke([SystemMessage(content=prompt)])
    return {'messages': [AIMessage(content=response.content.strip())]}
