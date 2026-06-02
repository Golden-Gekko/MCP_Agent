from langchain_core.messages import SystemMessage

from core import settings

from .state import AgentState


def evaluator_node(state: AgentState) -> dict:
    last_msg = state['messages'][-1]
    result = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
    prompt = (
        settings.langfuse.client.get_prompt(name='mcp_agent_evaluator_prompt')
        .compile(
            step_text=state['plan'][state['current_step']],
            step_result=result
        ))
    response = settings.llm.chat.llm.invoke([SystemMessage(content=prompt)])
    eval_res = response.content.strip().lower()

    if eval_res in ['pass', 'retry', 'fail']:
        step_results = state.get('step_results', {}).copy()
        step_results[state['current_step']] = result[:1000]
        return {'step_results': step_results, 'evaluation': eval_res}

    return {'evaluation': 'pass', 'error_log': [f'Evaluator вернул: {eval_res}']}
