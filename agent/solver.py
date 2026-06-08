from langchain_core.messages import HumanMessage
from loguru import logger

from .state import AgentState


class PlanSolverNode:
    @staticmethod
    async def node(state: AgentState) -> dict:
        logger.debug(f"User message: {state['user_input'].lower()}")
        if 'продолжить' in state['user_input'].lower():
            return {'is_approved': True, 'user_input': ''}
        return {'is_approved': False}


class AgentSolverNode:
    @staticmethod
    async def node(state: AgentState) -> dict:
        logger.debug(f"User message: {state['user_input'].lower()}")
        if 'продолжить' in state['user_input'].lower():
            return {'is_approved': True, 'user_input': ''}
        return {
            'is_approved': False,
            'messages': [HumanMessage(state['user_input'])]
        }

