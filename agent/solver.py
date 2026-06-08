from langchain_core.messages import HumanMessage
from loguru import logger

from .state import AgentState


class PlanSolverNode:
    @staticmethod
    async def node(state: AgentState) -> dict:
        logger.debug(f"User message: {state['user_request'].lower()}")
        if 'продолжить' in state['user_request'].lower():
            return {'is_approved': True, 'user_request': ''}
        return {'is_approved': False}


class AgentSolverNode:
    @staticmethod
    async def node(state: AgentState) -> dict:
        logger.debug(f"User message: {state['user_request'].lower()}")
        if 'продолжить' in state['user_request'].lower():
            return {'is_approved': True, 'user_request': ''}
        return {
            'is_approved': False,
            'messages': [HumanMessage(state['user_request'])]
        }

