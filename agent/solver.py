from loguru import logger

from .state import AgentState


class BaseSolver:
    @staticmethod
    async def node(state: AgentState) -> dict:
        logger.debug(f"User message: {state['user_input'].lower()}")
        if 'продолжить' in state['user_input'].lower():
            return {'is_approved': True, 'user_input': ''}
        return {'is_approved': False}


class PlanSolverNode(BaseSolver):
    pass


class AgentSolverNode(BaseSolver):
    pass
