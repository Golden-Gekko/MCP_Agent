from typing import Literal

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    user_request: str
    plan: list[str]
    current_step: int
    evaluation: Literal['pass', 'retry', 'fail']
    retry_count: int
    error_log: list[str]
