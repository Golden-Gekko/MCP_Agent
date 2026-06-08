from typing import Annotated, Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class PlanItem(BaseModel):
    description: Annotated[
        str, Field(description='Подробное описание шага, который нужно сделать')]
    done: Annotated[bool, Field(default=False, description='Статус выполнения')]


class Workflow(BaseModel):
    plan: Annotated[list[PlanItem], Field(description='Список шагов для выполнения задачи')]


class AgentState(MessagesState):
    user_request: str
    user_input: str
    plan: list[dict]
    current_step: int
    step_iteration: int
    history: list[str]
    phase: Literal['planning', 'executing', 'done']
    is_approved: bool
    trace_id: str
