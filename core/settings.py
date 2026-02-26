from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .logger import configure_logging


class ServiceConfig(BaseModel):
    host: str
    port: int
    workspace: str
    loglevel: str
    log_to_file: bool
    context7_api_key: str


class LLMConfig(BaseModel):
    host: str
    port: Annotated[int | None, Field(default=None)]
    model_name: str
    api_key: str

    @property
    def url(self) -> str:
        if self.port is not None:
            return f'http://{self.host}:{self.port}/v1'
        return f'https://{self.host}/v1'


class LLMsTypes(BaseModel):
    orchestrator: LLMConfig
    coder: LLMConfig


class Settings(BaseSettings):
    service: ServiceConfig
    llm: LLMsTypes

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / '.env',
        env_file_encoding='utf-8',
        extra='ignore',
        case_sensitive=False,
        env_nested_delimiter='__',
    )


settings = Settings()
configure_logging(level=settings.service.loglevel, create_file=settings.service.log_to_file)
