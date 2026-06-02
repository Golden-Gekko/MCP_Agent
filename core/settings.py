from functools import cached_property, lru_cache
from pathlib import Path
from typing import Annotated

from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from .logger import configure_logging


class LangfuseConfig(BaseModel):
    secret_key: str
    public_key: str
    base_url: str

    @cached_property
    def client(self) -> Langfuse:
        return Langfuse(
            public_key=self.public_key,
            secret_key=self.secret_key,
            base_url=self.base_url,
        )


class ServiceConfig(BaseModel):
    host: str
    port: int
    workspace: str
    loglevel: str
    log_to_file: bool
    context7_api_key: str
    max_retries: int = 2


class LLMConfig(BaseModel):
    host: str
    port: Annotated[int | None, Field(default=None, ge=1, le=65535)]
    model_name: str
    api_key: SecretStr
    temperature: Annotated[float, Field(default=0.0, ge=0.0, le=1.1)]

    @cached_property
    def llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            base_url=self.url,
            api_key=self.api_key,
            model=self.model_name,
            temperature=self.temperature,
        )

    @property
    def url(self) -> str:
        if self.port is not None:
            return f'http://{self.host}:{self.port}/v1'
        return f'https://{self.host}/v1'


class LLMsTypes(BaseModel):
    chat: LLMConfig
    coder: LLMConfig


class Settings(BaseSettings):
    service: ServiceConfig = Field(validation_alias='AGENT')
    llm: LLMsTypes
    langfuse: LangfuseConfig

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / '.env',
        env_file_encoding='utf-8',
        extra='ignore',
        case_sensitive=False,
        env_nested_delimiter='__',
    )

    def model_post_init(self, __context) -> None:
        _ = self.langfuse.client
        _ = self.llm.chat.llm
        _ = self.llm.coder.llm


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()
    except Exception as e:
        print(f'Ошибка создания Settings: {e}')
        raise


settings = get_settings()
configure_logging(level=settings.service.loglevel, create_file=settings.service.log_to_file)
