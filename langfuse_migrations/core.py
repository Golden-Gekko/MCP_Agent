from functools import lru_cache, cached_property
from pathlib import Path

from langfuse import Langfuse
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


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


class Settings(BaseSettings):
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


@lru_cache
def get_settings() -> Settings | None:
    return Settings()


settings = get_settings()
