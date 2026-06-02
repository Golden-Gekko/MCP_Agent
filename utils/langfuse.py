from loguru import logger

from core import settings


def load_prompt_from_langfuse(prompt_name: str, prompt_label: str = 'production') -> str:
    try:
        return (
            settings.langfuse.client
            .get_prompt(prompt_name, label=prompt_label)
            .get_langchain_prompt())

    except Exception as e:
        logger.error(f'Не удалось загрузить промпт "{prompt_name}" из Langfuse: {e}')
        raise RuntimeError(f'Не удалось загрузить промпт "{prompt_name}" из Langfuse')
