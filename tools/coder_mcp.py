from mcp.server.fastmcp import FastMCP

from core import settings

mcp = FastMCP('coder')


@mcp.tool()
def generate_code(task: str, code_context: str) -> str:
    """Генерирует код по заданию пользователя.
    Args:
        task: Краткое описание задачи (например, "напиши функцию factorial с кэшированием")
        code_context: Содержимое файлов, которые нужно изменить или которые влияют на задачу. ВСЕГДА заполняй этот параметр, если задача касается существующего кода. Если код пишется с нуля и файлов нет, напиши "Нет контекста".
    Returns:
        Сгенерированный код
    """
    prompt = (
        settings.langfuse.client
        .get_prompt(name='mcp_agent_coder_prompt')
        .compile(code_context=code_context.strip(), task=task)
    )
    response = settings.llm.coder.llm.invoke(prompt)

    content = response.content
    return content.strip()


if __name__ == '__main__':
    mcp.run(transport='stdio')
