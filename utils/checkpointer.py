from loguru import logger
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row


async def create_checkpointer(db_url: str) -> AsyncPostgresSaver:
    try:
        conn = await AsyncConnection.connect(
            db_url,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        )
        checkpointer = AsyncPostgresSaver(conn=conn)
        await checkpointer.setup()
        logger.success('Postgres checkpointer инициализирован')
        return checkpointer
    except Exception as e:
        raise RuntimeError(f'Не удалось подключиться к Postgres для checkpoint ({db_url}): {e}')


async def close_checkpointer(checkpointer: AsyncPostgresSaver) -> None:
    await checkpointer.conn.close()
