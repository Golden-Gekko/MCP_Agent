import sys

from loguru import logger


def configure_logging(level: str = 'INFO', create_file: bool = False):
    logger.remove()

    logger.add(
        sys.stdout,
        format=(
            '<green>{time:HH:mm:ss}</green> | '
            '<level>{level: <8}</level> | '
            '<cyan>{module}:{line}</cyan> - '
            '{message}'),
        level=level,
        colorize=True
    )

    if create_file:
        logger.add(
            'app.log',
            format='{time} | {level} | {module}:{line} - {message}',
            level=level,
            rotation='10 MB',
            retention='7 days',
            encoding='utf-8',
        )
