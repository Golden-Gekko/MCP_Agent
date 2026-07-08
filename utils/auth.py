import secrets
from pathlib import Path

from core import settings


def get_gradio_auth():
    if not settings.service.auth_enabled:
        return None

    path = Path(settings.service.auth_users_file)
    if not path.is_file():
        raise RuntimeError(f'Файл учётных записей не найден: {path}')

    credentials = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#') or ':' not in line:
            continue

        user, psw = line.split(':', 1)
        if (user := user.strip()) and (psw := psw.strip()):
            credentials[user] = psw

    if not credentials:
        raise RuntimeError(f'Файл учётных записей пуст или некорректен: {path}')

    def checker(username: str, password: str) -> bool:
        stored = credentials.get(username)
        return stored is not None and secrets.compare_digest(stored, password)

    return checker
