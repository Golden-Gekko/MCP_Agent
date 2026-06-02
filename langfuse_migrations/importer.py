import json
from pathlib import Path

from langfuse.api import (
    ChatMessage,
    CreateChatPromptRequest,
    CreateChatPromptType,
    CreateTextPromptRequest,
    CreateTextPromptType,
    PlaceholderMessage
)

from core import settings


def import_prompts(input_file_name: str):
    if not Path(input_file_name).exists():
        print(f'Файл не найден: {input_file_name}')
        return

    try:
        langfuse = settings.langfuse.client
    except Exception as e:
        print(f'Ошибка подключения к Langfuse: {e}')
        return

    with open(input_file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    prompts_to_import = data.get('prompts', [])
    if not prompts_to_import:
        print('Нет промптов для импорта')
        return

    print(f'Найдено {len(prompts_to_import)} промптов для импорта')

    migrated = 0
    failed = 0
    for prompt_to_import in prompts_to_import:
        try:
            if isinstance(prompt_to_import['prompt'], str):
                request = CreateTextPromptRequest(
                    name=prompt_to_import['name'],
                    type=CreateTextPromptType.TEXT,
                    prompt=prompt_to_import['prompt'],
                    labels=prompt_to_import.get('labels', []),
                    tags=prompt_to_import.get('tags', []),
                    config=prompt_to_import.get('config', {}),
                )
            elif isinstance(prompt_to_import['prompt'], list):
                prompts = []
                for prompt in prompt_to_import['prompt']:
                    item = json.loads(prompt)
                    if 'type' in item and item['type'] == 'placeholder':
                        prompts.append(PlaceholderMessage(**item))
                    else:
                        prompts.append(ChatMessage(**item))
                request = CreateChatPromptRequest(
                    name=prompt_to_import['name'],
                    type=CreateChatPromptType.CHAT,
                    prompt=prompts,
                    config=prompt_to_import.get('config', {}),
                    labels=prompt_to_import.get('labels', []),
                    tags=prompt_to_import.get('tags', []),
                )
            else:
                print(f'     Неизвестный тип промпта: {type(prompt_to_import["prompt"])}')
                failed += 1
                continue

            langfuse.api.prompts.create(request=request)
            migrated += 1

        except Exception as e:
            if 'already exists' in str(e).lower():
                print(f'   {prompt_to_import["name"]} уже существует, версия пропущена: {e}')
                failed += 1
            else:
                print(f'   Ошибка импорта {prompt_to_import["name"]}: {e}')
                failed += 1

    print('=' * 60)
    print(f'Импортировано {migrated} промптов, {failed} ошибок')
