from datetime import datetime
import json
from pathlib import Path

from langfuse.api import Prompt_Chat

from core import settings


def export_prompts(output_file_name: str, page: int, limit: int):
    try:
        langfuse = settings.langfuse.client
    except Exception as e:
        print(f'Ошибка подключения к Langfuse: {e}')
        return

    prompts_data = []
    print('Экспорт промптов')
    while True:
        try:
            overview = langfuse.api.prompts.list(page=page, limit=limit)
            if not overview.data:
                break
            print(f'Страница {overview.meta.page}/{overview.meta.total_pages}')
            for prompt_info in overview.data:
                prompt_name = prompt_info.name
                for v_num in prompt_info.versions:
                    try:
                        prompt_detail = langfuse.api.prompts.get(prompt_name, version=v_num)
                        if isinstance(prompt_detail, Prompt_Chat):
                            decode_prompt = [prompt.json() for prompt in prompt_detail.prompt]
                        else:
                            decode_prompt = prompt_detail.prompt

                        prompt_dict = {
                            'name': prompt_detail.name,
                            'version': prompt_detail.version,
                            'labels': prompt_detail.labels,
                            'prompt': decode_prompt,
                            'config': prompt_detail.config,
                            'tags': prompt_detail.tags,
                        }
                        prompts_data.append(prompt_dict)

                    except Exception as e:
                        print(f'   Ошибка "{prompt_name}" v{v_num}: {e}')

            if overview.meta.page >= overview.meta.total_pages:
                break
            page += 1

        except Exception as e:
            print(f'Ошибка при листинге: {e}')
            break

    if not prompts_data:
        print('Промпты не найдены. Экспорт прерван')
        return

    with open(output_file_name, 'w', encoding='utf-8') as f:
        json.dump({
            'exported_at': datetime.now().isoformat(),
            'source_host': settings.langfuse.base_url,
            'prompts': prompts_data
        }, f, ensure_ascii=False, indent=2)

    print('=' * 60)
    print(f'Экспортировано {len(prompts_data)} промптов')
    print(f'Файл: {Path(output_file_name).absolute()}')
