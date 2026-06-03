import argparse

from exporter import export_prompts
from importer import import_prompts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Langfuse Prompt Exporter')
    parser.add_argument(
        '-f', '--file',
        type=str,
        default='prompts_backup.json',
        help='Имя файла для промптов (по умолчанию: prompts_backup.json)'
    )
    parser.add_argument(
        '--export', 
        action='store_true', 
        dest='export_mode',
        help='Режим экспорта промтов из Langfuse'
    )
    parser.add_argument(
        '--import', 
        action='store_true', 
        dest='import_mode',
        help='Режим импорта промтов в Langfuse'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help='Лимит элементов на страницу при экспорте (по умолчанию: 100)'
    )
    parser.add_argument(
        '--page',
        type=int,
        default=1,
        help='Первая страница при экспорте (по умолчанию: 1)'
    )
    args = parser.parse_args()
    if args.export_mode:
        export_prompts(output_file_name=args.file, page=args.page, limit=args.limit)
    elif args.import_mode:
        import_prompts(input_file_name=args.file)
        # pass
    else:
        print('Необходимо задать флаг "--export" или "--import"')
