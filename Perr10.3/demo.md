# Демонстрация задания 10.3

## Что показывать проверяющему

1. [результаты.md](результаты.md) — отчёт и сверка с заданием.
2. [video_generator.py](video_generator.py) — генерация: создание задачи, ожидание с
   прогресс-баром, скачивание MP4.
3. [test.py](test.py) — проверка статуса, файл из условия задания.
4. [tests/test_video_generator.py](tests/test_video_generator.py) — 26 тестов без обращения к
   платному сервису.
5. `results/` — видео и параметры прогона (появятся после получения ключа).

## Как запустить

```powershell
cd Perr10.3
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe -m pytest tests -q          # тесты, бесплатно
.\.venv\Scripts\python.exe test.py                      # проверить доступ, бесплатно
.\.venv\Scripts\python.exe video_generator.py --list     # промпты проекта
.\.venv\Scripts\python.exe video_generator.py --dry-run  # запрос без отправки, бесплатно
.\.venv\Scripts\python.exe video_generator.py            # генерация (нужен ключ и баланс)
```

## Если генерация долгая

Не нужно ждать в том же окне. Скрипт печатает идентификатор задачи, дальше можно:

```powershell
python test.py <идентификатор>            # посмотреть статус
python test.py --watch <идентификатор>    # дождаться и скачать
python test.py --list                      # список последних задач
```

## Полезные команды

| Команда | Что делает |
|---|---|
| `video_generator.py --key lab` | другой заготовленный промпт |
| `video_generator.py --prompt "текст"` | свой промпт |
| `video_generator.py --seconds 8` | другая длительность |
| `video_generator.py --size 1280x720` | другое разрешение |
