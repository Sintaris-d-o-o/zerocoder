# Демонстрация задания 10.1

## Что показывать проверяющему

1. [результаты.md](результаты.md) — отчёт и сверка с заданием.
2. [yandex_art.py](yandex_art.py) — код генератора: сборка запроса, опрос операции,
   декодирование картинки, сохранение.
3. [tests/test_yandex_art.py](tests/test_yandex_art.py) — 21 тест, проверяющий логику без
   обращения к платному API.
4. `results/` — сгенерированные картинки (появятся после получения ключа).

## Как запустить

```powershell
cd Perr10.1
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe -m pytest tests -q   # тесты, бесплатно
.\.venv\Scripts\python.exe yandex_art.py --list      # промпты проекта
.\.venv\Scripts\python.exe yandex_art.py --dry-run   # запрос без отправки, бесплатно
.\.venv\Scripts\python.exe yandex_art.py             # генерация (нужен ключ)
```

Ключ кладётся в `.env` в корне репозитория, шаблон — [.env.example](.env.example).
Что именно нужно завести в Yandex Cloud, описано в [результатах](результаты.md).

## Полезные команды

| Команда | Что делает |
|---|---|
| `yandex_art.py --key shield-check` | сгенерировать один конкретный промпт |
| `yandex_art.py --prompt "свой текст"` | свой промпт из командной строки |
| `yandex_art.py --seed 12345` | зафиксировать зерно, чтобы повторить результат |
| `yandex_art.py --ratio 16:9` | другое соотношение сторон |
