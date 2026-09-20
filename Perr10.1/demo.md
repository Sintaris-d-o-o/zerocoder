# Демонстрация задания 10.1

## Что показывать проверяющему

1. [результаты.md](результаты.md) — отчёт и сверка с заданием.
2. [yandex_art.py](yandex_art.py) — код генератора: сборка запроса, опрос операции,
   декодирование картинки, сохранение.
3. [tests/test_yandex_art.py](tests/test_yandex_art.py) — 21 тест, проверяющий логику без
   обращения к платному API.
4. [results/](results/) — **сгенерированные логотипы**; в подпапке
   [v1-длинные-промпты/](results/v1-длинные-промпты/) — первая серия, для сравнения.

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
| `yandex_art.py --check` | проверить доступ, ничего не тратя |
| `yandex_art.py --model aliceai-image-art-3.0/latest` | другая модель генерации |
| `yandex_art.py --size 1280x720` | другой размер картинки |
| `yandex_art.py --legacy` | способ из урока (сейчас отвечает отказом) |
