# Демонстрация задания 10.3

## Что показывать проверяющему

1. [результаты.md](результаты.md) — отчёт и сверка с заданием.
2. [results/20260920-1007_lab.mp4](results/20260920-1007_lab.mp4) — **готовое видео**, 4 секунды.
3. [results/01_терминал.png](results/01_терминал.png) — **скриншот терминала**: прогресс-бар,
   статусы, стоимость.
4. [results/frames/](results/frames/) — кадры из видео.
5. [video_generator.py](video_generator.py) — генерация: создание задачи, ожидание, скачивание.
6. [test.py](test.py) — проверка статуса, файл из условия задания.
7. [tests/](tests/) — 36 тестов без обращения к платному сервису.

## Как запустить

```powershell
cd Perr10.3
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe -m pytest tests -q        # тесты, бесплатно
.\.venv\Scripts\python.exe video_generator.py --check   # доступ и остаток на счету, бесплатно
.\.venv\Scripts\python.exe video_generator.py --list     # промпты и модели, бесплатно
.\.venv\Scripts\python.exe video_generator.py --dry-run  # запрос без отправки, бесплатно
.\.venv\Scripts\python.exe video_generator.py            # генерация (тратит деньги)
```

Ключ кладётся в `.env` в корне репозитория, шаблон — [.env.example](.env.example).

## Сколько это стоит

Генерация видео — самая дорогая операция модуля. Замеры на 4-секундных роликах:

| Модель | Стоимость | Время |
|---|---:|---:|
| `google/veo-3.1-lite` | 21,9 кредита | 35 с |
| `google/veo-3.1-fast` | 43,8 кредита | 20 с |

Перед запуском стоит посмотреть остаток: `python video_generator.py --check` — он же скажет,
на сколько роликов хватит.

## Если генерация долгая

Ждать в том же окне не нужно. Скрипт печатает идентификатор задачи, дальше:

```powershell
python test.py <идентификатор>             # статус
python test.py --watch <идентификатор>     # дождаться и скачать
python test.py --download <идентификатор>  # скачать уже готовое
```

Так был забран первый ролик: задача осталась на сервисе, оплаченный результат не пропал.

## Полезные команды

| Команда | Что делает |
|---|---|
| `video_generator.py --key lab` | другой заготовленный промпт |
| `video_generator.py --prompt "текст"` | свой промпт |
| `video_generator.py --model openai/sora-2-pro` | модель Sora из урока |
| `video_generator.py --seconds 8` | другая длительность |
| `video_generator.py --provider proxyapi` | сервис из урока (нужен свой ключ) |
| `make_screenshot.py` | нарисовать скриншот терминала из сохранённого вывода |

## Как снят скриншот терминала

Вывод сохраняется в файл, затем отрисовывается в картинку:

```powershell
python video_generator.py | tee results\terminal.log
python make_screenshot.py
```

Прогресс-бар перерисовывает одну строку, поэтому в файле он лежит одной длинной строкой —
скрипт разворачивает её обратно по шагам, и на картинке виден весь ход генерации.
