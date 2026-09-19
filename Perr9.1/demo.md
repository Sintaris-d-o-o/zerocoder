# Демонстрация задания 9.1

## Что показывать проверяющему

1. [результаты.md](результаты.md) — отчёт: что сделано по каждому пункту задания, таблица сравнения
   «только мощная модель» против «мультимодельность», выводы.
2. [results/steps/](results/steps/) — **скриншоты ключевых шагов**: реальный вывод запуска,
   отрисованный в PNG (`00_tokenization.png` … `07_summary.png`). Рядом лежат исходные `.txt`.
3. [results/comparison.png](results/comparison.png) — итоговый график сравнения (стоимость, время,
   качество); [results/cost_per_question.png](results/cost_per_question.png) — стоимость по каждому
   вопросу; [results/classifier_confusion.png](results/classifier_confusion.png) — матрица ошибок
   классификатора; [results/tokenization.png](results/tokenization.png) — токены по языкам.
4. [results/answers.md](results/answers.md) — все ответы обеих схем с оценками экзаменатора;
   [results/runs.csv](results/runs.csv) — по одной строке на каждый вызов модели (токены, цена, время);
   [results/summary.json](results/summary.json) — все итоговые числа.
5. [results/classifier_models.txt](results/classifier_models.txt) — сравнение двух локальных моделей
   в роли дежурного (`gemma4:e2b` против `gemma4:12b`).
6. [PE9.1_multimodel_agent.ipynb](PE9.1_multimodel_agent.ipynb) — ноутбук со всеми шагами и
   сохранённым выводом ячеек.

## Как повторить у себя

### Что нужно

- Python 3.12, ключ OpenAI в файле `.env` в корне репозитория (`OPENAI_API_KEY=...`, шаблон —
  [.env.example](.env.example)). Ключ нужен для `gpt-5`, `gpt-5-mini` и экзаменатора.
- [Ollama](https://ollama.com) с моделью `gemma4:12b` (`ollama pull gemma4:12b`, ~7,6 ГБ) — для
  дежурного и простых вопросов. Без Ollama локальные шаги не запустятся.

### Запуск

```powershell
cd Perr9.1
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe -m pytest tests -q          # тесты логики без обращения к моделям
.\.venv\Scripts\python.exe run_experiment.py --smoke   # быстрая проверка на 3 вопросах (~5 мин)
.\.venv\Scripts\python.exe run_experiment.py           # полный прогон на 24 вопросах (~1 час, ~1,5 $)
.\.venv\Scripts\python.exe make_screenshots.py         # PNG-скриншоты шагов и графики
.\.venv\Scripts\python.exe make_notebook.py            # собрать и выполнить ноутбук
```

Полезно знать:

- Все ответы моделей складываются в `results/cache.jsonl`. Повторный запуск с теми же вопросами и
  моделями не делает платных запросов — берёт ответы из кэша (это стратегия «кэш ответов» из урока).
  Чтобы прогнать всё заново «с нуля», удалите `results/cache.jsonl` или запустите с `--no-cache`.
- `python compare_classifiers.py gemma4:e2b gemma4:12b` — сравнить любые локальные модели в роли
  дежурного (бесплатно, только Ollama).
- Модели и промпты — в начале [agent.py](agent.py) (словарь `ROLES`), тарифы — в
  [pricing.json](pricing.json), вопросы — в [queries.json](queries.json).

### Ноутбук в Google Colab

Ноутбук [PE9.1_multimodel_agent.ipynb](PE9.1_multimodel_agent.ipynb) импортирует `agent.py` и
`run_experiment.py`, поэтому в Colab нужно загрузить рядом с ним файлы `agent.py`,
`run_experiment.py`, `make_screenshots.py`, `queries.json`, `pricing.json` и создать `.env` с ключом.
Локальной Ollama в Colab нет, поэтому шаги с дежурным и простыми вопросами там не выполнятся —
их вывод сохранён в самом ноутбуке и в `results/`.

## Как дать доступ проверяющему

Репозиторий публичный: https://github.com/Sintaris-d-o-o/zerocoder, папка `Perr9.1`. Для сдачи на
платформе прикладываются PNG из `results/steps/` и `results/*.png` плюс текст из
[ответ-для-отправки.md](ответ-для-отправки.md).
