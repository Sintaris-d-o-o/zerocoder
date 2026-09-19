"""Собирает ноутбук PE9.1_multimodel_agent.ipynb из тех же функций, что и run_experiment.py,
и выполняет его (nbclient), чтобы в файле остался вывод каждой ячейки.

Код не дублируется: ячейки импортируют agent.py / run_experiment.py (урок 8.2: одна правка — оба места).
Благодаря кэшу results/cache.jsonl выполнение ноутбука не делает повторных платных запросов.

Запуск: python make_notebook.py [--no-execute]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
OUT = HERE / "PE9.1_multimodel_agent.ipynb"

CELLS = [
    ("md", """# PE9.1 — Мульти-модельный агент для Норм-ассистента SINTARIS

Повторяем шаги урока 9.1 на своей теме: вопросы по сертификатам и регламентам медизделий (MDR/IVDR).

| Роль | Модель | Где |
|---|---|---|
| дежурный (intent) | Gemma 4 (Ollama) | локально, цена 0 |
| simple | Gemma 4 (Ollama) | локально, цена 0 |
| medium | gpt-5-mini | OpenAI |
| hard | gpt-5 | OpenAI |
| экзаменатор | gpt-5 | OpenAI |

Ключ OpenAI берётся из `.env` (переменная `OPENAI_API_KEY`) — в код не вставляется.
Локальные шаги требуют запущенного Ollama (`OLLAMA_HOST`, по умолчанию `http://localhost:11434`);
в Google Colab их повторить нельзя, облачные шаги (режим А, экзаменатор) — можно.
"""),
    ("code", """# Библиотеки (в локальном .venv уже установлены; в Colab раскомментировать):
# !pip install openai tiktoken python-dotenv pandas matplotlib pillow
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd()))
import agent as ag
import run_experiment as rx
print("openai", __import__("openai").__version__, "| Ollama:", ag.OLLAMA_HOST, ag.OLLAMA_MODEL)"""),
    ("md", "## Конфигурация ролей и функция вызова модели\n\nАналог `config` и `call_model` из урока — см. `agent.py`."),
    ("code", """import inspect
print(inspect.getsource(ag.Agent.call_model))"""),
    ("code", """queries = rx.load_queries()
ctx = rx.Context(rx.HERE / "results", queries)   # кэш results/cache.jsonl: повторные запросы не оплачиваются
print(len(queries), "вопросов; первые три:")
for q in queries[:3]:
    print(f"  {q['id']} [{q['tier']}] {q['question']}")"""),
    ("md", "## Шаг 0. Токенизация: одна фраза на разных языках"),
    ("code", "rx.step0_tokenization(ctx)"),
    ("md", "## Шаг 1. Конфигурация: модели, промпты, тарифы"),
    ("code", "rx.step1_config(ctx)"),
    ("md", "## Шаг 2. Пробные запросы: простой и сложный вопрос"),
    ("code", "rx.step2_call_model(ctx)"),
    ("md", "## Шаг 3. Проверяем, корректно ли дежурный классифицирует запросы"),
    ("code", "rx.step3_classifier(ctx)"),
    ("md", "## Шаг 4. Режим А — только мощная модель (gpt-5)"),
    ("code", "rx.step4_mode_a(ctx)"),
    ("md", "## Шаг 5. Режим Б — мультимодельность (дежурный + модель по уровню)"),
    ("code", "rx.step5_mode_b(ctx)"),
    ("md", "## Шаг 6. Экзаменатор — качество ответов вслепую"),
    ("code", "rx.step6_judge(ctx)"),
    ("md", "## Шаг 7. Сравнение двух результатов"),
    ("code", "summary = rx.step7_summary(ctx)\nrx.save_results(ctx)"),
    ("code", """from IPython.display import Image, display
import make_screenshots as ms
ms.chart_comparison(summary, rx.HERE / "results" / "comparison.png")
display(Image(filename=str(rx.HERE / "results" / "comparison.png")))"""),
]


def build() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    for kind, src in CELLS:
        nb.cells.append(new_markdown_cell(src) if kind == "md" else new_code_cell(src))
    return nb


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-execute", action="store_true")
    args = ap.parse_args(argv)
    nb = build()
    if not args.no_execute:
        from nbclient import NotebookClient
        client = NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}})
        client.execute()
    nbformat.write(nb, OUT)
    n_out = sum(len(c.get("outputs", [])) for c in nb.cells if c.cell_type == "code")
    print(f"→ {OUT.name}: {len(nb.cells)} ячеек, {n_out} выводов")


if __name__ == "__main__":
    main()
