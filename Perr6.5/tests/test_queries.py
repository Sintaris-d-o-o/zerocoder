"""Тестовые запросы к RAG-консультанту: сравнение баз FAISS и ChromaDB.

Запуск как скрипта (сохраняет отчёт в tests/test-results.md):
    python -m tests.test_queries

Запуск как pytest (быстрые проверки на здравый смысл):
    pytest tests/test_queries.py
"""

import time
from pathlib import Path

import pytest

from src.rag_pipeline import answer_question, build_llm

QUESTIONS = [
    "Как ухаживать за золотыми изделиями?",
    "Посоветуй кольцо с бриллиантом",
    "Какие материалы используются в ювелирном деле?",
    "Как чистить серебро?",
    "Как ухаживать за изделиями из янтаря?",
]

BACKENDS = ["faiss", "chroma"]

RESULTS_PATH = Path(__file__).resolve().parent / "test-results.md"


def run_all(questions=QUESTIONS, backends=BACKENDS):
    llm = build_llm()
    rows = []
    for question in questions:
        for backend in backends:
            start = time.perf_counter()
            result = answer_question(question, backend=backend, llm=llm)
            elapsed = time.perf_counter() - start
            rows.append((elapsed, result))
    return rows


def write_report(rows, path: Path = RESULTS_PATH) -> None:
    lines = ["# Результаты тестовых запросов\n"]
    for elapsed, result in rows:
        lines.append(f"## «{result.question}» — база: {result.backend} ({elapsed:.1f} сек)\n")
        lines.append("**Ответ:**\n")
        lines.append(result.answer.strip() + "\n")
        lines.append("**Использованные фрагменты:**\n")
        for i, doc in enumerate(result.sources, start=1):
            source = doc.metadata.get("source", "документ")
            snippet = doc.page_content[:200].replace("\n", " ")
            lines.append(f"{i}. [{source}] {snippet}...")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("question", QUESTIONS)
def test_answer_is_grounded(question, backend):
    result = answer_question(question, backend=backend)
    assert result.answer.strip(), "модель вернула пустой ответ"
    assert result.sources, "поиск не нашёл ни одного фрагмента в базе"


if __name__ == "__main__":
    rows = run_all()
    write_report(rows)
    print(f"Готово: результаты сохранены в {RESULTS_PATH}")
