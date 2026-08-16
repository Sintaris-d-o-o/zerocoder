"""CLI: задать вопрос RAG-консультанту по ювелирным изделиям.

Запуск (из корня проекта, после ingest — см. README.md):
    python -m src.main "Как ухаживать за золотыми изделиями?"
    python -m src.main "Как чистить серебро?" --backend chroma
"""

import argparse

from src.rag_pipeline import TOP_K, answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG-консультант по ювелирным изделиям")
    parser.add_argument("question", help="Вопрос на русском языке")
    parser.add_argument("--backend", choices=["faiss", "chroma"], default="faiss")
    parser.add_argument("--k", type=int, default=TOP_K, help="Сколько фрагментов подмешивать в контекст")
    args = parser.parse_args()

    result = answer_question(args.question, backend=args.backend, k=args.k)

    print(f"\nВопрос: {result.question}")
    print(f"База: {result.backend}\n")
    print("Ответ:")
    print(result.answer)
    print("\nИспользованные фрагменты:")
    for i, doc in enumerate(result.sources, start=1):
        source = doc.metadata.get("source", "документ")
        snippet = doc.page_content[:150].replace("\n", " ")
        print(f"  {i}. [{source}] {snippet}...")


if __name__ == "__main__":
    main()
