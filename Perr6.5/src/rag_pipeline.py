"""RAG-цепочка: поиск релевантных фрагментов + генерация ответа локальной моделью Ollama."""

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from src.vector_store import get_embeddings, load_chroma_index, load_faiss_index

OLLAMA_MODEL = "llama3.2:3b"
TOP_K = 4

SYSTEM_PROMPT = (
    "Ты — консультант ювелирного магазина. Отвечай на вопросы покупателей ТОЛЬКО на основе "
    "приведённого ниже контекста из внутренних памяток магазина. Если в контексте нет ответа "
    "на вопрос — честно скажи, что не располагаешь такой информацией, не придумывай факты. "
    "Отвечай кратко и по-русски."
)


@dataclass
class RagAnswer:
    question: str
    answer: str
    sources: list[Document]
    backend: str


def _format_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "документ")
        parts.append(f"[Фрагмент {i} из «{source}»]\n{doc.page_content}")
    return "\n\n".join(parts)


def build_llm() -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.2,
        # num_predict ограничивает длину ответа, repeat_penalty снижает риск, что модель
        # зациклится на повторе одних и тех же токенов (на практике на этой модели без
        # этих параметров генерация иногда уходила в бесконечный повтор списка слов).
        num_predict=512,
        repeat_penalty=1.3,
    )


def answer_question(question: str, backend: str = "faiss", k: int = TOP_K, llm=None) -> RagAnswer:
    embeddings = get_embeddings()
    if backend == "faiss":
        store = load_faiss_index(embeddings)
    elif backend == "chroma":
        store = load_chroma_index(embeddings)
    else:
        raise ValueError(f"Неизвестный backend: {backend!r} (ожидался 'faiss' или 'chroma')")

    docs = store.similarity_search(question, k=k)
    context = _format_context(docs)

    llm = llm or build_llm()
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Контекст:\n{context}\n\nВопрос: {question}"),
    ]
    response = llm.invoke(messages)
    return RagAnswer(question=question, answer=response.content, sources=docs, backend=backend)
