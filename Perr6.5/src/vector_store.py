"""Построение и загрузка векторных баз FAISS и ChromaDB на одних и тех же данных."""

import pickle
from pathlib import Path

import faiss as faiss_lib
import numpy as np
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.document_loader import load_and_split

# Многоязычная модель эмбеддингов — понимает русский, работает полностью локально
# после первого скачивания (кешируется в ~/.cache/huggingface).
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_DIR = BASE_DIR / "storage" / "faiss_index"
CHROMA_DIR = BASE_DIR / "storage" / "chroma_db"
CHROMA_COLLECTION = "jewelry"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _save_faiss_local(store: FAISS) -> None:
    """Сохраняет индекс FAISS на диск в обход store.save_local().

    store.save_local() вызывает faiss.write_index() — функцию C++-библиотеки faiss,
    которая открывает файл через узкую (не Unicode) строку. На Windows это падает с
    "No such file or directory", если путь к проекту содержит не-ASCII символы (в этом
    репозитории — кириллица в пути, например "C:\\Проекты\\..."), даже когда папка
    реально существует. Обходим это: сериализуем индекс в байты в памяти
    (faiss.serialize_index) и пишем их обычным Python-файлом open()/write_bytes(),
    который корректно работает с любыми путями на Windows.
    """
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    index_bytes = faiss_lib.serialize_index(store.index)
    (FAISS_DIR / "index.faiss").write_bytes(index_bytes.tobytes())
    with open(FAISS_DIR / "index.pkl", "wb") as f:
        pickle.dump((store.docstore, store.index_to_docstore_id), f)


def build_faiss_index(chunks=None, embeddings=None) -> FAISS:
    chunks = chunks if chunks is not None else load_and_split()
    embeddings = embeddings or get_embeddings()
    store = FAISS.from_documents(chunks, embeddings)
    _save_faiss_local(store)
    return store


def load_faiss_index(embeddings=None) -> FAISS:
    embeddings = embeddings or get_embeddings()
    index_bytes = (FAISS_DIR / "index.faiss").read_bytes()
    index = faiss_lib.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8))
    # pickle здесь безопасен: файл создаётся локально этим же проектом (build_faiss_index),
    # а не скачивается из внешнего источника.
    with open(FAISS_DIR / "index.pkl", "rb") as f:
        docstore, index_to_docstore_id = pickle.load(f)
    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )


def build_chroma_index(chunks=None, embeddings=None) -> Chroma:
    chunks = chunks if chunks is not None else load_and_split()
    embeddings = embeddings or get_embeddings()
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=CHROMA_COLLECTION,
    )


def load_chroma_index(embeddings=None) -> Chroma:
    embeddings = embeddings or get_embeddings()
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION,
    )


def build_all() -> tuple[FAISS, Chroma]:
    """Загружает документы один раз и строит обе базы на одинаковых чанках/эмбеддингах."""
    chunks = load_and_split()
    embeddings = get_embeddings()
    faiss_store = build_faiss_index(chunks, embeddings)
    chroma_store = build_chroma_index(chunks, embeddings)
    return faiss_store, chroma_store


if __name__ == "__main__":
    build_all()
    print(f"FAISS индекс сохранён в {FAISS_DIR}")
    print(f"ChromaDB индекс сохранён в {CHROMA_DIR}")
