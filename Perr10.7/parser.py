"""Извлечение текста резюме из PDF (задание 10.7).

Резюме почти всегда приходит в PDF, а модель принимает текст. Между ними —
`pdfplumber` и чистка: в выгрузке из PDF переносы стоят там, где в исходнике был
край страницы, а не конец предложения, и такой текст модель читает хуже.

Объём ограничен: длинное резюме целиком отправлять незачем — платить за него
придётся, а полезного в конце обычно нет.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

MAX_TEXT_LENGTH = 6000

_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")     # перенос слова по слогам
_PARAGRAPH = re.compile(r"\n[ \t]*\n\s*")            # пустая строка = конец абзаца
_SINGLE_BREAK = re.compile(r"(?<![.\!\?:;])\n(?!\s*[-•*\d])")
_SPACES = re.compile(r"[ \t ]{2,}")

# Пустая строка в PDF — настоящая граница абзаца, и её надо сохранить. Поэтому
# абзацы прячутся под метку до склейки строк: иначе вторая пустая строка выглядит
# как обычный перенос и исчезает вместе с границей.
_PARA_MARK = "\x00PARA\x00"


def clean_text(text: str) -> str:
    """Приводит выгрузку из PDF к виду, пригодному для модели.

    Порядок важен: сначала склеиваются слова, разорванные переносом, потом —
    строки внутри абзаца. Если поменять местами, дефис останется посреди слова.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _PARAGRAPH.sub(_PARA_MARK, text)
    text = _SINGLE_BREAK.sub(" ", text)
    text = text.replace(_PARA_MARK, "\n\n")
    text = _SPACES.sub(" ", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def limit(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """Обрезает текст по границе абзаца, а не посреди слова."""
    if len(text) <= max_length:
        return text
    cut = text[:max_length]
    break_at = max(cut.rfind("\n\n"), cut.rfind(". "))
    if break_at > max_length * 0.6:          # обрезали не слишком рано
        cut = cut[:break_at + 1]
    return cut.rstrip() + "\n\n[…резюме обрезано по длине]"


def extract_pdf(source: bytes | str | Path | io.BytesIO,
                max_length: int = MAX_TEXT_LENGTH) -> str:
    """Текст из PDF: извлечение, чистка, ограничение длины.

    Принимает и байты (так отдаёт файл Streamlit), и путь к файлу.
    """
    import pdfplumber                      # импорт внутри: тесты чистки его не требуют

    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)

    pages: list[str] = []
    with pdfplumber.open(source) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return limit(clean_text("\n\n".join(pages)), max_length)


def read_vacancy(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """Текст вакансии вводится руками, но чистится теми же правилами."""
    return limit(clean_text(text), max_length)
