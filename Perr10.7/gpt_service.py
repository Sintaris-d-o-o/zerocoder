"""Обращение к модели и получение строгого JSON (задание 10.7).

Задание требует «ремонт» JSON. Он здесь есть, но как запасной путь, а не как основной:
сначала у модели просят машинный формат ответа (`response_format=json_object`), и тогда
чинить обычно нечего. Эталонная реализация из урока этого не делает и потому вынуждена
чинить ответ каждый раз — четырьмя функциями подряд.

Чинить всё равно приходится: не все модели поддерживают формат, а на старых ответ
приходит завёрнутым в ```json … ``` или с запятой перед закрывающей скобкой.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from models import Assessment

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env")
load_dotenv(HERE / ".env")

DEFAULT_MODEL = os.getenv("PRESCORE_MODEL", "gpt-4o-mini")
MAX_RETRIES = 3
TIMEOUT_S = 90

SYSTEM_PROMPT = """Ты HR-аналитик компании, которая сертифицирует медицинские изделия
по MDR и IVDR. Оцениваешь, насколько кандидат подходит под вакансию.

Как оценивать (шкала 0-100):
- совпадение роли и области — до 40 баллов;
- профильные навыки и регуляторные знания — до 30;
- опыт и уровень — до 10;
- формат занятости и локация — до 5;
- зарплатные ожидания — до 5;
- прочие сигналы: языки, домен, отрасль — до 10.

Решение по баллу: 60 и выше — подходит, 45-59 — почти подходит, ниже 45 — не подходит.

Правила:
- не выдумывай факты, которых нет в резюме;
- отсутствие подтверждения навыка — это не слабая сторона, а пункт missing_skills;
- слабые стороны формулируй по существу и проверяемо, без общих слов;
- пиши по-русски.

Отвечай строго объектом JSON с полями:
score (целое 0-100), strengths (массив строк), weaknesses (массив строк),
missing_skills (массив строк), summary (строка, 2-3 предложения),
risks (массив строк — чем кандидат рискован, может быть пустым)."""


# ─── Ремонт JSON ─────────────────────────────────────────────────────────────

def strip_code_block(text: str) -> str:
    """Снимает обёртку ```json … ```, которую модели добавляют по привычке."""
    match = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    return match.group(1) if match else text


def extract_json_payload(text: str) -> str:
    """Выдёргивает объект JSON из текста с пояснениями вокруг.

    Скобки считаются, а не ищутся с конца: модель нередко пишет «Готово!» после
    объекта, и поиск последней скобки цеплял бы лишнее.
    """
    start = text.find("{")
    if start < 0:
        return text
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return text[start:]


def sanitize_json(text: str) -> str:
    """Чинит то, что ломает json.loads чаще всего."""
    text = text.replace(" ", " ")
    text = re.sub(r",\s*([}\]])", r"\1", text)          # запятая перед скобкой
    text = re.sub(r"([{,]\s*)'([^']+)'(\s*:)", r'\1"\2"\3', text)   # ключ в кавычках
    text = re.sub(r":\s*'([^']*)'", r': "\1"', text)    # значение в кавычках
    return text


def repair_json(text: str) -> dict[str, Any]:
    """Последовательно снимает обёртки, пока не получится разобрать.

    Возвращает разобранный объект или пустой словарь — ошибка разбора не должна
    ронять приложение: резюме уже прочитано и оплачено, лучше показать пустой разбор
    и дать нажать «повторить».
    """
    for step in (lambda t: t, strip_code_block, extract_json_payload, sanitize_json,
                 lambda t: sanitize_json(extract_json_payload(strip_code_block(t)))):
        try:
            value = json.loads(step(text))
            if isinstance(value, dict):
                return value
        except (json.JSONDecodeError, TypeError):
            continue
    return {}


# ─── Запрос к модели ─────────────────────────────────────────────────────────

def build_messages(vacancy: str, resume: str, candidate: str = "") -> list[dict[str, str]]:
    who = f"Имя кандидата: {candidate}\n" if candidate.strip() else ""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"{who}ВАКАНСИЯ:\n{vacancy}\n\n"
            f"РЕЗЮМЕ КАНДИДАТА:\n{resume}\n\n"
            "Оцени кандидата и верни JSON."
        )},
    ]


def _default_caller(messages: list[dict[str, str]], model: str) -> str:
    """Единственное место, где сервис ходит в сеть."""
    from openai import OpenAI

    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise SystemExit("В .env нет OPENAI_API_KEY")
    client = OpenAI(api_key=key, timeout=TIMEOUT_S, max_retries=1)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,          # оценка человека — не место для фантазии
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


def assess(vacancy: str, resume: str, candidate: str = "", *,
           model: str = "", retries: int = MAX_RETRIES,
           caller: Callable[[list[dict[str, str]], str], str] = _default_caller
           ) -> tuple[Assessment, str]:
    """Оценка кандидата моделью. Возвращает разбор и текст ошибки (пустой, если всё хорошо).

    `caller` вынесен в параметр — так тесты проверяют разбор и повторы без обращения
    к сети и без расходов.
    """
    model = model or DEFAULT_MODEL
    messages = build_messages(vacancy, resume, candidate)
    last_error = ""

    for attempt in range(1, retries + 1):
        try:
            raw = caller(messages, model)
        except Exception as exc:                      # сеть, лимиты, ключ
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
                continue
            break

        payload = repair_json(raw)
        if payload:
            return Assessment.from_payload(payload), ""
        last_error = f"Модель вернула не JSON: {raw[:200]}"

    return Assessment(), last_error
