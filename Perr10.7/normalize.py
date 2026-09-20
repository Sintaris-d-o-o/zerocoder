"""Нормализация текста резюме и вакансии — перенос сильных сторон агента N3.

В n8n-агенте N3 («HR Assistant», 27 узлов) самой полезной частью оказался не сам
вызов модели, а подготовка к нему: приведение к нижнему регистру, склейка синонимов,
семейства должностей и уровни по годам опыта. Благодаря этому «React-разработчик» и
«frontend engineer» попадали в одно семейство, а «js» и «JavaScript» считались одним
навыком — без этого совпадения теряются на ровном месте.

Здесь то же самое сделано обычными функциями: их видно в тестах, и они работают
одинаково от запуска к запуску, в отличие от ответа модели.

Домен другой: в N3 это были ИТ-должности, здесь — регуляторика медицинских изделий.
Устройство осталось прежним, поменялись словари.
"""
from __future__ import annotations

import re
from typing import Iterable

# ─── Синонимы ────────────────────────────────────────────────────────────────
# Слева — то, что пишут в резюме и вакансиях, справа — единая форма.
# Без этой склейки «MDR» и «Regulation 2017/745» считались бы разными навыками.
SYNONYMS: dict[str, str] = {
    "regulation 2017/745": "mdr",
    "2017/745": "mdr",
    "medical device regulation": "mdr",
    "регламент 2017/745": "mdr",
    "мдр": "mdr",
    "regulation 2017/746": "ivdr",
    "2017/746": "ivdr",
    "ивдр": "ivdr",
    "iso13485": "iso 13485",
    "исо 13485": "iso 13485",
    "iso 14971": "iso 14971",
    "исо 14971": "iso 14971",
    "управление рисками": "iso 14971",
    "risk management": "iso 14971",
    "технический файл": "technical file",
    "техническая документация": "technical file",
    "technical documentation": "technical file",
    "нотифицированный орган": "notified body",
    "нотифицированного органа": "notified body",
    "notified bodies": "notified body",
    "клинические испытания": "clinical evaluation",
    "клиническая оценка": "clinical evaluation",
    "clinical evaluation report": "clinical evaluation",
    "cer": "clinical evaluation",
    "post-market surveillance": "pms",
    "постмаркетинговый надзор": "pms",
    "пострыночный надзор": "pms",
    "система менеджмента качества": "qms",
    "смк": "qms",
    "quality management system": "qms",
    "регуляторные вопросы": "regulatory affairs",
    "regulatory affairs": "regulatory affairs",
    "ra-специалист": "regulatory affairs",
    "ce-маркировка": "ce marking",
    "ce marking": "ce marking",
    "маркировка ce": "ce marking",
    "udi": "udi",
    "eudamed": "eudamed",
}

# ─── Семейства должностей ────────────────────────────────────────────────────
# Кандидат и вакансия сравниваются не по точному названию, а по семейству:
# «инженер по качеству» и «QA engineer» — одна работа, написанная по-разному.
POSITION_FAMILIES: dict[str, tuple[str, ...]] = {
    "regulatory": ("regulatory affairs", "регуляторн", "ra specialist", "ra manager",
                   "регистрац", "mdr", "ivdr", "notified body", "ce marking"),
    "quality": ("quality", "качеств", "qms", "iso 13485", "аудит", "auditor",
                "quality assurance", "qa engineer"),
    "clinical": ("clinical", "клиническ", "clinical evaluation", "pms",
                 "vigilance", "биостатист"),
    "techwriter": ("technical writer", "технический писатель", "documentation specialist",
                   "technical file", "документац"),
    "rnd": ("r&d", "разработк", "development engineer", "конструктор", "design engineer"),
    "software": ("developer", "разработчик", "frontend", "backend", "fullstack",
                 "python", "javascript", "программист"),
}

# ─── Уровни по годам опыта ───────────────────────────────────────────────────
SENIORITY_BANDS: tuple[tuple[float, str], ...] = (
    (1.0, "junior"),
    (4.0, "middle"),
    (8.0, "senior"),
    (float("inf"), "lead"),
)

_YEARS = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:\+\s*)?(?:years?|yrs?|лет|года|год)\b",
    re.IGNORECASE)
_TOKEN = re.compile(r"[a-zа-яё0-9][a-zа-яё0-9/.\-]{2,}", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Нижний регистр, единые пробелы и склеенные синонимы.

    Синонимы заменяются от длинных к коротким: иначе «regulation 2017/745» успело бы
    превратиться в «mdr» по куску «2017/745» и потеряло бы остаток фразы.
    """
    result = (text or "").lower().replace("ё", "е")
    result = re.sub(r"\s+", " ", result)
    for source in sorted(SYNONYMS, key=len, reverse=True):
        if source in result:
            result = result.replace(source, SYNONYMS[source])
    return result.strip()


def tokens(text: str) -> set[str]:
    """Значимые слова нормализованного текста.

    Стоп-слова выброшены: без них «который» и «работа» из вакансии считались бы
    совпадением навыка и завышали бы балл любому резюме.
    """
    return {t for t in _TOKEN.findall(normalize_text(text)) if t not in STOP_WORDS}


STOP_WORDS: frozenset[str] = frozenset("""
опыт работы года лет года компании должности обязанности требования знание умение
навыки владение участие обеспечение проведение подготовка разработка ведение
которые который которая также более менее наших ваших свои этой этот для при про
что как так или его нее them with from this that have will your our the and for
""".split())


def years_of_experience(text: str) -> float:
    """Наибольшее число лет, названное в тексте.

    Берётся максимум, а не сумма: «3 года в X и 2 года в Y» — это не пять лет
    стажа, а два места работы, которые часто пересекаются по времени.
    """
    found = [float(m.replace(",", ".")) for m in _YEARS.findall(text or "")]
    return max(found, default=0.0)


def seniority(years: float) -> str:
    for limit, name in SENIORITY_BANDS:
        if years <= limit:
            return name
    return "lead"


def position_family(text: str) -> str:
    """Семейство должностей, к которому ближе всего текст.

    Считается по числу попаданий ключевых слов, а не по первому совпадению: в резюме
    слово «разработка» встречается почти всегда, и по первому совпадению любой
    регуляторщик оказался бы программистом.
    """
    normalized = normalize_text(text)
    hits = {family: sum(normalized.count(key) for key in keys)
            for family, keys in POSITION_FAMILIES.items()}
    best = max(hits, key=lambda f: hits[f])
    return best if hits[best] else "unknown"


def overlap(required: Iterable[str], present: Iterable[str]) -> tuple[set[str], set[str]]:
    """Что из требуемого нашлось, а чего нет."""
    required, present = set(required), set(present)
    return required & present, required - present
