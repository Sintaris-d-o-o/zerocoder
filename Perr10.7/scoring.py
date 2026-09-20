"""Расчёт балла и история проверок (задание 10.7).

Балл складывается из двух частей:

* **формула** — считается здесь, без модели: совпадение навыков, годы опыта,
  упоминания мягких навыков. Она не меняется от запуска к запуску, её можно
  объяснить кандидату и проверить руками;
* **модель** — читает резюме целиком и видит то, чего в словарях нет.

Итог — взвешенная смесь. Одной модели доверять целиком не стоит: на одном и том же
резюме она выдаёт разные баллы, а решение о человеке должно быть воспроизводимым.
Одной формулы тоже мало: она не поймёт, что «вёл переписку с BSI» — это и есть опыт
работы с нотифицированным органом.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from models import Assessment, Breakdown, Result, verdict_for
from normalize import (
    normalize_text,
    overlap,
    position_family,
    seniority,
    tokens,
    years_of_experience,
)

HISTORY_FILE = Path(__file__).parent / "history.json"

DEFAULT_WEIGHTS: dict[str, float] = {"hard": 0.55, "experience": 0.25, "soft": 0.20}

# Доля итога, которую даёт модель. Остальное — формула.
GPT_SHARE = 0.55

# Мягкие навыки в нашей области: регуляторщик половину времени пишет и договаривается.
SOFT_SKILL_MARKERS: tuple[str, ...] = (
    "коммуникац", "переговор", "команд", "самостоятельн", "внимательн",
    "ответственн", "обучен", "наставнич", "английск", "english",
    "presentation", "stakeholder", "cross-functional", "teamwork", "communication",
)

# Максимум лет, после которого опыт перестаёт добавлять баллы: пятнадцать лет и
# двадцать пять для этой работы неразличимы.
EXPERIENCE_CAP = 15.0


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Приводит веса к сумме 1.

    Пользователь двигает три ползунка независимо, и их сумма почти никогда не равна
    ста. Нормализация избавляет от требования «попади ровно в 100%».
    """
    clean = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(clean.values())
    if not total:
        return dict(DEFAULT_WEIGHTS)
    return {k: v / total for k, v in clean.items()}


def hard_skill_score(vacancy: str, resume: str) -> tuple[float, list[str], list[str]]:
    """Какая доля требований вакансии подтверждена резюме."""
    required = tokens(vacancy)
    present = tokens(resume)
    if not required:
        return 0.0, [], []
    matched, missing = overlap(required, present)
    score = round(len(matched) / len(required) * 100, 2)
    return score, sorted(matched), sorted(missing)


def experience_score(resume: str) -> tuple[float, float]:
    """Годы опыта, приведённые к шкале 0–100."""
    years = years_of_experience(resume)
    return round(min(years, EXPERIENCE_CAP) / EXPERIENCE_CAP * 100, 2), years


def soft_skill_score(resume: str) -> float:
    """Доля упомянутых мягких навыков из нашего списка."""
    text = normalize_text(resume)
    hits = sum(1 for marker in SOFT_SKILL_MARKERS if marker in text)
    return round(hits / len(SOFT_SKILL_MARKERS) * 100, 2)


def analyse(vacancy: str, resume: str) -> Breakdown:
    """Вся детерминированная часть — то, что известно до обращения к модели."""
    hard, matched, missing = hard_skill_score(vacancy, resume)
    experience, years = experience_score(resume)
    return Breakdown(
        hard=hard,
        experience=experience,
        soft=soft_skill_score(resume),
        years=years,
        seniority=seniority(years),
        family_vacancy=position_family(vacancy),
        family_resume=position_family(resume),
        matched_skills=matched[:40],
        missing_skills=missing[:40],
    )


def heuristic_score(breakdown: Breakdown, weights: dict[str, float]) -> float:
    """Взвешенная сумма трёх частей плюс поправка за семейство должностей.

    Поправка — из агента N3, где совпадение роли весило 40 из 100 баллов. Без неё
    инженер-программист с богатым резюме обгонял бы профильного регуляторщика:
    общих слов в тексте много, а работа совсем другая.
    """
    w = normalize_weights(weights)
    base = (breakdown.hard * w.get("hard", 0)
            + breakdown.experience * w.get("experience", 0)
            + breakdown.soft * w.get("soft", 0))
    if breakdown.family_match:
        base += 10
    elif breakdown.family_resume != "unknown":
        base -= 10
    return round(max(0.0, min(100.0, base)), 2)


def blend(gpt_score: int, heuristic: float, gpt_share: float = GPT_SHARE) -> int:
    """Итоговый балл из оценки модели и формулы."""
    share = max(0.0, min(1.0, gpt_share))
    return int(round(gpt_score * share + heuristic * (1 - share)))


def build_result(candidate: str, vacancy_title: str, vacancy: str, resume: str,
                 assessment: Assessment, weights: dict[str, float],
                 model: str = "", gpt_share: float = GPT_SHARE) -> Result:
    breakdown = analyse(vacancy, resume)
    heuristic = heuristic_score(breakdown, weights)
    final = blend(assessment.score, heuristic, gpt_share)
    return Result(
        candidate=candidate.strip() or "Без имени",
        vacancy_title=vacancy_title.strip() or "Вакансия",
        final_score=final,
        gpt_score=assessment.score,
        heuristic_score=int(round(heuristic)),
        verdict=verdict_for(final),
        weights=normalize_weights(weights),
        assessment=assessment,
        breakdown=breakdown,
        model=model,
    )


# ─── История ─────────────────────────────────────────────────────────────────

def load_history(path: Path = HISTORY_FILE) -> list[dict[str, Any]]:
    """Прошлые проверки. Битый файл не должен ронять приложение."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_result(result: Result, path: Path = HISTORY_FILE) -> list[dict[str, Any]]:
    """Дописывает проверку в историю.

    Запись идёт через временный файл: приложение может быть открыто в двух вкладках,
    и обрыв записи посреди файла стоил бы всей истории.
    """
    items = load_history(path)
    items.insert(0, result.to_dict())
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return items


def history_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Плоские строки для таблицы сравнения кандидатов."""
    rows = []
    for item in items:
        breakdown = item.get("breakdown") or {}
        rows.append({
            "Кандидат": item.get("candidate", ""),
            "Вакансия": item.get("vacancy_title", ""),
            "Итог": item.get("final_score", 0),
            "Модель": item.get("gpt_score", 0),
            "Формула": item.get("heuristic_score", 0),
            "Решение": item.get("verdict", ""),
            "Опыт, лет": breakdown.get("years", 0),
            "Уровень": breakdown.get("seniority", ""),
        })
    return rows
