"""Структуры данных сервиса прескоринга (задание 10.7)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any

# Пороги взяты из агента N3: там они отлажены на живом потоке резюме.
THRESHOLD_MATCH = 60      # приглашаем
THRESHOLD_NEAR = 45       # почти подходит, стоит посмотреть глазами


@dataclass
class Assessment:
    """Ответ модели. Поля — те, что требует задание."""
    score: int = 0
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    summary: str = ""
    risks: list[str] = field(default_factory=list)   # из N3: чем кандидат рискован

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Assessment":
        """Собирает ответ из разобранного JSON, прощая мелкие вольности модели.

        Модель иногда отвечает `strong_sides` вместо `strengths` или кладёт в список
        строку вместо массива. Это не повод терять весь разбор резюме, за который уже
        заплачено, поэтому синонимы полей и одиночные строки принимаются.
        """
        def as_list(value: Any) -> list[str]:
            if value is None:
                return []
            if isinstance(value, str):
                return [value] if value.strip() else []
            return [str(v).strip() for v in value if str(v).strip()]

        def pick(*names: str) -> Any:
            for name in names:
                if name in payload:
                    return payload[name]
            return None

        raw_score = pick("score", "final_score", "total") or 0
        try:
            score = int(round(float(raw_score)))
        except (TypeError, ValueError):
            score = 0

        return cls(
            score=max(0, min(100, score)),
            strengths=as_list(pick("strengths", "strong_sides", "pros")),
            weaknesses=as_list(pick("weaknesses", "weak_sides", "cons")),
            missing_skills=as_list(pick("missing_skills", "missing", "gaps")),
            summary=str(pick("summary", "conclusion") or "").strip(),
            risks=as_list(pick("risks", "red_flags")),
        )


@dataclass
class Breakdown:
    """Детерминированная часть оценки — считается без модели и всегда одинаково."""
    hard: float = 0.0
    experience: float = 0.0
    soft: float = 0.0
    years: float = 0.0
    seniority: str = "junior"
    family_vacancy: str = "unknown"
    family_resume: str = "unknown"
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)

    @property
    def family_match(self) -> bool:
        return (self.family_vacancy == self.family_resume
                and self.family_vacancy != "unknown")


@dataclass
class Result:
    """Итог по одному кандидату — то, что показывается и кладётся в историю."""
    candidate: str
    vacancy_title: str
    final_score: int
    gpt_score: int
    heuristic_score: int
    verdict: str
    weights: dict[str, float]
    assessment: Assessment
    breakdown: Breakdown
    model: str = ""
    created_at: float = field(default_factory=time.time)

    def as_row(self) -> dict[str, Any]:
        """Плоская строка для таблицы сравнения кандидатов."""
        return {
            "Кандидат": self.candidate,
            "Вакансия": self.vacancy_title,
            "Итог": self.final_score,
            "Модель": self.gpt_score,
            "Формула": self.heuristic_score,
            "Решение": self.verdict,
            "Опыт, лет": self.breakdown.years,
            "Уровень": self.breakdown.seniority,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verdict_for(score: int) -> str:
    """Словесное решение по баллу — пороги те же, что в агенте N3."""
    if score >= THRESHOLD_MATCH:
        return "подходит"
    if score >= THRESHOLD_NEAR:
        return "почти подходит"
    return "не подходит"
