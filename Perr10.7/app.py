"""Веб-интерфейс прескоринга кандидатов на Streamlit (задание 10.7).

Запуск:
    streamlit run app.py

Ключ и модель берутся из .env:
    OPENAI_API_KEY   ключ
    PRESCORE_MODEL   модель, по умолчанию gpt-4o-mini
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

import gpt_service
import parser as resume_parser
import scoring
from models import THRESHOLD_MATCH, THRESHOLD_NEAR

HERE = Path(__file__).resolve().parent
VACANCIES_FILE = HERE / "data" / "vacancies.json"

st.set_page_config(page_title="Прескоринг кандидатов", page_icon="📄", layout="wide")


@st.cache_data
def load_vacancies() -> list[dict]:
    if not VACANCIES_FILE.exists():
        return []
    try:
        return json.loads(VACANCIES_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def score_color(score: int) -> str:
    if score >= THRESHOLD_MATCH:
        return "🟢"
    if score >= THRESHOLD_NEAR:
        return "🟡"
    return "🔴"


# ─── Боковая панель: веса и модель ───────────────────────────────────────────

st.sidebar.header("Настройки оценки")
st.sidebar.caption(
    "Ползунки задают, что важнее в этой вакансии. Сумму подгонять не нужно — "
    "веса приводятся к 100% автоматически."
)
hard = st.sidebar.slider("Профильные навыки", 0, 100, 55, 5)
experience = st.sidebar.slider("Опыт", 0, 100, 25, 5)
soft = st.sidebar.slider("Мягкие навыки", 0, 100, 20, 5)
weights = scoring.normalize_weights({"hard": hard, "experience": experience, "soft": soft})
st.sidebar.write({k: f"{v:.0%}" for k, v in weights.items()})

gpt_share = st.sidebar.slider(
    "Доля оценки модели в итоге", 0, 100, int(scoring.GPT_SHARE * 100), 5,
    help="Остальное даёт формула: она считается без модели и не меняется от запуска "
         "к запуску.") / 100

model = st.sidebar.text_input("Модель", gpt_service.DEFAULT_MODEL)

# ─── Основная форма ──────────────────────────────────────────────────────────

st.title("Прескоринг кандидатов на вакансии")
st.caption("Резюме в PDF сопоставляется с текстом вакансии: балл 0–100, сильные и "
           "слабые стороны, недостающие навыки.")

vacancies = load_vacancies()
left, right = st.columns(2)

with left:
    st.subheader("Вакансия")
    titles = [v["title"] for v in vacancies] + ["Своя вакансия"]
    chosen = st.selectbox("Выберите вакансию", titles, index=0 if vacancies else len(titles) - 1)
    if chosen == "Своя вакансия":
        vacancy_title = st.text_input("Название вакансии", "Вакансия")
        vacancy_text = st.text_area("Текст вакансии", height=300,
                                    placeholder="Задачи, требования, условия…")
    else:
        found = next(v for v in vacancies if v["title"] == chosen)
        vacancy_title = found["title"]
        vacancy_text = st.text_area("Текст вакансии", found["text"], height=300)

with right:
    st.subheader("Кандидат")
    candidate = st.text_input("Имя кандидата", placeholder="Иванов Иван")
    uploaded = st.file_uploader("Резюме в PDF", type=["pdf"])
    resume_text = ""
    if uploaded is not None:
        with st.spinner("Читаем PDF…"):
            try:
                resume_text = resume_parser.extract_pdf(uploaded.getvalue())
            except Exception as exc:
                st.error(f"Не удалось прочитать PDF: {exc}")
        if resume_text:
            st.success(f"Извлечено {len(resume_text)} знаков"
                       + (" (текст обрезан по длине)" if "обрезано" in resume_text else ""))
            with st.expander("Показать извлечённый текст"):
                st.text(resume_text)

run = st.button("Оценить кандидата", type="primary",
                disabled=not (vacancy_text.strip() and resume_text.strip()))

# ─── Оценка ──────────────────────────────────────────────────────────────────

if run:
    vacancy_clean = resume_parser.read_vacancy(vacancy_text)
    with st.spinner("Модель читает резюме…"):
        assessment, error = gpt_service.assess(vacancy_clean, resume_text, candidate,
                                               model=model)
    if error:
        st.error(f"Модель не ответила разбором: {error}")
        st.info("Формула ниже посчитана без модели — на неё это не влияет.")

    result = scoring.build_result(candidate, vacancy_title, vacancy_clean, resume_text,
                                  assessment, weights, model=model, gpt_share=gpt_share)
    scoring.save_result(result)

    st.divider()
    st.header(f"{score_color(result.final_score)} {result.final_score} из 100 — "
              f"{result.verdict}")

    columns = st.columns(4)
    columns[0].metric("Итог", result.final_score)
    columns[1].metric("Оценка модели", result.gpt_score)
    columns[2].metric("Формула", result.heuristic_score)
    columns[3].metric("Опыт", f"{result.breakdown.years:g} лет",
                      result.breakdown.seniority)

    if result.assessment.summary:
        st.write(result.assessment.summary)

    strong, weak = st.columns(2)
    with strong:
        st.subheader("Сильные стороны")
        for item in result.assessment.strengths or ["— модель ничего не выделила"]:
            st.markdown(f"- {item}")
    with weak:
        st.subheader("Слабые стороны")
        for item in result.assessment.weaknesses or ["— модель ничего не выделила"]:
            st.markdown(f"- {item}")

    if result.assessment.missing_skills:
        st.subheader("Чего не хватает")
        st.markdown("\n".join(f"- {item}" for item in result.assessment.missing_skills))

    if result.assessment.risks:
        st.subheader("Риски")
        st.markdown("\n".join(f"- {item}" for item in result.assessment.risks))

    with st.expander("Как посчитана формула"):
        breakdown = result.breakdown
        st.write({
            "Профильные навыки, %": breakdown.hard,
            "Опыт, %": breakdown.experience,
            "Мягкие навыки, %": breakdown.soft,
            "Семейство вакансии": breakdown.family_vacancy,
            "Семейство резюме": breakdown.family_resume,
            "Совпадение роли": "да (+10)" if breakdown.family_match else "нет (−10)",
            "Веса": {k: f"{v:.0%}" for k, v in result.weights.items()},
        })
        st.caption("Совпавшие требования: " + (", ".join(breakdown.matched_skills[:25])
                                               or "—"))
        st.caption("Не подтверждено: " + (", ".join(breakdown.missing_skills[:25]) or "—"))

# ─── История ─────────────────────────────────────────────────────────────────

st.divider()
st.header("История проверок")
history = scoring.load_history()
if not history:
    st.info("Пока пусто: оцените первого кандидата.")
else:
    st.dataframe(scoring.history_rows(history), use_container_width=True, hide_index=True)
    st.download_button("Скачать историю (JSON)",
                       json.dumps(history, ensure_ascii=False, indent=2),
                       file_name="history.json", mime="application/json")
    if st.button("Очистить историю"):
        scoring.HISTORY_FILE.unlink(missing_ok=True)
        st.rerun()
