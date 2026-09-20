"""Готовит обезличенные резюме в PDF для проверки сервиса (задание 10.7).

Настоящие резюме в отчёт не годятся: это персональные данные, а отчёт уходит
проверяющему. Поэтому кандидаты вымышленные, а тексты написаны под наши вакансии —
регуляторика медицинских изделий.

Три резюме подобраны так, чтобы попасть в три разных исхода:
    profile   — профильный кандидат, должен пройти;
    partial   — смежный опыт, должен попасть в «почти подходит»;
    offtopic  — чужая область, должен отсеяться.

Запуск:
    python make_resumes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "data" / "resumes"

# Кириллица требует шрифта с нужными глифами — встроенные в reportlab их не имеют.
FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\calibri.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]

RESUMES: dict[str, tuple[str, str]] = {
    "profile-regulatory": ("Анна Ковалевская — специалист по регуляторным вопросам", """
<b>Анна Ковалевская</b><br/>
Специалист по регуляторным вопросам, медицинские изделия<br/>
Берлин, готова к удалённой работе. Английский B2, немецкий B1.

<b>Опыт работы: 6 лет</b>

<b>2021 — настоящее время. Производитель медицинских изделий класса IIb</b><br/>
Ведение технических файлов по Регламенту 2017/745 (MDR) для линейки изделий
класса IIa и IIb. Подготовка документации для нотифицированного органа, переписка
по замечаниям и закрытие несоответствий. Поддержка анализа рисков по ISO 14971.
Подготовка отчётов клинической оценки совместно с клиническим отделом.
Регистрация изделий в EUDAMED, присвоение UDI. Отвечала за CE-маркировку и
декларацию соответствия по двум продуктовым линейкам.

<b>2019 — 2021. Дистрибьютор медицинского оборудования</b><br/>
Поддержка системы менеджмента качества по ISO 13485. Участие в аудите
нотифицированного органа со стороны компании. Ведение постмаркетингового надзора
(PMS), обработка обращений и vigilance-отчётность.

<b>Навыки</b><br/>
MDR, ISO 13485, ISO 14971, технический файл, нотифицированный орган,
клиническая оценка, EUDAMED, UDI, CE-маркировка, PMS.

<b>О себе</b><br/>
Внимательность к деталям, самостоятельность, опыт коммуникации с нотифицированным
органом напрямую. Веду обучение младших коллег.
"""),
    "partial-quality": ("Дмитрий Орлов — инженер по качеству", """
<b>Дмитрий Орлов</b><br/>
Инженер по качеству<br/>
Варшава, гибридный формат. Английский B1.

<b>Опыт работы: 3 года</b>

<b>2022 — настоящее время. Производитель промышленного оборудования</b><br/>
Поддержка системы менеджмента качества по ISO 9001. Проведение внутренних аудитов,
работа с несоответствиями и CAPA. Контроль документации: процедуры, записи,
валидация производственных процессов. Подготовка к внешним аудитам.

<b>2021 — 2022. Контрактный производитель электроники</b><br/>
Входной контроль, работа с поставщиками, ведение записей качества.

<b>Навыки</b><br/>
ISO 9001, внутренние аудиты, CAPA, валидация процессов, управление документацией,
статистические методы контроля.

<b>О себе</b><br/>
Ответственность, коммуникация с производственными командами, работа в команде.
Сейчас изучаю требования ISO 13485 к медицинским изделиям.
"""),
    "offtopic-frontend": ("Павел Ситник — frontend-разработчик", """
<b>Павел Ситник</b><br/>
Frontend-разработчик<br/>
Удалённо. Английский B2.

<b>Опыт работы: 7 лет</b>

<b>2020 — настоящее время. Продуктовая компания</b><br/>
Разработка интерфейсов на React и TypeScript. Проектирование компонентной
библиотеки, оптимизация производительности, покрытие тестами. Работа в команде
из шести человек, код-ревью, наставничество.

<b>2017 — 2020. Веб-студия</b><br/>
Вёрстка и разработка клиентских сайтов, Next.js, интеграция с REST API.

<b>Навыки</b><br/>
React, TypeScript, Next.js, JavaScript, HTML, CSS, тестирование интерфейсов, Git.

<b>О себе</b><br/>
Коммуникация, работа в команде, самостоятельность. Интересуюсь доступностью
интерфейсов и дизайн-системами.
"""),
}


def register_font() -> str:
    for path in FONT_CANDIDATES:
        if path.exists():
            pdfmetrics.registerFont(TTFont("Body", str(path)))
            return "Body"
    raise SystemExit("Не найден шрифт с кириллицей — укажите свой в FONT_CANDIDATES")


def build(name: str, title: str, body: str, font: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.pdf"
    styles = getSampleStyleSheet()
    style = ParagraphStyle("body", parent=styles["Normal"], fontName=font,
                           fontSize=10.5, leading=15)
    heading = ParagraphStyle("heading", parent=style, fontSize=14, leading=18,
                             spaceAfter=8)
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm,
                            title=title)
    flow = [Paragraph(title, heading), Spacer(1, 4)]
    for block in body.strip().split("\n\n"):
        flow.append(Paragraph(block.replace("\n", " "), style))
        flow.append(Spacer(1, 6))
    doc.build(flow)
    return path


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    font = register_font()
    for name, (title, body) in RESUMES.items():
        path = build(name, title, body, font)
        print(f"  {path.name:28} {path.stat().st_size // 1024} КБ")
    print(f"\nГотово: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
