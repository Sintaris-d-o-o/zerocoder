"""Прогон нескольких резюме по одной вакансии из командной строки (задание 10.7).

Интерфейс на Streamlit удобно показывать, но неудобно повторять: каждый прогон —
это клики. Этот скрипт делает то же самое пачкой, печатает таблицу и пишет историю,
поэтому результат можно приложить к отчёту и повторить одной командой.

Запуск:
    python run_batch.py                       # все резюме из data/resumes по первой вакансии
    python run_batch.py --vacancy 2           # по второй вакансии
    python run_batch.py --dry-run             # без обращения к модели, только формула
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gpt_service
import parser as resume_parser
import scoring
from models import Assessment

HERE = Path(__file__).resolve().parent
RESUMES_DIR = HERE / "data" / "resumes"
VACANCIES_FILE = HERE / "data" / "vacancies.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vacancy", type=int, default=1, help="номер вакансии, с единицы")
    ap.add_argument("--model", default=gpt_service.DEFAULT_MODEL)
    ap.add_argument("--dry-run", action="store_true",
                    help="не обращаться к модели — посчитать только формулу")
    ap.add_argument("--no-history", action="store_true", help="не писать в history.json")
    args = ap.parse_args(argv)

    vacancies = json.loads(VACANCIES_FILE.read_text(encoding="utf-8"))
    vacancy = vacancies[args.vacancy - 1]
    vacancy_text = resume_parser.read_vacancy(vacancy["text"])

    files = sorted(RESUMES_DIR.glob("*.pdf"))
    if not files:
        raise SystemExit(f"Нет резюме в {RESUMES_DIR} — запустите make_resumes.py")

    print(f"Вакансия: {vacancy['title']}")
    print(f"Модель:   {'не используется (--dry-run)' if args.dry_run else args.model}")
    print(f"Резюме:   {len(files)}\n")

    rows = []
    for path in files:
        resume = resume_parser.extract_pdf(path)
        if args.dry_run:
            assessment, error = Assessment(), ""
        else:
            assessment, error = gpt_service.assess(vacancy_text, resume, path.stem,
                                                   model=args.model)
        if error:
            print(f"  {path.name}: модель не ответила — {error[:120]}")

        result = scoring.build_result(path.stem, vacancy["title"], vacancy_text, resume,
                                      assessment, scoring.DEFAULT_WEIGHTS,
                                      model="" if args.dry_run else args.model)
        if not args.no_history and not args.dry_run:
            scoring.save_result(result)
        rows.append(result)

    width = max(len(r.candidate) for r in rows) + 2
    print(f"{'Резюме'.ljust(width)}{'Итог':>6}{'Модель':>8}{'Формула':>9}"
          f"{'Опыт':>7}  Решение")
    print("-" * (width + 40))
    for r in sorted(rows, key=lambda x: -x.final_score):
        print(f"{r.candidate.ljust(width)}{r.final_score:>6}{r.gpt_score:>8}"
              f"{r.heuristic_score:>9}{r.breakdown.years:>6g}л  {r.verdict}")

    print()
    for r in sorted(rows, key=lambda x: -x.final_score):
        print(f"── {r.candidate} ({r.final_score}/100, {r.verdict})")
        if r.assessment.summary:
            print(f"   {r.assessment.summary}")
        if r.assessment.strengths:
            print(f"   сильные:   {'; '.join(r.assessment.strengths[:3])}")
        if r.assessment.weaknesses:
            print(f"   слабые:    {'; '.join(r.assessment.weaknesses[:3])}")
        if r.assessment.missing_skills:
            print(f"   не хватает: {', '.join(r.assessment.missing_skills[:6])}")
        print(f"   роль: вакансия «{r.breakdown.family_vacancy}», "
              f"резюме «{r.breakdown.family_resume}»")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
