"""
Общее хранилище заявок — единая таблица requests.csv, в которую пишут и
Telegram-бот (bot.py), и сайт-версия ассистента (website/website_app.py).

Файл создаётся автоматически при первой заявке рядом с этим модулем, с
заголовками-колонками. Запись под файловой блокировкой (filelock) — чтобы
Telegram-бот и сайт, работая как два отдельных процесса, не повредили файл,
если запишут заявку одновременно.
"""

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from filelock import FileLock

CSV_PATH = Path(__file__).resolve().parent / "requests.csv"
LOCK_PATH = CSV_PATH.with_suffix(".csv.lock")

HEADERS = [
    "ID",
    "Дата/время",
    "Имя",
    "Телефон",
    "Комментарий",
    "Источник обращения",
    "Тип консультации",
]


@dataclass
class Lead:
    name: str
    phone: str
    comment: str
    consultation_type: str
    source: str


def save_lead(lead: Lead) -> int:
    """Дописывает заявку в общую таблицу и возвращает присвоенный ей ID."""
    with FileLock(str(LOCK_PATH)):
        is_new = not CSV_PATH.exists()
        next_id = 1 if is_new else _next_id()
        row = [
            next_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            lead.name,
            lead.phone,
            lead.comment,
            lead.source,
            lead.consultation_type,
        ]
        if is_new:
            # encoding="utf-8-sig" — только при создании файла, ровно один раз:
            # так в начале файла появляется BOM, и Excel сразу видит кириллицу
            # правильно. Повторное открытие с utf-8-sig при каждом дозаписывании
            # добавляло бы новый BOM в середину файла — поэтому дальше "a"+utf-8.
            with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(HEADERS)
                writer.writerow(row)
        else:
            with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
    return next_id


def _next_id() -> int:
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        rows = [row for row in csv.reader(f) if row]
    data_rows = rows[1:]
    if not data_rows:
        return 1
    return int(data_rows[-1][0]) + 1
