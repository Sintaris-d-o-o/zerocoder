"""
Тесты для общей таблицы заявок (storage.py) — без реальных обращений к
OpenAI/Telegram: проверяем, что строки корректно дописываются, ID растёт
по порядку и кириллица не портится при чтении обратно.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import storage  # noqa: E402


def _isolate_csv(tmp_path, monkeypatch):
    csv_path = tmp_path / "requests.csv"
    monkeypatch.setattr(storage, "CSV_PATH", csv_path)
    monkeypatch.setattr(storage, "LOCK_PATH", csv_path.with_suffix(".csv.lock"))
    return csv_path


def test_save_lead_creates_file_with_headers(tmp_path, monkeypatch):
    csv_path = _isolate_csv(tmp_path, monkeypatch)

    lead_id = storage.save_lead(
        storage.Lead(
            name="Иван Тестов",
            phone="+7 900 000-00-00",
            comment="Интересует CRM для учеников",
            consultation_type="Вопросы по CRM",
            source="Telegram-бот",
        )
    )

    assert lead_id == 1
    assert csv_path.exists()

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    assert rows[0] == storage.HEADERS
    assert rows[1][0] == "1"
    assert rows[1][2] == "Иван Тестов"
    assert rows[1][5] == "Telegram-бот"
    assert rows[1][6] == "Вопросы по CRM"


def test_save_lead_increments_id_across_calls(tmp_path, monkeypatch):
    _isolate_csv(tmp_path, monkeypatch)

    ids = [
        storage.save_lead(
            storage.Lead(
                name=f"Тестовый {i}",
                phone="+7 900 000-00-0" + str(i),
                comment="Тестовая заявка",
                consultation_type="Другое",
                source="Сайт",
            )
        )
        for i in range(1, 4)
    ]

    assert ids == [1, 2, 3]

    with open(storage.CSV_PATH, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    assert len(rows) == 4  # заголовок + 3 заявки
    assert [row[0] for row in rows[1:]] == ["1", "2", "3"]
