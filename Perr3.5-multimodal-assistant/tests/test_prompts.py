"""
Тесты без реальных обращений к OpenAI/Telegram: проверяем, что промпты
формируются правильно и что состояние "ожидания деталей после фото" работает
как задумано.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bot  # noqa: E402


def test_selling_text_prompt_includes_facts_and_details():
    prompt = bot.PROMPT_SELLING_TEXT.format(
        facts="- Свитер, шерсть, синий, б/у без дефектов",
        details="Размер M, носили пару раз, цена 1500",
    )
    assert "Свитер, шерсть, синий" in prompt
    assert "Размер M, носили пару раз, цена 1500" in prompt


def test_final_report_prompt_includes_selling_text_and_facts():
    prompt = bot.PROMPT_FINAL_REPORT.format(
        selling_text="Продающий текст про свитер",
        facts="- Свитер, шерсть, синий",
    )
    assert "Продающий текст про свитер" in prompt
    assert "Свитер, шерсть, синий" in prompt


def test_pending_photos_flow():
    bot.pending_photos.clear()
    user_id = 42
    assert user_id not in bot.pending_photos

    bot.pending_photos[user_id] = b"fake-jpeg-bytes"
    assert user_id in bot.pending_photos

    photo_bytes = bot.pending_photos.pop(user_id)
    assert photo_bytes == b"fake-jpeg-bytes"
    assert user_id not in bot.pending_photos
