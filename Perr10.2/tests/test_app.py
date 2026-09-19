"""Тесты веб-приложения без обращения к платному API (задание 10.2)."""
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "Perr10.1"))

import app as webapp  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(webapp, "RESULTS_DIR", tmp_path / "logos")
    monkeypatch.setattr(webapp, "GALLERY_FILE", tmp_path / "gallery.json")
    application = webapp.create_app(demo=True)      # демо-режим: сеть не трогаем
    application.config.update(TESTING=True)
    with application.test_client() as c:
        yield c


def wait_for(client, job_id, timeout=20):
    """Ждёт завершения фоновой задачи генерации."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/status/{job_id}").get_json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.2)
    raise AssertionError("задача не завершилась вовремя")


# --------------------------------------------------------------- сборка промпта

def test_prompt_combines_company_style_and_extra():
    p = webapp.build_prompt("SINTARIS", "medtech", "щит с галочкой", "logo")
    assert "«SINTARIS»" in p
    assert "медицинская тематика" in p                  # текст стиля подмешан
    assert "щит с галочкой" in p                        # свободное описание пользователя
    assert webapp.NO_TEXT_TAIL in p                     # запрет надписей на знаке


def test_prompt_changes_with_format():
    logo = webapp.build_prompt("Firma", "minimal", "", "logo")
    banner = webapp.build_prompt("Firma", "minimal", "", "banner")
    story = webapp.build_prompt("Firma", "minimal", "", "story")
    assert logo.startswith("Логотип")
    assert "баннер" in banner.lower()
    assert "вертикальная" in story.lower()


def test_prompt_survives_empty_extra_and_unknown_style():
    p = webapp.build_prompt("Firma", "нет-такого-стиля", "", "logo")
    assert p.count("..") == 0        # не должно быть двойных точек от пустых кусков
    assert "Firma" in p


def test_every_style_and_format_produces_a_prompt():
    for style in webapp.STYLES:
        for fmt in webapp.FORMATS:
            p = webapp.build_prompt("X", style, "", fmt)
            assert len(p) > 60 and "«X»" in p


# --------------------------------------------------------------- маршруты

def test_index_lists_styles_and_formats(client):
    html = client.get("/").get_data(as_text=True)
    assert "Генератор логотипов" in html
    for s in webapp.STYLES.values():
        assert s["title"] in html
    assert "Режим демонстрации" in html          # в демо показываем предупреждение


def test_health(client):
    data = client.get("/health").get_json()
    assert data["demo"] is True
    assert data["styles"] == len(webapp.STYLES)
    assert data["formats"] == len(webapp.FORMATS)


@pytest.mark.parametrize("payload,fragment", [
    ({}, "название"),
    ({"company": "   "}, "название"),
    ({"company": "X" * 81}, "длинное"),
    ({"company": "ok", "seed": "не число"}, "целым числом"),
])
def test_validation_errors(client, payload, fragment):
    r = client.post("/generate", json=payload)
    assert r.status_code == 400
    assert fragment in r.get_json()["error"].lower()


def test_generate_and_download_roundtrip(client):
    r = client.post("/generate", json={"company": "SINTARIS", "style": "medtech",
                                       "format": "logo", "extra": "щит с галочкой"})
    assert r.status_code == 200
    started = r.get_json()
    assert "щит с галочкой" in started["prompt"]

    job = wait_for(client, started["job_id"])
    assert job["status"] == "done", job.get("message")
    assert job["filename"].endswith(".jpeg")

    img = client.get(f"/image/{job['filename']}")
    assert img.status_code == 200 and img.data[:3] == b"\xff\xd8\xff"   # это JPEG

    dl = client.get(f"/download/{job['filename']}")
    assert dl.status_code == 200
    assert "attachment" in dl.headers.get("Content-Disposition", "")


def test_seed_is_preserved_for_repeat(client):
    r = client.post("/generate", json={"company": "Firma", "seed": "12345"})
    job = wait_for(client, r.get_json()["job_id"])
    assert job["seed"] == 12345          # то же зерно вернулось, генерацию можно повторить


def test_gallery_records_result(client):
    r = client.post("/generate", json={"company": "Firma"})
    wait_for(client, r.get_json()["job_id"])
    items = webapp.load_gallery()
    assert len(items) == 1
    assert items[0]["company"] == "Firma" and items[0]["status"] == "done"
    assert "Firma" in client.get("/").get_data(as_text=True)


def test_filename_has_no_forbidden_characters(client):
    # двоеточие и слеш на Windows увели бы запись не в тот файл (урок задания 8.2)
    r = client.post("/generate", json={"company": 'ООО "Альфа/Бета: тест"'})
    job = wait_for(client, r.get_json()["job_id"])
    assert job["status"] == "done"
    for ch in ':<>"/\\|?*':
        assert ch not in job["filename"]


def test_unknown_job_returns_404(client):
    assert client.get("/status/нет-такой").status_code == 404


def test_error_from_api_becomes_message(client, monkeypatch):
    import yandex_art as ya

    def boom(*a, **k):
        raise ya.YandexArtError("HTTP 403: доступ запрещён")

    monkeypatch.setattr(ya, "generate_image", boom)
    application = webapp.create_app(demo=False)       # не демо: вызовет подменённую функцию
    application.config.update(TESTING=True)
    with application.test_client() as c:
        job_id = c.post("/generate", json={"company": "Firma"}).get_json()["job_id"]
        job = wait_for(c, job_id)
        assert job["status"] == "error"
        assert "403" in job["message"]
