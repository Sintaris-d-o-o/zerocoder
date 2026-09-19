"""Тесты генератора изображений без обращения к платному API.

Сетевые вызовы подменяются: проверяется сборка запроса, опрос операции, декодирование
картинки, обработка ошибок и очистка имени файла.
"""
import base64
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yandex_art as ya  # noqa: E402

PIXEL = base64.b64encode(b"\xff\xd8\xff\xe0 fake jpeg bytes").decode()


class FakeApi:
    """Заменяет сеть: отдаёт id операции, затем N раз «не готово», затем результат."""

    def __init__(self, not_ready: int = 2, image_b64: str = PIXEL, error: dict | None = None,
                 operation_id: str = "op-123"):
        self.not_ready = not_ready
        self.image_b64 = image_b64
        self.error = error
        self.operation_id = operation_id
        self.posts: list[tuple] = []
        self.gets: list[str] = []
        self.slept: list[float] = []

    def post(self, url, payload, headers, timeout=30):
        self.posts.append((url, payload, headers))
        return {"id": self.operation_id}

    def get(self, url, headers, timeout=30):
        self.gets.append(url)
        if len(self.gets) <= self.not_ready:
            return {"done": False}
        if self.error:
            return {"done": True, "error": self.error}
        return {"done": True, "response": {"image": self.image_b64}}

    def sleep(self, seconds):
        self.slept.append(seconds)


@pytest.fixture()
def creds(monkeypatch):
    monkeypatch.setenv("YANDEX_API_KEY", "test-key")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "test-folder")


def run(api: FakeApi, prompt="логотип", seed=None, **kw):
    return ya.generate_image(prompt, seed, post_json=api.post, get_json=api.get,
                             sleep=api.sleep, **kw)


def test_payload_matches_lesson_structure():
    p = ya.build_payload("логотип щита", seed=42, folder_id="b1gtest")
    assert p["modelUri"] == "art://b1gtest/yandex-art/latest"
    assert p["generationOptions"]["seed"] == 42
    assert p["generationOptions"]["aspectRatio"] == {"widthRatio": "1", "heightRatio": "1"}
    assert p["messages"] == [{"weight": "1", "text": "логотип щита"}]


def test_payload_respects_aspect_ratio():
    p = ya.build_payload("баннер", None, "f", width_ratio="16", height_ratio="9")
    assert p["generationOptions"]["aspectRatio"] == {"widthRatio": "16", "heightRatio": "9"}
    assert p["generationOptions"]["seed"] is None   # None = каждый раз новое изображение


def test_headers_use_api_key_prefix():
    h = ya.build_headers("abc123")
    assert h["Authorization"] == "Api-Key abc123"   # у Яндекса не Bearer
    assert h["Content-Type"] == "application/json"


def test_generation_polls_until_done(creds):
    api = FakeApi(not_ready=2)
    res = run(api, poll_interval_s=0)
    assert res.image_bytes.startswith(b"\xff\xd8\xff")   # раскодировался JPEG
    assert res.polls == 3 and len(api.gets) == 3          # два «не готово» и один результат
    assert res.operation_id == "op-123"
    assert api.posts[0][0] == ya.GENERATE_URL
    assert "op-123" in api.gets[0]


def test_seed_goes_into_request(creds):
    api = FakeApi(not_ready=0)
    run(api, seed=777, poll_interval_s=0)
    assert api.posts[0][1]["generationOptions"]["seed"] == 777


@pytest.mark.parametrize("missing", ["YANDEX_API_KEY", "YANDEX_FOLDER_ID"])
def test_missing_credentials_explained(monkeypatch, creds, missing):
    monkeypatch.delenv(missing, raising=False)
    monkeypatch.delenv("YANDEX_CLOUD_ID", raising=False)
    with pytest.raises(ya.YandexArtError, match=missing):
        run(FakeApi(), poll_interval_s=0)


def test_api_error_is_reported(creds):
    api = FakeApi(not_ready=0, error={"message": "quota exceeded"})
    with pytest.raises(ya.YandexArtError, match="quota exceeded"):
        run(api, poll_interval_s=0)


def test_missing_image_in_response(creds):
    api = FakeApi(not_ready=0, image_b64="")
    with pytest.raises(ya.YandexArtError, match="нет изображения"):
        run(api, poll_interval_s=0)


def test_broken_base64_is_reported(creds):
    api = FakeApi(not_ready=0, image_b64="это не base64!!!")
    with pytest.raises(ya.YandexArtError, match="раскодировать"):
        run(api, poll_interval_s=0)


def test_timeout_when_never_done(creds, monkeypatch):
    api = FakeApi(not_ready=10_000)
    ticks = iter(range(0, 10_000, 100))          # каждый вызов времени прыгает на 100 секунд
    monkeypatch.setattr(ya.time, "perf_counter", lambda: next(ticks))
    with pytest.raises(ya.YandexArtError, match="не завершилась"):
        run(api, poll_interval_s=0, timeout_s=300)


def test_no_operation_id(creds):
    api = FakeApi()
    api.post = lambda *a, **k: {}                # сервис ответил без id
    with pytest.raises(ya.YandexArtError, match="идентификатор операции"):
        run(api, poll_interval_s=0)


@pytest.mark.parametrize("code,fragment", [
    (401, "Api-Key"), (403, "ai.imageGeneration.user"), (404, "FOLDER_ID"), (429, "подождать"),
])
def test_http_errors_are_explained(code, fragment):
    assert fragment in ya._explain_http_error(code, "{}")


def test_save_image_strips_forbidden_characters(creds, tmp_path):
    api = FakeApi(not_ready=0)
    res = run(api, poll_interval_s=0)
    # двоеточие на Windows увело бы запись в альтернативный поток данных (урок задания 8.2)
    path = ya.save_image(res, tmp_path, 'лого: версия 1/2')
    assert ":" not in path.name and "/" not in path.name
    assert path.exists() and path.read_bytes() == res.image_bytes
    assert res.saved_to == path


def test_project_prompts_are_own_and_valid():
    assert len(ya.PROMPTS) >= 3
    for p in ya.PROMPTS:
        assert len(p.text) > 100, f"{p.key}: промпт слишком короткий"
        # задание требует собственный промпт, а не пример эксперта
        assert "код на мопеде" not in p.text.lower()
        assert "без текста" in p.text.lower(), f"{p.key}: для логотипа важно запретить надписи"
    assert len({p.key for p in ya.PROMPTS}) == len(ya.PROMPTS), "ключи промптов должны быть уникальны"


def test_find_prompt():
    assert ya.find_prompt("shield-check").key == "shield-check"
    with pytest.raises(KeyError):
        ya.find_prompt("нет-такого")


def test_cli_list_and_dry_run_do_not_call_api(creds, capsys):
    assert ya.main(["--list"]) == 0
    assert "shield-check" in capsys.readouterr().out
    assert ya.main(["--dry-run", "--key", "tech-pulse"]) == 0
    out = capsys.readouterr().out
    assert "art://test-folder/yandex-art/latest" in out
    assert "деньги не потрачены" in out


def test_cli_rejects_bad_ratio(creds, capsys):
    assert ya.main(["--dry-run", "--ratio", "квадрат"]) == 2
