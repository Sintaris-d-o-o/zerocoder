"""Тесты генератора видео без обращения к платному сервису (задание 10.3).

Генерация одного ролика стоит заметных денег, поэтому вся логика проверяется на
подменённых сетевых вызовах: сборка запроса, опрос статуса, скачивание, обработка ошибок.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import video_generator as vg  # noqa: E402
import test as check          # noqa: E402  — файл test.py из задания

MP4 = b"\x00\x00\x00 ftypisom" + b"\x00" * 64      # правдоподобное начало MP4-файла


class FakeApi:
    """Подменяет сеть RouterAI: создание задачи и последовательность статусов."""

    def __init__(self, statuses=("pending", "processing", "completed"), video_id="vid-1",
                 fail_create=None, urls=None, cost=43.78, credits_left=109.5):
        self.statuses = list(statuses)
        self.video_id = video_id
        self.fail_create = fail_create
        self.urls = urls if urls is not None else [f"https://routerai.ru/api/v1/videos/{video_id}/content?index=0"]
        self.cost = cost
        self.credits_left = credits_left
        self.created: list[dict] = []
        self.polls = 0

    def __call__(self, url, key, payload=None, method="GET", timeout=60):
        if url.endswith("/credits"):
            return {"data": {"credits": self.credits_left}}
        if method == "POST":
            if self.fail_create:
                raise self.fail_create
            self.created.append(payload)
            return {"id": self.video_id, "status": self.statuses[0],
                    "polling_url": f"{vg.ROUTERAI_BASE_URL}/videos/{self.video_id}"}
        idx = min(self.polls + 1, len(self.statuses) - 1)
        self.polls += 1
        status = self.statuses[idx]
        data = {"id": self.video_id, "status": status}
        if status in vg.DONE_STATUSES:
            data["unsigned_urls"] = self.urls
            data["usage"] = {"cost": self.cost}
        return data


def fake_opener(data=MP4):
    def opener(req, timeout=600):
        return SimpleNamespace(read=lambda: data)
    return opener


@pytest.fixture()
def creds(monkeypatch):
    monkeypatch.setenv("ROUTERAIRU_API_KEY", "sk-test")
    monkeypatch.delenv("PROXYAPI_KEY", raising=False)
    monkeypatch.delenv("VIDEO_PROVIDER", raising=False)
    monkeypatch.delenv("VIDEO_MODEL", raising=False)


def run(api, prompt="тестовый промпт", **kw):
    kw.setdefault("poll_interval_s", 0)
    return vg.generate_video(prompt, request=api, sleep=lambda s: None, **kw)


# --------------------------------------------------------------- выбор сервиса

def test_provider_follows_the_key_in_env(monkeypatch):
    monkeypatch.delenv("VIDEO_PROVIDER", raising=False)
    monkeypatch.setenv("ROUTERAIRU_API_KEY", "sk-1")
    monkeypatch.delenv("PROXYAPI_KEY", raising=False)
    assert vg.provider_name() == "routerai"

    monkeypatch.delenv("ROUTERAIRU_API_KEY")
    monkeypatch.setenv("PROXYAPI_KEY", "sk-2")
    assert vg.provider_name() == "proxyapi"          # остаётся способ из урока

    monkeypatch.setenv("VIDEO_PROVIDER", "routerai")  # ручной выбор сильнее
    assert vg.provider_name() == "routerai"


def test_missing_key_is_explained(monkeypatch):
    monkeypatch.delenv("ROUTERAIRU_API_KEY", raising=False)
    monkeypatch.delenv("VIDEO_PROVIDER", raising=False)
    monkeypatch.delenv("PROXYAPI_KEY", raising=False)
    with pytest.raises(vg.VideoError, match="ROUTERAIRU_API_KEY"):
        vg.api_key("routerai")


def test_model_can_be_overridden(monkeypatch, creds):
    assert vg.model_for("routerai") == vg.DEFAULT_MODEL
    monkeypatch.setenv("VIDEO_MODEL", "openai/sora-2-pro")
    assert vg.model_for("routerai") == "openai/sora-2-pro"


# --------------------------------------------------------------- генерация

def test_request_carries_task_parameters(creds):
    api = FakeApi()
    run(api, seconds=4, model="google/veo-3.1-fast", aspect_ratio="16:9", resolution="720p")
    sent = api.created[0]
    assert sent["duration"] == 4          # задание требует проверить 4 секунды
    assert sent["model"] == "google/veo-3.1-fast"
    assert sent["aspect_ratio"] == "16:9"
    assert sent["resolution"] == "720p"
    assert sent["prompt"] == "тестовый промпт"


def test_polls_until_completed(creds):
    api = FakeApi(statuses=("pending", "pending", "processing", "completed"))
    res = run(api)
    assert res.statuses[0] == "pending" and res.statuses[-1] == "completed"
    assert "processing" in res.statuses
    assert res.polls == 3
    assert res.cost == 43.78
    assert res.download_url.endswith("content?index=0")


def test_progress_callback_sees_every_status(creds):
    seen = []
    run(FakeApi(), on_progress=lambda s, e: seen.append(s))
    assert seen[0] == "pending" and seen[-1] == "completed"


@pytest.mark.parametrize("status", ["failed", "cancelled", "rejected"])
def test_failure_statuses_raise(creds, status):
    with pytest.raises(vg.VideoError, match=status):
        run(FakeApi(statuses=("pending", status)))


def test_unknown_status_is_not_swallowed(creds):
    with pytest.raises(vg.VideoError, match="Неизвестный статус"):
        run(FakeApi(statuses=("pending", "странное-слово")))


def test_timeout(creds, monkeypatch):
    ticks = iter(range(0, 100_000, 200))
    monkeypatch.setattr(vg.time, "perf_counter", lambda: next(ticks))
    with pytest.raises(vg.VideoError, match="не завершилась"):
        run(FakeApi(statuses=("pending",) * 50), timeout_s=900)


def test_missing_id(creds):
    def api(url, key, payload=None, method="GET", timeout=60):
        return {"status": "pending"}
    with pytest.raises(vg.VideoError, match="идентификатор"):
        run(api)


def test_done_without_url_is_an_error(creds):
    with pytest.raises(vg.VideoError, match="ссылки на файл нет"):
        run(FakeApi(statuses=("pending", "completed"), urls=[]))


@pytest.mark.parametrize("code,fragment", [
    (401, "ROUTERAIRU_API_KEY"), (402, "средств"), (404, "модель"),
    (422, "параметры"), (429, "подождать"),
])
def test_http_errors_are_explained(code, fragment):
    assert fragment in vg._explain(code, "{}")


# --------------------------------------------------------------- скачивание

def test_download_saves_mp4(creds, tmp_path):
    api = FakeApi()
    res = run(api)
    path = vg.download_video(res, tmp_path, "видео: тест/1", key="k", opener=fake_opener())
    assert path.suffix == ".mp4"
    for ch in ':<>"/\\|?*':
        assert ch not in path.name          # запрещённые в Windows символы вычищены
    assert path.read_bytes()[4:8] == b"ftyp"
    assert res.size_bytes > 0


def test_download_rejects_empty_file(creds, tmp_path):
    res = run(FakeApi())
    with pytest.raises(vg.VideoError, match="пустой"):
        vg.download_video(res, tmp_path, "x", key="k", opener=fake_opener(b""))


def test_download_rejects_non_mp4(creds, tmp_path):
    """Если вместо файла пришла страница с ошибкой, это должно быть видно сразу."""
    res = run(FakeApi())
    with pytest.raises(vg.VideoError, match="не похоже на MP4"):
        vg.download_video(res, tmp_path, "x", key="k", opener=fake_opener(b"<html>404</html>"))


def test_report_saved_next_to_video(creds, tmp_path):
    res = run(FakeApi(), prompt="промпт для отчёта", seconds=4)
    vg.download_video(res, tmp_path, "клип", key="k", opener=fake_opener())
    data = json.loads(vg.save_report(res, tmp_path).read_text(encoding="utf-8"))
    assert data["prompt"] == "промпт для отчёта"
    assert data["seconds"] == 4 and data["file"] == "клип.mp4"
    assert data["statuses"][-1] == "completed"
    assert data["provider"] == "routerai"
    assert data["cost_credits"] == 43.78


# --------------------------------------------------------------- деньги и промпты

def test_credits_are_read(creds):
    assert vg.credits(key="k", request=FakeApi()) == pytest.approx(109.5)


def test_credits_failure_does_not_stop_work(creds):
    """Остаток — справочная величина: если сервис не ответил, генерация всё равно идёт."""
    def boom(*a, **k):
        raise vg.VideoError("HTTP 500")
    assert vg.credits(key="k", request=boom) is None


def test_prompts_are_own_not_the_expert_example():
    assert len(vg.PROMPTS) >= 3
    for p in vg.PROMPTS:
        low = p.text.lower()
        assert "coffee" not in low and "кофе" not in low, f"{p.key}: это пример эксперта"
        assert len(p.text) > 120
    assert len({p.key for p in vg.PROMPTS}) == len(vg.PROMPTS)


def test_find_prompt():
    assert vg.find_prompt("certificates").key == "certificates"
    with pytest.raises(KeyError):
        vg.find_prompt("нет")


def test_progress_bar_reflects_stage():
    assert " 25%" in vg.render_bar("pending", 1)
    assert " 60%" in vg.render_bar("processing", 10)
    assert "100%" in vg.render_bar("completed", 20)
    mid = vg.render_bar("processing", 10)
    assert "█" in mid and "·" in mid


# --------------------------------------------------------------- test.py

def test_show_status_running(creds, capsys):
    api = FakeApi(statuses=("pending", "pending"))
    assert check.show_status("vid-1", request=api, key="k") == 0
    out = capsys.readouterr().out
    assert "vid-1" in out and "pending" in out and "--watch" in out


def test_show_status_done(creds, capsys):
    api = FakeApi(statuses=("completed", "completed"))
    assert check.show_status("vid-1", request=api, key="k") == 0
    out = capsys.readouterr().out
    assert "файл готов" in out and "--download" in out


def test_watch_downloads_when_ready(creds, tmp_path, capsys):
    api = FakeApi(statuses=("pending", "processing", "completed"))
    rc = check.watch("vid-1", tmp_path, request=api, key="k", interval=0,
                     sleep=lambda s: None, opener=fake_opener())
    assert rc == 0
    files = list(tmp_path.glob("*.mp4"))
    assert len(files) == 1 and files[0].stat().st_size > 0
    assert "Путь статусов" in capsys.readouterr().out


def test_watch_reports_failure(creds, tmp_path):
    api = FakeApi(statuses=("pending", "failed"))
    assert check.watch("vid-1", tmp_path, request=api, key="k", interval=0,
                       sleep=lambda s: None) == 1


def test_download_ready_video(creds, tmp_path, capsys):
    api = FakeApi(statuses=("completed", "completed"))
    assert check.download("vid-1", tmp_path, request=api, key="k", opener=fake_opener()) == 0
    assert list(tmp_path.glob("*.mp4"))
    assert "сохранено" in capsys.readouterr().out


def test_download_refuses_unfinished(creds, tmp_path, capsys):
    api = FakeApi(statuses=("pending", "pending"))
    assert check.download("vid-1", tmp_path, request=api, key="k") == 1
    assert "ещё не готово" in capsys.readouterr().err


# --------------------------------------------------------------- CLI

def test_cli_list(creds, capsys):
    assert vg.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "certificates" in out and "sora-2-pro" in out


def test_cli_dry_run_spends_nothing(creds, capsys):
    assert vg.main(["--dry-run", "--seconds", "4"]) == 0
    out = capsys.readouterr().out
    assert '"duration": 4' in out and "деньги не потрачены" in out
    assert vg.DEFAULT_MODEL in out


def test_cli_check_shows_credits(creds, monkeypatch, capsys):
    monkeypatch.setattr(vg, "credits", lambda key=None: 109.5)
    assert vg.main(["--check"]) == 0
    out = capsys.readouterr().out
    assert "109.5 кредитов" in out and "routerai" in out
