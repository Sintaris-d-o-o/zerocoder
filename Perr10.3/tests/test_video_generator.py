"""Тесты генератора видео без обращения к платному сервису (задание 10.3)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import video_generator as vg  # noqa: E402
import test as check          # noqa: E402  — файл test.py из задания


class FakeVideos:
    """Подменяет client.videos: выдаёт заданную последовательность статусов."""

    def __init__(self, statuses=("queued", "in_progress", "completed"), content=b"\x00\x00\x00\x18ftypmp42",
                 fail_on_create=None, video_id="vid_abc123"):
        self.statuses = list(statuses)
        self.content = content
        self.fail_on_create = fail_on_create
        self.video_id = video_id
        self.created: list[dict] = []
        self.retrieves = 0
        self.downloads = 0

    def create(self, **kwargs):
        if self.fail_on_create:
            raise self.fail_on_create
        self.created.append(kwargs)
        return SimpleNamespace(id=self.video_id, status=self.statuses[0], progress=0)

    def retrieve(self, video_id):
        self.retrieves += 1
        idx = min(self.retrieves, len(self.statuses) - 1)
        status = self.statuses[idx]
        pct = 100 if status in vg.DONE_STATUSES else idx * 40
        return SimpleNamespace(id=video_id, status=status, progress=pct,
                               model="sora-2", seconds="4", size="1280x720")

    def download_content(self, video_id):
        self.downloads += 1
        return SimpleNamespace(read=lambda: self.content)

    def list(self):
        return SimpleNamespace(data=[SimpleNamespace(id=self.video_id, status="completed", model="sora-2")])


def fake_client(**kw):
    return SimpleNamespace(videos=FakeVideos(**kw))


def run(client, prompt="test prompt", **kw):
    kw.setdefault("poll_interval_s", 0)
    return vg.generate_video(prompt, client=client, sleep=lambda s: None, **kw)


# --------------------------------------------------------------- генерация

def test_create_receives_lesson_parameters():
    c = fake_client()
    run(c, "мой промпт", seconds="4", size="1280x720")
    sent = c.videos.created[0]
    assert sent["prompt"] == "мой промпт"
    assert sent["seconds"] == "4"          # задание требует проверить 4 секунды
    assert sent["size"] == "1280x720"
    assert sent["model"] == vg.DEFAULT_MODEL


def test_polls_until_completed():
    c = fake_client(statuses=("queued", "queued", "in_progress", "completed"))
    res = run(c)
    assert res.video_id == "vid_abc123"
    assert res.statuses[0] == "queued" and res.statuses[-1] == "completed"
    assert "in_progress" in res.statuses
    assert res.polls == 3


def test_progress_callback_receives_updates():
    seen = []
    c = fake_client(statuses=("queued", "in_progress", "completed"))
    run(c, on_progress=lambda s, p, e: seen.append((s, p)))
    assert seen[0][0] == "queued"
    assert seen[-1][0] == "completed"
    assert seen[-1][1] == 100


@pytest.mark.parametrize("status", ["failed", "cancelled", "error"])
def test_failure_statuses_raise(status):
    c = fake_client(statuses=("queued", status))
    with pytest.raises(vg.VideoError, match=status):
        run(c)


def test_timeout(monkeypatch):
    c = fake_client(statuses=("queued",) * 50)
    ticks = iter(range(0, 100_000, 200))
    monkeypatch.setattr(vg.time, "perf_counter", lambda: next(ticks))
    with pytest.raises(vg.VideoError, match="не завершилась"):
        run(c, timeout_s=900)


def test_missing_id():
    c = fake_client()
    c.videos.create = lambda **kw: SimpleNamespace(status="queued")
    with pytest.raises(vg.VideoError, match="идентификатор"):
        run(c)


def test_create_error_is_explained():
    err = RuntimeError("boom")
    err.status_code = 402
    c = fake_client(fail_on_create=err)
    with pytest.raises(vg.VideoError, match="баланс"):
        run(c)


@pytest.mark.parametrize("code,fragment", [
    (401, "PROXYAPI_KEY"), (402, "баланс"), (404, "BASE_URL"), (429, "подождать"),
])
def test_error_hints(code, fragment):
    exc = RuntimeError("x")
    exc.status_code = code
    assert fragment in vg._explain(exc)


# --------------------------------------------------------------- скачивание

def test_download_saves_mp4(tmp_path):
    c = fake_client()
    res = run(c)
    path = vg.download_video(res, tmp_path, "видео: тест/1", client=c)
    assert path.suffix == ".mp4"
    for ch in ':<>"/\\|?*':
        assert ch not in path.name        # запрещённые в Windows символы вычищены
    assert path.read_bytes().startswith(b"\x00\x00\x00\x18ftyp")   # сигнатура MP4
    assert res.size_bytes > 0


def test_download_rejects_empty_file(tmp_path):
    c = fake_client(content=b"")
    res = run(c)
    with pytest.raises(vg.VideoError, match="пустой"):
        vg.download_video(res, tmp_path, "x", client=c)


def test_report_saved_next_to_video(tmp_path):
    c = fake_client()
    res = run(c, "промпт для отчёта")
    vg.download_video(res, tmp_path, "клип", client=c)
    report = vg.save_report(res, tmp_path)
    import json
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["prompt"] == "промпт для отчёта"
    assert data["seconds"] == "4" and data["file"] == "клип.mp4"
    assert data["statuses"][-1] == "completed"


# --------------------------------------------------------------- промпты и прогресс-бар

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


def test_progress_bar_rendering():
    assert "  0%" in vg.render_bar("queued", 0, 1)
    assert "100%" in vg.render_bar("completed", None, 40)     # готово = полная полоса
    mid = vg.render_bar("in_progress", 50, 20)
    assert "█" in mid and "·" in mid and "in_progress" in mid


# --------------------------------------------------------------- test.py

def test_check_access_without_key(monkeypatch, capsys):
    monkeypatch.delenv("PROXYAPI_KEY", raising=False)
    assert check.check_access() == 2
    assert "НЕ ЗАДАН" in capsys.readouterr().out


def test_check_access_with_client(monkeypatch, capsys):
    monkeypatch.setenv("PROXYAPI_KEY", "sk-test")
    assert check.check_access(client=fake_client()) == 0
    assert "сервис доступен" in capsys.readouterr().out


def test_show_status(capsys):
    c = fake_client(statuses=("completed", "completed"))
    assert check.show_status("vid_abc123", client=c) == 0
    out = capsys.readouterr().out
    assert "vid_abc123" in out and "completed" in out


def test_watch_downloads_when_ready(tmp_path, capsys):
    c = fake_client(statuses=("queued", "in_progress", "completed"))
    rc = check.watch("vid_abc123", tmp_path, client=c, interval=0, sleep=lambda s: None)
    assert rc == 0
    files = list(tmp_path.glob("*.mp4"))
    assert len(files) == 1 and files[0].stat().st_size > 0
    assert "Путь статусов" in capsys.readouterr().out


def test_watch_reports_failure(tmp_path, capsys):
    c = fake_client(statuses=("queued", "failed"))
    assert check.watch("vid", tmp_path, client=c, interval=0, sleep=lambda s: None) == 1


def test_list_jobs(capsys):
    assert check.list_jobs(client=fake_client()) == 0
    assert "vid_abc123" in capsys.readouterr().out


# --------------------------------------------------------------- CLI

def test_cli_list_and_dry_run(capsys, monkeypatch):
    monkeypatch.setenv("PROXYAPI_KEY", "sk-test")
    assert vg.main(["--list"]) == 0
    assert "certificates" in capsys.readouterr().out
    assert vg.main(["--dry-run", "--seconds", "4"]) == 0
    out = capsys.readouterr().out
    assert '"seconds": "4"' in out and "деньги не потрачены" in out
