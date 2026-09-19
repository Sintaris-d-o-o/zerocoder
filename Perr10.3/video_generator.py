"""Генерация видео по текстовому описанию через Proxy API (задание 10.3).

Как в уроке: используется стандартная библиотека `openai`, но сервер указывается другой
(`base_url` сервиса-посредника). Код и логика от этого не меняются.

Генерация асинхронная, в три шага:
    1. создание задачи         → получаем идентификатор;
    2. опрос статуса в цикле   → queued → in_progress → completed;
    3. скачивание готового MP4.

Ключи только из .env:
    PROXYAPI_KEY      — ключ сервиса (показывается один раз при создании)
    PROXYAPI_BASE_URL — адрес API сервиса

Запуск:
    python video_generator.py                      # промпт проекта по умолчанию
    python video_generator.py --key certificates   # другой заготовленный промпт
    python video_generator.py --prompt "текст" --seconds 4
    python video_generator.py --list               # показать промпты, ничего не генерируя
    python video_generator.py --dry-run            # собрать запрос, не отправляя
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env")
load_dotenv(HERE / ".env")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_BASE_URL = "https://api.proxyapi.ru/openai/v1"
DEFAULT_MODEL = "sora-2"
DEFAULT_SECONDS = "4"        # задание требует проверить генерацию на 4 секундах
DEFAULT_SIZE = "1280x720"

POLL_INTERVAL_S = 5
POLL_TIMEOUT_S = 900         # генерация видео долгая: очередь плюс рендеринг

DONE_STATUSES = {"completed", "succeeded", "complete", "done"}
FAILED_STATUSES = {"failed", "error", "cancelled", "canceled"}


class VideoError(RuntimeError):
    """Ошибка сервиса, настроек доступа или самой генерации."""


@dataclass
class VideoResult:
    prompt: str
    video_id: str
    seconds: str
    size: str
    model: str
    elapsed_s: float
    polls: int
    statuses: list[str] = field(default_factory=list)
    saved_to: Optional[Path] = None
    size_bytes: int = 0

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1024 / 1024


# --------------------------------------------------------------------------- промпты

@dataclass
class Prompt:
    key: str
    title: str
    text: str


# Собственные промпты: задание прямо запрещает брать пример эксперта с чашкой кофе.
# Тема — наш продукт: управление сертификатами медицинских изделий.
PROMPTS: list[Prompt] = [
    Prompt(
        key="certificates",
        title="Документы на столе — порядок вместо хаоса",
        text=(
            "Close-up shot of a clean white desk in a bright modern office. A neat stack of "
            "official documents with embossed seals slowly slides into a perfectly aligned pile. "
            "Soft daylight from a window on the left, shallow depth of field, slow dolly-in "
            "camera movement. Calm corporate atmosphere, blue and white colour palette, "
            "no text, no logos, photorealistic."
        ),
    ),
    Prompt(
        key="lab",
        title="Лаборатория — контроль качества",
        text=(
            "A medical device laboratory, sterile and bright. A gloved hand places a small "
            "electronic device onto a testing stand; indicator lights turn from amber to green. "
            "Slow push-in camera, crisp reflections on stainless steel, cool white and teal "
            "lighting, shallow depth of field, photorealistic, no text."
        ),
    ),
    Prompt(
        key="dashboard",
        title="Панель контроля сроков",
        text=(
            "Abstract visualization of a compliance dashboard floating in dark space: "
            "rows of glowing cards drift slowly, their status indicators shifting from red "
            "to green one by one. Thin blue and teal light lines connect the cards. "
            "Smooth parallax camera drift, soft bloom, minimalist and futuristic, "
            "no readable text, no letters."
        ),
    ),
]


def find_prompt(key: str) -> Prompt:
    for p in PROMPTS:
        if p.key == key:
            return p
    raise KeyError(f"Нет промпта с ключом {key!r}. Доступны: {', '.join(p.key for p in PROMPTS)}")


# --------------------------------------------------------------------------- клиент

def make_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
    """Стандартный клиент OpenAI, направленный на сервер посредника."""
    api_key = api_key or os.getenv("PROXYAPI_KEY", "")
    base_url = base_url or os.getenv("PROXYAPI_BASE_URL", DEFAULT_BASE_URL)
    if not api_key:
        raise VideoError("Не задан PROXYAPI_KEY. Положите ключ в .env — см. .env.example")
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=base_url, timeout=120.0, max_retries=2)


def _status_of(job) -> str:
    return str(getattr(job, "status", "") or "").lower()


def _progress_of(job) -> Optional[int]:
    value = getattr(job, "progress", None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def render_bar(status: str, progress: Optional[int], elapsed: float, width: int = 28) -> str:
    """Строка прогресса для терминала: задание просит скриншот с прогресс-баром."""
    pct = progress if progress is not None else (100 if status in DONE_STATUSES else 0)
    filled = max(0, min(width, round(width * pct / 100)))
    bar = "█" * filled + "·" * (width - filled)
    return f"  [{bar}] {pct:3d}%  {status:<12} {elapsed:5.0f} с"


# --------------------------------------------------------------------------- генерация

def generate_video(prompt: str, *, seconds: str = DEFAULT_SECONDS, size: str = DEFAULT_SIZE,
                   model: str = DEFAULT_MODEL, client=None,
                   poll_interval_s: float = POLL_INTERVAL_S, timeout_s: float = POLL_TIMEOUT_S,
                   on_progress: Optional[Callable[[str, Optional[int], float], None]] = None,
                   sleep: Callable[[float], None] = time.sleep) -> VideoResult:
    """Создаёт задачу генерации и ждёт готовности.

    `client` и `sleep` вынесены в параметры — так тесты подменяют сеть и проверяют логику
    без обращения к платному сервису.
    """
    client = client or make_client()
    started = time.perf_counter()

    try:
        job = client.videos.create(model=model, prompt=prompt, seconds=seconds, size=size)
    except Exception as exc:  # noqa: BLE001 — превращаем любую ошибку SDK в понятный текст
        raise VideoError(_explain(exc)) from exc

    video_id = getattr(job, "id", None)
    if not video_id:
        raise VideoError(f"Сервис не вернул идентификатор задачи. Ответ: {job!r}")

    statuses = [_status_of(job) or "queued"]
    if on_progress:
        on_progress(statuses[-1], _progress_of(job), 0.0)

    polls = 0
    while True:
        elapsed = time.perf_counter() - started
        if elapsed > timeout_s:
            raise VideoError(
                f"Генерация не завершилась за {timeout_s:.0f} с (задача {video_id}, "
                f"последний статус «{statuses[-1]}»). Задача могла остаться в очереди")
        sleep(poll_interval_s)
        polls += 1
        try:
            job = client.videos.retrieve(video_id)
        except Exception as exc:  # noqa: BLE001
            raise VideoError(_explain(exc)) from exc

        status = _status_of(job)
        if status and (not statuses or status != statuses[-1]):
            statuses.append(status)
        if on_progress:
            on_progress(status, _progress_of(job), time.perf_counter() - started)

        if status in DONE_STATUSES:
            return VideoResult(prompt=prompt, video_id=video_id, seconds=seconds, size=size,
                               model=model, elapsed_s=round(time.perf_counter() - started, 1),
                               polls=polls, statuses=statuses)
        if status in FAILED_STATUSES:
            detail = getattr(job, "error", None)
            raise VideoError(f"Сервис сообщил о неудаче: статус «{status}»"
                             + (f", подробности: {detail}" if detail else ""))


def download_video(result: VideoResult, directory: Path, name: str, *, client=None) -> Path:
    """Скачивает готовый MP4 и сохраняет на диск."""
    client = client or make_client()
    directory.mkdir(parents=True, exist_ok=True)
    safe = "".join("_" if ch in ':<>"/\\|?*' else ch for ch in name).strip() or "video"
    path = directory / f"{safe}.mp4"
    try:
        content = client.videos.download_content(result.video_id)
        data = content.read() if hasattr(content, "read") else bytes(content)
    except Exception as exc:  # noqa: BLE001
        raise VideoError(f"Не удалось скачать готовое видео: {_explain(exc)}") from exc
    if not data:
        raise VideoError("Сервис вернул пустой файл")
    path.write_bytes(data)
    result.saved_to = path
    result.size_bytes = len(data)
    return path


def _explain(exc: Exception) -> str:
    """Понятное объяснение вместо технической ошибки SDK."""
    text = str(exc)
    code = getattr(exc, "status_code", None)
    hints = {
        401: "ключ не принят — проверьте PROXYAPI_KEY",
        402: "недостаточно средств на балансе сервиса",
        403: "доступ к модели закрыт для этого ключа",
        404: "адрес не найден — проверьте PROXYAPI_BASE_URL и название модели",
        429: "слишком много запросов, сервис просит подождать",
    }
    if code in hints:
        return f"HTTP {code}: {hints[code]}. Ответ сервиса: {text[:300]}"
    return f"{type(exc).__name__}: {text[:400]}"


def save_report(result: VideoResult, directory: Path) -> Path:
    """Сохраняет параметры прогона рядом с видео — чтобы результат можно было повторить."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{(result.saved_to.stem if result.saved_to else result.video_id)}.json"
    path.write_text(json.dumps({
        "prompt": result.prompt, "video_id": result.video_id, "model": result.model,
        "seconds": result.seconds, "size": result.size, "elapsed_s": result.elapsed_s,
        "polls": result.polls, "statuses": result.statuses,
        "file": result.saved_to.name if result.saved_to else None,
        "size_mb": round(result.size_mb, 2),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- CLI

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", help="свой текст промпта")
    ap.add_argument("--key", default="certificates", help="ключ заготовленного промпта")
    ap.add_argument("--seconds", default=DEFAULT_SECONDS, help="длительность видео в секундах")
    ap.add_argument("--size", default=DEFAULT_SIZE, help="разрешение, например 1280x720")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="results", help="папка для видео")
    ap.add_argument("--list", action="store_true", help="показать промпты и выйти")
    ap.add_argument("--dry-run", action="store_true", help="показать запрос, не отправляя его")
    args = ap.parse_args(argv)

    if args.list:
        print(f"Заготовленные промпты ({len(PROMPTS)}):\n")
        for p in PROMPTS:
            print(f"  {p.key:15s} {p.title}")
            print(f"  {'':15s} {p.text[:110]}…\n")
        return 0

    prompt = args.prompt or find_prompt(args.key).text

    if args.dry_run:
        print("Запрос, который будет отправлен:")
        print(json.dumps({"model": args.model, "prompt": prompt,
                          "seconds": args.seconds, "size": args.size},
                         ensure_ascii=False, indent=2))
        key = os.getenv("PROXYAPI_KEY")
        print(f"\nАдрес сервиса: {os.getenv('PROXYAPI_BASE_URL', DEFAULT_BASE_URL)}")
        print(f"Ключ: {'задан, ' + str(len(key)) + ' символов' if key else 'НЕ ЗАДАН'}")
        print("--dry-run: запрос не отправлялся, деньги не потрачены.")
        return 0

    print(f"Промпт: {prompt}\n")
    print(f"Модель {args.model}, длительность {args.seconds} с, размер {args.size}\n")

    last_line = {"n": 0}

    def progress(status: str, pct: Optional[int], elapsed: float) -> None:
        line = render_bar(status, pct, elapsed)
        print("\r" + line, end="", flush=True)
        last_line["n"] += 1

    try:
        result = generate_video(prompt, seconds=args.seconds, size=args.size, model=args.model,
                                on_progress=progress)
    except VideoError as exc:
        print()
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 1

    print()
    print(f"\nГотово за {result.elapsed_s} с, проверок статуса: {result.polls}")
    print(f"Путь статусов: {' → '.join(result.statuses)}")

    out_dir = HERE / args.out
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    try:
        path = download_video(result, out_dir, f"{stamp}_{args.key}")
    except VideoError as exc:
        print(f"ОШИБКА при скачивании: {exc}", file=sys.stderr)
        print(f"Видео осталось на сервисе, его идентификатор: {result.video_id}", file=sys.stderr)
        return 1

    report = save_report(result, out_dir)
    print(f"Видео сохранено: {path.relative_to(HERE)} ({result.size_mb:.1f} МБ)")
    print(f"Параметры прогона: {report.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
