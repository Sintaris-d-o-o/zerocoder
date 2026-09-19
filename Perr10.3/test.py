"""Проверка статуса генерации видео — файл `test.py` из задания 10.3.

Задание просит «проверить статус через test.py». Скрипт делает три вещи:

    python test.py                 — проверяет доступ: виден ли ключ, отвечает ли сервис;
    python test.py <id видео>      — показывает статус конкретной задачи;
    python test.py --watch <id>    — следит за задачей до завершения и скачивает результат;
    python test.py --list          — показывает последние задачи сервиса.

Отдельный файл нужен потому, что генерация долгая: запустили в одном окне, а проверять
статус и забирать готовый файл удобно отдельной командой, не перезапуская генерацию.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import video_generator as vg  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def check_access(client=None) -> int:
    """Проверяет, что ключ на месте и сервис отвечает. Денег не тратит."""
    import os
    key = os.getenv("PROXYAPI_KEY")
    base = os.getenv("PROXYAPI_BASE_URL", vg.DEFAULT_BASE_URL)
    print("Проверка доступа к сервису")
    print(f"  адрес:  {base}")
    print(f"  ключ:   {'задан, ' + str(len(key)) + ' символов' if key else 'НЕ ЗАДАН — добавьте PROXYAPI_KEY в .env'}")
    if not key:
        return 2
    try:
        client = client or vg.make_client()
        jobs = client.videos.list()
        items = list(getattr(jobs, "data", []) or [])
        print(f"  ответ:  сервис доступен, задач в истории: {len(items)}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"  ответ:  {vg._explain(exc)}", file=sys.stderr)
        return 1


def show_status(video_id: str, client=None) -> int:
    client = client or vg.make_client()
    try:
        job = client.videos.retrieve(video_id)
    except Exception as exc:  # noqa: BLE001
        print(f"ОШИБКА: {vg._explain(exc)}", file=sys.stderr)
        return 1
    status = vg._status_of(job)
    pct = vg._progress_of(job)
    print(f"Задача {video_id}")
    print(f"  статус:   {status}")
    if pct is not None:
        print(f"  прогресс: {pct} %")
    for attr in ("model", "seconds", "size", "created_at"):
        value = getattr(job, attr, None)
        if value is not None:
            print(f"  {attr}: {value}")
    if status in vg.DONE_STATUSES:
        print("  видео готово — скачать: python test.py --watch " + video_id)
    return 0


def watch(video_id: str, out_dir: Path, client=None, interval: float = vg.POLL_INTERVAL_S,
          timeout: float = vg.POLL_TIMEOUT_S, sleep=time.sleep) -> int:
    """Следит за задачей до завершения и скачивает готовый файл."""
    client = client or vg.make_client()
    started = time.perf_counter()
    statuses: list[str] = []
    while True:
        elapsed = time.perf_counter() - started
        if elapsed > timeout:
            print(f"\nЗадача не завершилась за {timeout:.0f} с", file=sys.stderr)
            return 1
        try:
            job = client.videos.retrieve(video_id)
        except Exception as exc:  # noqa: BLE001
            print(f"\nОШИБКА: {vg._explain(exc)}", file=sys.stderr)
            return 1
        status = vg._status_of(job)
        if not statuses or statuses[-1] != status:
            statuses.append(status)
        print("\r" + vg.render_bar(status, vg._progress_of(job), elapsed), end="", flush=True)
        if status in vg.DONE_STATUSES:
            print()
            break
        if status in vg.FAILED_STATUSES:
            print(f"\nСервис сообщил о неудаче: статус «{status}»", file=sys.stderr)
            return 1
        sleep(interval)

    result = vg.VideoResult(prompt="(получено через test.py)", video_id=video_id,
                            seconds=str(getattr(job, "seconds", "") or ""),
                            size=str(getattr(job, "size", "") or ""),
                            model=str(getattr(job, "model", "") or ""),
                            elapsed_s=round(time.perf_counter() - started, 1),
                            polls=len(statuses), statuses=statuses)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    try:
        path = vg.download_video(result, out_dir, f"{stamp}_{video_id[:8]}", client=client)
    except vg.VideoError as exc:
        print(f"ОШИБКА при скачивании: {exc}", file=sys.stderr)
        return 1
    vg.save_report(result, out_dir)
    print(f"Путь статусов: {' → '.join(statuses)}")
    print(f"Видео сохранено: {path} ({result.size_mb:.1f} МБ)")
    return 0


def list_jobs(client=None, limit: int = 10) -> int:
    client = client or vg.make_client()
    try:
        jobs = client.videos.list()
    except Exception as exc:  # noqa: BLE001
        print(f"ОШИБКА: {vg._explain(exc)}", file=sys.stderr)
        return 1
    items = list(getattr(jobs, "data", []) or [])[:limit]
    if not items:
        print("Задач пока нет.")
        return 0
    print(f"{'идентификатор':<40} {'статус':<12} модель")
    for job in items:
        print(f"{getattr(job, 'id', '?'):<40} {vg._status_of(job):<12} {getattr(job, 'model', '?')}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_id", nargs="?", help="идентификатор задачи генерации")
    ap.add_argument("--watch", metavar="ID", help="следить за задачей до конца и скачать результат")
    ap.add_argument("--list", action="store_true", help="показать последние задачи")
    ap.add_argument("--out", default="results", help="папка для скачанного видео")
    args = ap.parse_args(argv)

    if args.list:
        return list_jobs()
    if args.watch:
        return watch(args.watch, HERE / args.out)
    if args.video_id:
        return show_status(args.video_id)
    return check_access()


if __name__ == "__main__":
    raise SystemExit(main())
