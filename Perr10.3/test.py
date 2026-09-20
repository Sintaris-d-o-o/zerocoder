"""Проверка статуса генерации видео — файл `test.py` из задания 10.3.

Задание просит «проверить статус через test.py». Скрипт делает четыре вещи:

    python test.py                 — проверяет доступ и остаток на счету;
    python test.py <id задачи>     — показывает статус конкретной задачи;
    python test.py --watch <id>    — следит за задачей до конца и скачивает результат;
    python test.py --download <id> — скачивает уже готовое видео.

Отдельный файл нужен потому, что генерация долгая и платная: запустили в одном окне, а
проверять статус и забирать готовый файл удобно отдельной командой, не запуская генерацию
заново.
"""
from __future__ import annotations

import argparse
import json
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


def job_url(video_id: str) -> str:
    return f"{vg.ROUTERAI_BASE_URL}/videos/{video_id}"


def fetch(video_id: str, request=vg._request, key: Optional[str] = None) -> dict:
    return request(job_url(video_id), key or vg.api_key("routerai"))


def show_status(video_id: str, request=vg._request, key: Optional[str] = None) -> int:
    try:
        data = fetch(video_id, request, key)
    except vg.VideoError as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 1
    status = str(data.get("status", "")).lower()
    print(f"Задача {video_id}")
    print(f"  статус: {status}")
    cost = (data.get("usage") or {}).get("cost")
    if cost is not None:
        print(f"  стоимость: {cost} кредитов")
    urls = data.get("unsigned_urls") or []
    if urls:
        print(f"  файл готов, ссылок: {len(urls)}")
        print(f"  скачать: python test.py --download {video_id}")
    elif status in vg.RUNNING_STATUSES:
        print("  ещё генерируется — следить: python test.py --watch " + video_id)
    return 0


def watch(video_id: str, out_dir: Path, request=vg._request, key: Optional[str] = None,
          interval: float = vg.POLL_INTERVAL_S, timeout: float = vg.POLL_TIMEOUT_S,
          sleep=time.sleep, opener=None) -> int:
    """Следит за задачей до завершения и скачивает готовый файл."""
    key = key or vg.api_key("routerai")
    started = time.perf_counter()
    statuses: list[str] = []
    while True:
        elapsed = time.perf_counter() - started
        if elapsed > timeout:
            print(f"\nЗадача не завершилась за {timeout:.0f} с", file=sys.stderr)
            return 1
        try:
            data = fetch(video_id, request, key)
        except vg.VideoError as exc:
            print(f"\nОШИБКА: {exc}", file=sys.stderr)
            return 1
        status = str(data.get("status", "")).lower()
        if not statuses or statuses[-1] != status:
            statuses.append(status)
        print("\r" + vg.render_bar(status, elapsed), end="", flush=True)
        if status in vg.DONE_STATUSES:
            print()
            break
        if status in vg.FAILED_STATUSES:
            print(f"\nСервис сообщил о неудаче: статус «{status}»", file=sys.stderr)
            return 1
        sleep(interval)

    urls = data.get("unsigned_urls") or []
    if not urls:
        print("Задача готова, но ссылки на файл нет", file=sys.stderr)
        return 1

    result = vg.VideoResult(
        prompt="(получено через test.py)", video_id=video_id,
        seconds=int(data.get("duration") or 0), model=str(data.get("model") or ""),
        provider="routerai", elapsed_s=round(time.perf_counter() - started, 1),
        polls=len(statuses), statuses=statuses, download_url=urls[0],
        cost=(data.get("usage") or {}).get("cost"))

    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    try:
        kwargs = {"key": key}
        if opener is not None:
            kwargs["opener"] = opener
        path = vg.download_video(result, out_dir, f"{stamp}_{video_id[:8]}", **kwargs)
    except vg.VideoError as exc:
        print(f"ОШИБКА при скачивании: {exc}", file=sys.stderr)
        return 1
    vg.save_report(result, out_dir)
    print(f"Путь статусов: {' → '.join(statuses)}")
    print(f"Видео сохранено: {path} ({result.size_mb:.1f} МБ)")
    return 0


def download(video_id: str, out_dir: Path, request=vg._request, key: Optional[str] = None,
             opener=None) -> int:
    """Скачивает уже готовое видео, не дожидаясь ничего."""
    key = key or vg.api_key("routerai")
    try:
        data = fetch(video_id, request, key)
    except vg.VideoError as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 1
    urls = data.get("unsigned_urls") or []
    if not urls:
        print(f"Видео ещё не готово, статус «{data.get('status')}»", file=sys.stderr)
        return 1
    result = vg.VideoResult(prompt="(скачано через test.py)", video_id=video_id, seconds=0,
                            model=str(data.get("model") or ""), provider="routerai",
                            elapsed_s=0.0, polls=0, statuses=[str(data.get("status"))],
                            download_url=urls[0], cost=(data.get("usage") or {}).get("cost"))
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    kwargs = {"key": key}
    if opener is not None:
        kwargs["opener"] = opener
    try:
        path = vg.download_video(result, out_dir, f"{stamp}_{video_id[:8]}", **kwargs)
    except vg.VideoError as exc:
        print(f"ОШИБКА при скачивании: {exc}", file=sys.stderr)
        return 1
    vg.save_report(result, out_dir)
    print(f"Видео сохранено: {path} ({result.size_mb:.1f} МБ)")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_id", nargs="?", help="идентификатор задачи генерации")
    ap.add_argument("--watch", metavar="ID", help="следить за задачей и скачать результат")
    ap.add_argument("--download", metavar="ID", help="скачать уже готовое видео")
    ap.add_argument("--out", default="results", help="папка для скачанного видео")
    args = ap.parse_args(argv)

    if args.watch:
        return watch(args.watch, HERE / args.out)
    if args.download:
        return download(args.download, HERE / args.out)
    if args.video_id:
        return show_status(args.video_id)
    return vg.check_access()


if __name__ == "__main__":
    raise SystemExit(main())
