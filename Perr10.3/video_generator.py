"""Генерация видео по текстовому описанию (задание 10.3).

Поддерживаются два сервиса-посредника, выбор — по тому, какой ключ лежит в `.env`:

  **RouterAI** (`ROUTERAIRU_API_KEY`) — используется по умолчанию. Свой REST-интерфейс:
      1. POST /api/v1/videos                  → {"id", "polling_url", "status": "pending"}
      2. GET  polling_url                     → опрашиваем, пока не станет "completed"
      3. GET  unsigned_urls[0]                → скачиваем готовый MP4

  **Proxy API** (`PROXYAPI_KEY`) — способ из урока: стандартный клиент `openai` с другим
      адресом сервера, вызовы `videos.create` / `videos.retrieve` / `videos.download_content`.

Сервис в уроке другой, но урок прямо разрешает: «Использование иностранных сервисов не
является обязательным. Вы вправе выполнять домашние задания на аналогичных разрешённых
платформах по вашему выбору». Суть задания та же — посредник с единым API к чужим моделям,
асинхронная генерация в три шага.

Запуск:
    python video_generator.py                    # промпт проекта, 4 секунды
    python video_generator.py --list             # промпты и модели, ничего не тратя
    python video_generator.py --check            # баланс и доступ, ничего не тратя
    python video_generator.py --dry-run          # показать запрос, не отправляя
    python video_generator.py --model google/veo-3.1-lite --seconds 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
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

# ── RouterAI ────────────────────────────────────────────────────────────────
ROUTERAI_BASE_URL = "https://routerai.ru/api/v1"
DEFAULT_MODEL = "google/veo-3.1-fast"     # проверено рабочим 20.09.2026
DEFAULT_SECONDS = 4                        # задание требует проверить генерацию на 4 секундах
DEFAULT_ASPECT = "16:9"
DEFAULT_RESOLUTION = "720p"

# ── Proxy API (способ из урока) ─────────────────────────────────────────────
PROXYAPI_BASE_URL = "https://api.proxyapi.ru/openai/v1"
PROXYAPI_MODEL = "sora-2"

POLL_INTERVAL_S = 5
POLL_TIMEOUT_S = 900          # генерация видео долгая: очередь плюс рендеринг

DONE_STATUSES = {"completed", "succeeded", "complete", "done", "success"}
FAILED_STATUSES = {"failed", "error", "cancelled", "canceled", "rejected"}
RUNNING_STATUSES = {"pending", "queued", "processing", "in_progress", "running", "starting"}


class VideoError(RuntimeError):
    """Ошибка сервиса, настроек доступа или самой генерации."""


@dataclass
class VideoResult:
    prompt: str
    video_id: str
    seconds: int
    model: str
    provider: str
    elapsed_s: float
    polls: int
    statuses: list[str] = field(default_factory=list)
    download_url: Optional[str] = None
    cost: Optional[float] = None
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
# По-английски намеренно: видеомодели обучены в основном на английских описаниях и понимают
# их точнее — это приём «перевести на более ресурсный язык» из урока 9.1.
PROMPTS: list[Prompt] = [
    Prompt(
        key="certificates",
        title="Документы на столе — порядок вместо хаоса",
        text=("Close-up of a clean white desk in a bright office. A stack of official documents "
              "with embossed seals slides into a perfectly aligned pile. Soft daylight from the "
              "left, shallow depth of field, slow dolly-in. Calm corporate mood, blue and white, "
              "photorealistic, no text."),
    ),
    Prompt(
        key="lab",
        title="Лаборатория — контроль качества",
        text=("A sterile medical device laboratory. A gloved hand places a small electronic "
              "device onto a testing stand; indicator lights turn from amber to green. Slow "
              "push-in, crisp reflections on stainless steel, cool teal lighting, "
              "photorealistic, no text."),
    ),
    Prompt(
        key="dashboard",
        title="Панель контроля сроков",
        text=("Abstract compliance dashboard floating in dark space: rows of glowing cards drift "
              "slowly, status indicators shifting from red to green one by one. Thin blue light "
              "lines connect the cards. Smooth parallax drift, soft bloom, minimalist and "
              "futuristic, no readable text."),
    ),
]


def find_prompt(key: str) -> Prompt:
    for p in PROMPTS:
        if p.key == key:
            return p
    raise KeyError(f"Нет промпта с ключом {key!r}. Доступны: {', '.join(p.key for p in PROMPTS)}")


# --------------------------------------------------------------------------- выбор сервиса

def provider_name() -> str:
    """Какой сервис использовать: тот, чей ключ есть в .env. RouterAI приоритетнее."""
    forced = os.getenv("VIDEO_PROVIDER", "").strip().lower()
    if forced:
        return forced
    if os.getenv("ROUTERAIRU_API_KEY"):
        return "routerai"
    if os.getenv("PROXYAPI_KEY"):
        return "proxyapi"
    return "routerai"


def api_key(provider: Optional[str] = None) -> str:
    provider = provider or provider_name()
    name = "ROUTERAIRU_API_KEY" if provider == "routerai" else "PROXYAPI_KEY"
    key = os.getenv(name, "")
    if not key:
        raise VideoError(f"Не задан {name}. Положите ключ в .env — см. .env.example")
    return key


def model_for(provider: Optional[str] = None) -> str:
    provider = provider or provider_name()
    env = os.getenv("VIDEO_MODEL", "")
    if env:
        return env
    return DEFAULT_MODEL if provider == "routerai" else PROXYAPI_MODEL


# --------------------------------------------------------------------------- сеть

def _request(url: str, key: str, payload: Optional[dict] = None, method: str = "GET",
             timeout: int = 60) -> dict:
    headers = {"Authorization": f"Bearer {key}"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:400]
        raise VideoError(_explain(exc.code, body)) from exc
    except urllib.error.URLError as exc:
        raise VideoError(f"Сеть недоступна: {exc.reason}") from exc


def _explain(code: int, body: str) -> str:
    hints = {
        401: "ключ не принят — проверьте ROUTERAIRU_API_KEY",
        402: "недостаточно средств на счету сервиса",
        403: "доступ к модели закрыт для этого ключа",
        404: "адрес или модель не найдены — проверьте название модели",
        422: "сервис не принял параметры запроса (длительность, разрешение или пропорции)",
        429: "слишком много запросов, сервис просит подождать",
    }
    return f"HTTP {code}: {hints.get(code, 'непредвиденный ответ сервиса')}. Ответ: {body}"


def credits(key: Optional[str] = None, request: Callable = _request) -> Optional[float]:
    """Сколько кредитов осталось на счету RouterAI.

    Ошибку не поднимает: остаток — справочная величина, из-за неё не должна падать генерация.
    """
    try:
        data = request(f"{ROUTERAI_BASE_URL}/credits", key or api_key("routerai"))
        return float(data.get("data", {}).get("credits"))
    except (VideoError, TypeError, ValueError, AttributeError):
        return None


# --------------------------------------------------------------------------- прогресс

def render_bar(status: str, elapsed: float, width: int = 28) -> str:
    """Строка прогресса для терминала: задание просит скриншот с прогресс-баром.

    Сервис не сообщает процент готовности, поэтому полоса отражает стадию:
    очередь — четверть, обработка — половина, готово — целиком.
    """
    pct = {"pending": 25, "queued": 25, "starting": 25,
           "processing": 60, "in_progress": 60, "running": 60}.get(status, 0)
    if status in DONE_STATUSES:
        pct = 100
    filled = max(0, min(width, round(width * pct / 100)))
    bar = "█" * filled + "·" * (width - filled)
    return f"  [{bar}] {pct:3d}%  {status:<12} {elapsed:5.0f} с"


# --------------------------------------------------------------------------- генерация

def generate_video(prompt: str, *, seconds: int = DEFAULT_SECONDS, model: Optional[str] = None,
                   aspect_ratio: str = DEFAULT_ASPECT, resolution: str = DEFAULT_RESOLUTION,
                   provider: Optional[str] = None, key: Optional[str] = None,
                   poll_interval_s: float = POLL_INTERVAL_S, timeout_s: float = POLL_TIMEOUT_S,
                   on_progress: Optional[Callable[[str, float], None]] = None,
                   request: Callable = _request, sleep: Callable[[float], None] = time.sleep,
                   client=None) -> VideoResult:
    """Запускает генерацию и ждёт готовности.

    `request`, `sleep` и `client` вынесены в параметры — так их подменяют тесты, и логика
    проверяется без обращения к платному сервису.
    """
    provider = provider or provider_name()
    model = model or model_for(provider)
    if provider == "proxyapi":
        return _generate_proxyapi(prompt, seconds=seconds, model=model, client=client,
                                  poll_interval_s=poll_interval_s, timeout_s=timeout_s,
                                  on_progress=on_progress, sleep=sleep)

    key = key or api_key("routerai")
    started = time.perf_counter()
    created = request(f"{ROUTERAI_BASE_URL}/videos", key, {
        "model": model, "prompt": prompt, "duration": seconds,
        "aspect_ratio": aspect_ratio, "resolution": resolution,
    }, "POST")

    video_id = created.get("id")
    if not video_id:
        raise VideoError(f"Сервис не вернул идентификатор задачи. Ответ: {created}")
    poll_url = created.get("polling_url") or f"{ROUTERAI_BASE_URL}/videos/{video_id}"

    statuses = [str(created.get("status", "pending")).lower()]
    if on_progress:
        on_progress(statuses[-1], 0.0)

    polls = 0
    while True:
        elapsed = time.perf_counter() - started
        if elapsed > timeout_s:
            raise VideoError(
                f"Генерация не завершилась за {timeout_s:.0f} с (задача {video_id}, "
                f"последний статус «{statuses[-1]}»). Задача могла остаться в очереди")
        sleep(poll_interval_s)
        polls += 1
        data = request(poll_url, key)
        status = str(data.get("status", "")).lower()
        if status and status != statuses[-1]:
            statuses.append(status)
        if on_progress:
            on_progress(status, time.perf_counter() - started)

        if status in DONE_STATUSES:
            urls = data.get("unsigned_urls") or data.get("urls") or []
            if not urls:
                raise VideoError(f"Задача готова, но ссылки на файл нет. Ответ: {json.dumps(data)[:300]}")
            cost = (data.get("usage") or {}).get("cost")
            return VideoResult(prompt=prompt, video_id=video_id, seconds=seconds, model=model,
                               provider="routerai",
                               elapsed_s=round(time.perf_counter() - started, 1), polls=polls,
                               statuses=statuses, download_url=urls[0],
                               cost=float(cost) if cost is not None else None)
        if status in FAILED_STATUSES:
            detail = data.get("error") or data.get("message")
            raise VideoError(f"Сервис сообщил о неудаче: статус «{status}»"
                             + (f", подробности: {detail}" if detail else ""))
        if status and status not in RUNNING_STATUSES:
            raise VideoError(f"Неизвестный статус «{status}». Ответ: {json.dumps(data)[:200]}")


def _generate_proxyapi(prompt: str, *, seconds: int, model: str, client,
                       poll_interval_s: float, timeout_s: float,
                       on_progress, sleep) -> VideoResult:
    """Способ из урока: стандартный клиент OpenAI, направленный на сервер посредника."""
    if client is None:
        from openai import OpenAI
        client = OpenAI(api_key=api_key("proxyapi"),
                        base_url=os.getenv("PROXYAPI_BASE_URL", PROXYAPI_BASE_URL),
                        timeout=120.0, max_retries=2)
    started = time.perf_counter()
    try:
        job = client.videos.create(model=model, prompt=prompt, seconds=str(seconds))
    except Exception as exc:  # noqa: BLE001
        raise VideoError(f"{type(exc).__name__}: {str(exc)[:300]}") from exc

    video_id = getattr(job, "id", None)
    if not video_id:
        raise VideoError(f"Сервис не вернул идентификатор задачи. Ответ: {job!r}")
    statuses = [str(getattr(job, "status", "queued")).lower()]
    polls = 0
    while True:
        if time.perf_counter() - started > timeout_s:
            raise VideoError(f"Генерация не завершилась за {timeout_s:.0f} с (задача {video_id})")
        sleep(poll_interval_s)
        polls += 1
        job = client.videos.retrieve(video_id)
        status = str(getattr(job, "status", "")).lower()
        if status and status != statuses[-1]:
            statuses.append(status)
        if on_progress:
            on_progress(status, time.perf_counter() - started)
        if status in DONE_STATUSES:
            return VideoResult(prompt=prompt, video_id=video_id, seconds=seconds, model=model,
                               provider="proxyapi",
                               elapsed_s=round(time.perf_counter() - started, 1), polls=polls,
                               statuses=statuses)
        if status in FAILED_STATUSES:
            raise VideoError(f"Сервис сообщил о неудаче: статус «{status}»")


def download_video(result: VideoResult, directory: Path, name: str, *,
                   key: Optional[str] = None, client=None,
                   opener: Callable = urllib.request.urlopen) -> Path:
    """Скачивает готовый MP4 и сохраняет на диск."""
    directory.mkdir(parents=True, exist_ok=True)
    safe = "".join("_" if ch in ':<>"/\\|?*' else ch for ch in name).strip() or "video"
    path = directory / f"{safe}.mp4"

    if result.provider == "routerai":
        if not result.download_url:
            raise VideoError("У результата нет ссылки на файл")
        req = urllib.request.Request(result.download_url,
                                     headers={"Authorization": f"Bearer {key or api_key('routerai')}"})
        try:
            data = opener(req, timeout=600).read()
        except urllib.error.HTTPError as exc:
            raise VideoError(f"Не удалось скачать видео: {_explain(exc.code, '')}") from exc
    else:
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key("proxyapi"),
                            base_url=os.getenv("PROXYAPI_BASE_URL", PROXYAPI_BASE_URL))
        content = client.videos.download_content(result.video_id)
        data = content.read() if hasattr(content, "read") else bytes(content)

    if not data:
        raise VideoError("Сервис вернул пустой файл")
    if not data[4:8] == b"ftyp":
        raise VideoError(f"Скачанное не похоже на MP4: первые байты {data[:12]!r}")
    path.write_bytes(data)
    result.saved_to = path
    result.size_bytes = len(data)
    return path


def save_report(result: VideoResult, directory: Path) -> Path:
    """Сохраняет параметры прогона рядом с видео — чтобы результат можно было повторить."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = result.saved_to.stem if result.saved_to else result.video_id
    path = directory / f"{stem}.json"
    path.write_text(json.dumps({
        "prompt": result.prompt, "video_id": result.video_id, "provider": result.provider,
        "model": result.model, "seconds": result.seconds, "elapsed_s": result.elapsed_s,
        "polls": result.polls, "statuses": result.statuses, "cost_credits": result.cost,
        "file": result.saved_to.name if result.saved_to else None,
        "size_mb": round(result.size_mb, 2),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- проверка доступа

def check_access() -> int:
    """Показывает сервис, ключ, модель и остаток на счету. Генерацию не запускает."""
    provider = provider_name()
    print("Проверка доступа")
    print(f"  сервис:  {provider}")
    try:
        key = api_key(provider)
        print(f"  ключ:    задан, {len(key)} символов")
    except VideoError as exc:
        print(f"  ключ:    {exc}")
        return 2
    print(f"  модель:  {model_for(provider)}")
    if provider != "routerai":
        print("\nОстаток на счету этот сервис через API не отдаёт — смотрите в личном кабинете.")
        return 0
    left = credits(key)
    if left is None:
        print("  счёт:    не удалось узнать остаток")
        return 1
    print(f"  счёт:    {left:.1f} кредитов")
    print()
    print(f"Одно видео на 4 секунды моделью {DEFAULT_MODEL} стоило 43,8 кредита —")
    print(f"этого остатка хватит примерно на {int(left // 44)} ролика.")
    return 0


# --------------------------------------------------------------------------- CLI

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", help="свой текст промпта")
    ap.add_argument("--key", default="certificates", help="ключ заготовленного промпта")
    ap.add_argument("--seconds", type=int, default=DEFAULT_SECONDS, help="длительность в секундах")
    ap.add_argument("--model", help=f"модель, по умолчанию {DEFAULT_MODEL}")
    ap.add_argument("--aspect", default=DEFAULT_ASPECT)
    ap.add_argument("--resolution", default=DEFAULT_RESOLUTION)
    ap.add_argument("--provider", choices=["routerai", "proxyapi"], help="какой сервис использовать")
    ap.add_argument("--out", default="results", help="папка для видео")
    ap.add_argument("--list", action="store_true", help="показать промпты и выйти")
    ap.add_argument("--check", action="store_true", help="проверить доступ и остаток на счету")
    ap.add_argument("--dry-run", action="store_true", help="показать запрос, не отправляя его")
    args = ap.parse_args(argv)

    if args.check:
        return check_access()

    if args.list:
        print(f"Заготовленные промпты ({len(PROMPTS)}):\n")
        for p in PROMPTS:
            print(f"  {p.key:15s} {p.title}")
            print(f"  {'':15s} {p.text[:110]}…\n")
        print("Видеомодели RouterAI: openai/sora-2-pro, google/veo-3.1, google/veo-3.1-fast,")
        print("google/veo-3.1-lite, kwaivgi/kling-v3.0-pro, kwaivgi/kling-v3.0-std")
        return 0

    provider = args.provider or provider_name()
    model = args.model or model_for(provider)
    prompt = args.prompt or find_prompt(args.key).text

    if args.dry_run:
        print(f"Сервис: {provider}")
        print("Запрос, который будет отправлен:")
        print(json.dumps({"model": model, "prompt": prompt, "duration": args.seconds,
                          "aspect_ratio": args.aspect, "resolution": args.resolution},
                         ensure_ascii=False, indent=2))
        print("\n--dry-run: запрос не отправлялся, деньги не потрачены.")
        return 0

    print(f"Сервис: {provider}, модель: {model}")
    print(f"Длительность: {args.seconds} с, {args.aspect}, {args.resolution}")
    print(f"Промпт: {prompt}\n")

    def progress(status: str, elapsed: float) -> None:
        print("\r" + render_bar(status, elapsed), end="", flush=True)

    try:
        result = generate_video(prompt, seconds=args.seconds, model=model,
                                aspect_ratio=args.aspect, resolution=args.resolution,
                                provider=provider, on_progress=progress)
    except VideoError as exc:
        print(f"\nОШИБКА: {exc}", file=sys.stderr)
        return 1

    print()
    print(f"\nГотово за {result.elapsed_s} с, проверок статуса: {result.polls}")
    print(f"Путь статусов: {' → '.join(result.statuses)}")
    if result.cost is not None:
        print(f"Стоимость: {result.cost:.1f} кредитов")

    out_dir = HERE / args.out
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    try:
        path = download_video(result, out_dir, f"{stamp}_{args.key}")
    except VideoError as exc:
        print(f"ОШИБКА при скачивании: {exc}", file=sys.stderr)
        print(f"Видео осталось на сервисе, идентификатор: {result.video_id}", file=sys.stderr)
        return 1

    report = save_report(result, out_dir)
    print(f"Видео сохранено: {path.relative_to(HERE)} ({result.size_mb:.1f} МБ)")
    print(f"Параметры прогона: {report.relative_to(HERE)}")
    left = credits()
    if left is not None:
        print(f"Остаток на счету: {left:.1f} кредитов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
