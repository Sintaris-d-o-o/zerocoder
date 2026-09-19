"""Генерация изображений через Yandex ART API (задание 10.1).

Генерация асинхронная, как описано в уроке:
    1. POST .../imageGenerationAsync  → получаем id операции;
    2. GET  .../operations/{id}       → опрашиваем, пока не появится done=true;
    3. response.image                 → base64, декодируем и сохраняем в файл.

Ключи только из .env (в корне репозитория или рядом со скриптом):
    YANDEX_API_KEY  — API-ключ сервисного аккаунта
    YANDEX_FOLDER_ID — идентификатор каталога в Yandex Cloud

Запуск:
    python yandex_art.py                       # все промпты проекта
    python yandex_art.py --prompt "текст"      # свой промпт
    python yandex_art.py --list                # показать заготовленные промпты, ничего не генерируя
    python yandex_art.py --dry-run             # проверить сборку запроса без обращения к API
"""
from __future__ import annotations

import argparse
import base64
import binascii
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
load_dotenv(HERE.parent / ".env")   # zerocoder/.env — общий для всех заданий
load_dotenv(HERE / ".env")          # локальная копия рядом со скриптом, если есть

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

GENERATE_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/imageGenerationAsync"
OPERATION_URL = "https://llm.api.cloud.yandex.net:443/operations/{operation_id}"

POLL_INTERVAL_S = 5      # урок советует опрашивать раз в несколько секунд, а не десять раз в секунду
POLL_TIMEOUT_S = 300     # потолок ожидания: генерация обычно занимает 10–40 секунд


class YandexArtError(RuntimeError):
    """Ошибка на стороне API или в настройках доступа."""


@dataclass
class GenerationResult:
    prompt: str
    seed: Optional[int]
    image_bytes: bytes
    operation_id: str
    elapsed_s: float
    polls: int
    saved_to: Optional[Path] = None

    @property
    def size_kb(self) -> float:
        return len(self.image_bytes) / 1024


# --------------------------------------------------------------------------- запрос

def build_payload(prompt: str, seed: Optional[int], folder_id: str,
                  width_ratio: str = "1", height_ratio: str = "1") -> dict:
    """Тело запроса ровно той структуры, что разобрана в уроке."""
    return {
        "modelUri": f"art://{folder_id}/yandex-art/latest",
        "generationOptions": {
            "seed": seed,
            "aspectRatio": {"widthRatio": width_ratio, "heightRatio": height_ratio},
        },
        "messages": [{"weight": "1", "text": prompt}],
    }


def build_headers(api_key: str) -> dict:
    # Префикс именно Api-Key — у Яндекса свой формат, у других сервисов бывает Bearer
    return {"Content-Type": "application/json", "Authorization": f"Api-Key {api_key}"}


def _post_json(url: str, payload: dict, headers: dict, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers=headers, method="POST")
    return _read(req, timeout)


def _get_json(url: str, headers: dict, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers=headers, method="GET")
    return _read(req, timeout)


def _read(req: urllib.request.Request, timeout: int) -> dict:
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:500]
        raise YandexArtError(_explain_http_error(exc.code, body)) from exc
    except urllib.error.URLError as exc:
        raise YandexArtError(f"Сеть недоступна: {exc.reason}") from exc


def _explain_http_error(code: int, body: str) -> str:
    """Понятное объяснение вместо голого кода ошибки."""
    hints = {
        401: "ключ не принят. Проверьте YANDEX_API_KEY и что перед ним стоит префикс Api-Key",
        403: "доступ запрещён. У сервисного аккаунта должна быть роль ai.imageGeneration.user, "
             "а у платёжного аккаунта — статус ACTIVE или TRIAL_ACTIVE",
        404: "адрес не найден. Проверьте YANDEX_FOLDER_ID — он входит в modelUri",
        429: "слишком много запросов, сервис просит подождать",
    }
    hint = hints.get(code, "непредвиденный ответ сервиса")
    return f"HTTP {code}: {hint}. Ответ сервера: {body}"


# --------------------------------------------------------------------------- генерация

def generate_image(prompt: str, seed: Optional[int] = None, *,
                   api_key: Optional[str] = None, folder_id: Optional[str] = None,
                   width_ratio: str = "1", height_ratio: str = "1",
                   poll_interval_s: float = POLL_INTERVAL_S,
                   timeout_s: float = POLL_TIMEOUT_S,
                   on_poll: Optional[Callable[[int, float], None]] = None,
                   post_json: Callable = _post_json, get_json: Callable = _get_json,
                   sleep: Callable[[float], None] = time.sleep) -> GenerationResult:
    """Запускает генерацию и ждёт результат.

    Сетевые вызовы вынесены в параметры (`post_json`, `get_json`, `sleep`) — так их
    подменяют тесты, и логика проверяется без обращения к платному API.
    """
    api_key = api_key or os.getenv("YANDEX_API_KEY", "")
    folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID", "") or os.getenv("YANDEX_CLOUD_ID", "")
    if not api_key:
        raise YandexArtError("Не задан YANDEX_API_KEY. Положите его в .env — см. .env.example")
    if not folder_id:
        raise YandexArtError("Не задан YANDEX_FOLDER_ID. Положите его в .env — см. .env.example")

    headers = build_headers(api_key)
    payload = build_payload(prompt, seed, folder_id, width_ratio, height_ratio)

    started = time.perf_counter()
    created = post_json(GENERATE_URL, payload, headers)
    operation_id = created.get("id")
    if not operation_id:
        raise YandexArtError(f"Сервис не вернул идентификатор операции. Ответ: {created}")

    polls = 0
    while True:
        elapsed = time.perf_counter() - started
        if elapsed > timeout_s:
            raise YandexArtError(
                f"Генерация не завершилась за {timeout_s:.0f} с (операция {operation_id}). "
                f"Обычно это занимает 10–40 секунд — вероятно, сервис перегружен")
        sleep(poll_interval_s)
        polls += 1
        status = get_json(OPERATION_URL.format(operation_id=operation_id), headers)
        if on_poll:
            on_poll(polls, time.perf_counter() - started)
        if not status.get("done"):
            continue
        if "error" in status:
            raise YandexArtError(f"Сервис вернул ошибку генерации: {status['error']}")
        image_b64 = (status.get("response") or {}).get("image")
        if not image_b64:
            raise YandexArtError(f"В ответе нет изображения. Ответ: {json.dumps(status)[:300]}")
        try:
            image_bytes = base64.b64decode(image_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise YandexArtError(f"Не удалось раскодировать изображение: {exc}") from exc
        return GenerationResult(prompt=prompt, seed=seed, image_bytes=image_bytes,
                                operation_id=operation_id,
                                elapsed_s=round(time.perf_counter() - started, 1), polls=polls)


def save_image(result: GenerationResult, directory: Path, name: str) -> Path:
    """Сохраняет изображение. Имя чистится от символов, запрещённых в Windows."""
    directory.mkdir(parents=True, exist_ok=True)
    safe = "".join("_" if ch in ':<>"/\\|?*' else ch for ch in name).strip() or "image"
    path = directory / f"{safe}.jpeg"
    path.write_bytes(result.image_bytes)
    result.saved_to = path
    return path


# --------------------------------------------------------------------------- промпты проекта

@dataclass
class Prompt:
    key: str
    title: str
    text: str
    tags: list[str] = field(default_factory=list)


# Собственные промпты (задание требует не брать промпт эксперта): логотипы для SINTARIS —
# продукта по управлению сертификатами медицинских изделий.
PROMPTS: list[Prompt] = [
    Prompt(
        key="shield-check",
        title="Щит с галочкой — основной знак",
        text=(
            "Минималистичный векторный логотип для компании SINTARIS: щит со вписанной "
            "галочкой внутри, в центре тонкая линия в виде медицинского креста. "
            "Два цвета: глубокий синий и белый, на белом фоне. "
            "Плоский стиль, чистые геометрические формы, без текста и без букв, "
            "чёткие края, подходит для печати на документах."
        ),
        tags=["минимализм", "доверие", "медицина"],
    ),
    Prompt(
        key="document-seal",
        title="Документ с печатью — знак соответствия",
        text=(
            "Строгий корпоративный логотип: стилизованный лист документа с круглой печатью "
            "в правом нижнем углу, печать образована кольцом из двенадцати звёзд. "
            "Синий и серебристый цвета на белом фоне, плоский векторный стиль, "
            "тонкие линии, без текста и без букв, симметричная композиция."
        ),
        tags=["официальный", "сертификация", "Европа"],
    ),
    Prompt(
        key="tech-pulse",
        title="Пульс и микросхема — ассистент по нормам",
        text=(
            "Современный технологичный логотип: линия кардиограммы переходит в дорожку "
            "печатной платы с узлами-точками. Градиент от бирюзового к синему, тёмный фон, "
            "неоновое свечение линий, плоский стиль, без текста и без букв, квадратная "
            "композиция по центру."
        ),
        tags=["технологии", "ИИ", "медтех"],
    ),
]


def find_prompt(key: str) -> Prompt:
    for p in PROMPTS:
        if p.key == key:
            return p
    raise KeyError(f"Нет промпта с ключом {key!r}. Доступны: {', '.join(p.key for p in PROMPTS)}")


# --------------------------------------------------------------------------- CLI

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", help="свой текст промпта (иначе берутся заготовленные)")
    ap.add_argument("--key", help="ключ заготовленного промпта")
    ap.add_argument("--seed", type=int, help="зерно генерации: с ним результат воспроизводим")
    ap.add_argument("--ratio", default="1:1", help="соотношение сторон, например 1:1 или 16:9")
    ap.add_argument("--out", default="results", help="папка для картинок")
    ap.add_argument("--list", action="store_true", help="показать заготовленные промпты и выйти")
    ap.add_argument("--dry-run", action="store_true", help="собрать запрос и показать его, не отправляя")
    args = ap.parse_args(argv)

    if args.list:
        print(f"Заготовленные промпты ({len(PROMPTS)}):\n")
        for p in PROMPTS:
            print(f"  {p.key:15s} {p.title}")
            print(f"  {'':15s} теги: {', '.join(p.tags)}")
            print(f"  {'':15s} {p.text[:100]}…\n")
        return 0

    try:
        width, height = args.ratio.split(":")
    except ValueError:
        print(f"Неверный формат --ratio: {args.ratio!r}. Нужно вида 1:1", file=sys.stderr)
        return 2

    if args.prompt:
        jobs = [Prompt(key="custom", title="Свой промпт", text=args.prompt)]
    elif args.key:
        jobs = [find_prompt(args.key)]
    else:
        jobs = PROMPTS

    if args.dry_run:
        folder = os.getenv("YANDEX_FOLDER_ID") or os.getenv("YANDEX_CLOUD_ID") or "<нет в .env>"
        key = os.getenv("YANDEX_API_KEY")
        print(f"Каталог (folder id): {folder}")
        print(f"Ключ: {'задан, ' + str(len(key)) + ' символов' if key else 'НЕ ЗАДАН'}")
        for p in jobs:
            payload = build_payload(p.text, args.seed, folder, width, height)
            print(f"\n--- {p.key}: {p.title}")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        print("\n--dry-run: запросы не отправлялись, деньги не потрачены.")
        return 0

    out_dir = HERE / args.out
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    ok = 0
    for p in jobs:
        print(f"\n=== {p.key}: {p.title}")
        print(f"Промпт: {p.text}")
        try:
            def report(n: int, elapsed: float, _p=p) -> None:
                print(f"  ожидание… проверка {n}, прошло {elapsed:.0f} с")

            res = generate_image(p.text, args.seed, width_ratio=width, height_ratio=height,
                                 on_poll=report)
        except YandexArtError as exc:
            print(f"  ОШИБКА: {exc}", file=sys.stderr)
            continue
        path = save_image(res, out_dir, f"{stamp}_{p.key}")
        ok += 1
        print(f"  готово за {res.elapsed_s} с ({res.polls} проверок), {res.size_kb:.0f} КБ")
        print(f"  сохранено: {path.relative_to(HERE)}")

    print(f"\nУспешно сгенерировано: {ok} из {len(jobs)}")
    return 0 if ok == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
