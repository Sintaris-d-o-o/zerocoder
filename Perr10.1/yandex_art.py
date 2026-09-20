"""Генерация изображений через Yandex AI Studio (задание 10.1).

**Важно: API изменился по сравнению с уроком.** В уроке используется асинхронный вызов
`foundationModels/v1/imageGenerationAsync` с моделью `yandex-art`. На сентябрь 2026 этот путь
отвечает отказом «Access to model art://yandex-art/latest denied» при полностью правильных
настройках: у каталога есть роли, у ключа — нужная область действия, оплата активна, а
текстовые модели тем же ключом работают. Модель `yandex-art` в каталоге моделей AI Studio
больше не значится.

Актуальный путь — **OpenAI-совместимый API** Yandex AI Studio: тот же клиент `openai`, но с
другим адресом сервера. Генерация синхронная, отдельный опрос операции не нужен.
Доступные модели генерации изображений видны в консоли AI Studio, раздел «Модели».

Старый способ из урока сохранён и вызывается ключом `--legacy`, чтобы можно было показать
разницу и проверить, не вернулась ли модель.

Ключи только из .env (в корне репозитория или рядом со скриптом):
    YANDEX_API_KEY     — API-ключ сервисного аккаунта
    YANDEX_FOLDER_ID   — идентификатор каталога в Yandex Cloud
    YANDEX_ART_API_KEY — необязательный отдельный ключ для картинок
    YANDEX_ART_MODEL   — необязательная замена модели

Запуск:
    python yandex_art.py                       # все промпты проекта
    python yandex_art.py --prompt "текст"      # свой промпт
    python yandex_art.py --list                # показать заготовленные промпты, ничего не генерируя
    python yandex_art.py --check               # проверить доступ, ничего не генерируя
    python yandex_art.py --dry-run             # показать запрос без обращения к API
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

# --- актуальный путь: OpenAI-совместимый API Yandex AI Studio ---
AI_STUDIO_BASE_URL = "https://ai.api.cloud.yandex.net/v1"
# Задание требует именно YandexART. В каталоге моделей AI Studio она называется
# yandex-art-2.0 (старое имя yandex-art из урока больше не обслуживается).
# Альтернатива из того же раздела: aliceai-image-art-3.0/latest.
DEFAULT_MODEL = "yandex-art-2.0/latest"          # проверено рабочим 20.09.2026
DEFAULT_SIZE = "1024x1024"
PROMPT_LIMIT = 500   # предел длины промпта у YandexART 2.0, указан в карточке модели

# --- путь из урока (сейчас отвечает отказом, оставлен для сравнения) ---
GENERATE_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/imageGenerationAsync"
# Статус операции спрашивают у отдельной службы операций, а не у той, что приняла запрос
# (в уроке указан адрес llm.api..., документация называет operation.api...).
OPERATION_URL = "https://operation.api.cloud.yandex.net:443/operations/{operation_id}"
LEGACY_MODEL = "yandex-art/latest"

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

def make_client(api_key: Optional[str] = None, folder_id: Optional[str] = None):
    """Клиент OpenAI, направленный на сервер Yandex AI Studio.

    Каталог передаётся параметром `project` — так этот сервер понимает, чьи квоты тратить.
    """
    api_key = api_key or art_api_key()
    folder_id = folder_id or folder()
    if not api_key:
        raise YandexArtError("Не задан YANDEX_API_KEY. Положите его в .env — см. .env.example")
    if not folder_id:
        raise YandexArtError("Не задан YANDEX_FOLDER_ID. Положите его в .env — см. .env.example")
    import openai
    return openai.OpenAI(api_key=api_key, base_url=AI_STUDIO_BASE_URL, project=folder_id,
                         timeout=180.0, max_retries=1)


def folder() -> str:
    return os.getenv("YANDEX_FOLDER_ID", "") or os.getenv("YANDEX_CLOUD_ID", "")


def model_name() -> str:
    return os.getenv("YANDEX_ART_MODEL", "") or DEFAULT_MODEL


def build_payload(prompt: str, seed: Optional[int], folder_id: str,
                  width_ratio: str = "1", height_ratio: str = "1") -> dict:
    """Тело запроса ровно той структуры, что разобрана в уроке (старый путь)."""
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


def art_api_key() -> str:
    """Ключ для генерации картинок.

    У API-ключей Yandex Cloud есть собственная «область действия», отдельная от ролей
    аккаунта: ключ с областью `yc.ai.languageModels.execute` работает с текстовыми моделями,
    но к картинкам его не пустят, сколько ролей аккаунту ни выдай. Если область выбирается
    только одна, нужен второй ключ — его кладут в YANDEX_ART_API_KEY.
    """
    return os.getenv("YANDEX_ART_API_KEY", "") or os.getenv("YANDEX_API_KEY", "")


def text_api_key() -> str:
    """Ключ для текстовых моделей — используется только в проверке доступа."""
    return os.getenv("YANDEX_API_KEY", "") or os.getenv("YANDEX_ART_API_KEY", "")


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
    if code == 403 and "denied" in body:
        return ("HTTP 403: доступ к модели генерации изображений закрыт. Если текстовые модели "
                "при этом работают (проверьте командой --check), дело не в оплате, а в правах: "
                "выдайте сервисному аккаунту роль ai.imageGeneration.user в своём каталоге. "
                f"Ответ сервера: {body}")
    if code == 400 and "does not match with service account folder" in body:
        return ("HTTP 400: в YANDEX_FOLDER_ID указан не тот каталог. Правильный идентификатор "
                f"сервис назвал сам в ответе: {body}")
    hints = {
        401: "ключ не принят. Проверьте YANDEX_API_KEY и что перед ним стоит префикс Api-Key",
        403: "доступ запрещён. Проверьте роль сервисного аккаунта и статус платёжного аккаунта",
        404: "адрес не найден. Проверьте YANDEX_FOLDER_ID — он входит в modelUri",
        429: "слишком много запросов, сервис просит подождать",
    }
    hint = hints.get(code, "непредвиденный ответ сервиса")
    return f"HTTP {code}: {hint}. Ответ сервера: {body}"


def check_access(post_json: Callable = _post_json) -> int:
    """Проверяет доступ и различает две частые причины отказа. Генерацию не запускает.

    Текстовая модель и модель картинок оплачиваются одним аккаунтом, но права на них выдаются
    отдельно. Если текст работает, а картинки нет — дело в роли, а не в деньгах.
    """
    import hashlib

    text_key, art_key = text_api_key(), art_api_key()
    folder = os.getenv("YANDEX_FOLDER_ID", "") or os.getenv("YANDEX_CLOUD_ID", "")
    print("Проверка доступа к Yandex Cloud")

    def describe(k: str) -> str:
        if not k:
            return "НЕ ЗАДАН"
        # отпечаток: показывает, менялся ли ключ, но сам ключ не раскрывает
        return f"{len(k)} символов, отпечаток {hashlib.sha256(k.encode()).hexdigest()[:12]}"

    print(f"  ключ для текста:   {describe(text_key)}")
    print(f"  ключ для картинок: {describe(art_key)}"
          + ("  (тот же ключ)" if art_key == text_key and art_key else ""))
    print(f"  каталог:           {folder or 'НЕ ЗАДАН'}")
    if not art_key or not folder:
        print("\nДобавьте недостающее в .env — см. .env.example")
        return 2

    print(f"  модель:            {model_name()}")
    results = {}

    # 1) текстовая модель — показывает, что с оплатой и ключом всё хорошо
    try:
        post_json("https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
                  {"modelUri": f"gpt://{folder}/yandexgpt-lite",
                   "completionOptions": {"maxTokens": 1, "temperature": 0},
                   "messages": [{"role": "user", "text": "."}]},
                  build_headers(text_key))
        results["текст"] = True
        print("  текстовая модель:  доступна")
    except YandexArtError as exc:
        results["текст"] = False
        print(f"  текстовая модель:  НЕТ — {str(exc).split('. Ответ сервера')[0]}")

    # 2) картинки — актуальным способом
    try:
        generate_image("простой синий круг на белом фоне")
        results["картинки"] = True
        print("  генерация картинок: доступна")
    except YandexArtError as exc:
        results["картинки"] = False
        print(f"  генерация картинок: НЕТ — {str(exc).split('. Ответ сервиса')[0]}")

    print()
    if results.get("картинки"):
        print("Всё готово: можно запускать генерацию.")
        return 0
    if results.get("текст"):
        print("Оплата и ключ в порядке — текстовая модель отвечает.")
        print("Закрыта только генерация картинок. Что проверить:")
        print(f"  1. Модель «{model_name()}» должна быть в каталоге моделей AI Studio:")
        print(f"     https://aistudio.yandex.ru/platform/folders/{folder}/models")
        print("     Названия моделей меняются — если этой нет, возьмите любую из раздела")
        print("     «Изображения» и укажите её в YANDEX_ART_MODEL.")
        print("  2. У ключа должна быть область действия yc.ai.imageGeneration.execute")
        print("     (задаётся при создании ключа, ролями аккаунта не меняется).")
        print("  3. У сервисного аккаунта — роль ai.imageGeneration.user.")
    else:
        print("Не отвечает ни одна модель. Проверьте статус платёжного аккаунта")
        print("(должен быть ACTIVE или TRIAL_ACTIVE) и правильность ключа.")
    return 1


# --------------------------------------------------------------------------- генерация

def generate_image(prompt: str, seed: Optional[int] = None, *,
                   client=None, model: Optional[str] = None, size: str = DEFAULT_SIZE,
                   api_key: Optional[str] = None, folder_id: Optional[str] = None) -> GenerationResult:
    """Генерирует изображение через актуальный API Yandex AI Studio.

    Вызов синхронный: изображение приходит сразу в ответе, опрашивать операцию не нужно.
    `client` вынесен в параметр — так его подменяют тесты, и логика проверяется без
    обращения к платному API.
    """
    client = client or make_client(api_key, folder_id)
    model = model or model_name()
    folder_id = folder_id or folder()
    started = time.perf_counter()
    try:
        response = client.images.generate(model=f"art://{folder_id}/{model}",
                                          prompt=prompt, size=size)
    except Exception as exc:  # noqa: BLE001 — превращаем ошибку SDK в понятный текст
        raise YandexArtError(_explain_sdk_error(exc)) from exc

    items = getattr(response, "data", None) or []
    if not items:
        raise YandexArtError("Сервис не вернул изображение")
    b64 = getattr(items[0], "b64_json", None)
    if not b64:
        url = getattr(items[0], "url", None)
        raise YandexArtError(f"В ответе нет картинки в виде данных"
                             + (f", только ссылка: {url}" if url else ""))
    try:
        image_bytes = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise YandexArtError(f"Не удалось раскодировать изображение: {exc}") from exc

    return GenerationResult(prompt=prompt, seed=seed, image_bytes=image_bytes,
                            operation_id=model, polls=1,
                            elapsed_s=round(time.perf_counter() - started, 1))


def _explain_sdk_error(exc: Exception) -> str:
    """Понятное объяснение ошибки OpenAI-клиента."""
    text = str(exc).replace("\n", " ")
    code = getattr(exc, "status_code", None)
    if code == 403 and "denied" in text:
        return ("HTTP 403: доступ к этой модели закрыт. Проверьте, что модель есть в каталоге "
                "моделей AI Studio и что у ключа область действия yc.ai.imageGeneration.execute. "
                f"Ответ сервиса: {text[:220]}")
    hints = {
        401: "ключ не принят — проверьте YANDEX_API_KEY",
        402: "недостаточно средств на платёжном аккаунте",
        404: "модель или адрес не найдены — проверьте название модели и YANDEX_FOLDER_ID",
        429: "слишком много запросов, сервис просит подождать",
    }
    if code in hints:
        return f"HTTP {code}: {hints[code]}. Ответ сервиса: {text[:220]}"
    return f"{type(exc).__name__}: {text[:300]}"


def generate_image_legacy(prompt: str, seed: Optional[int] = None, *,
                          api_key: Optional[str] = None, folder_id: Optional[str] = None,
                          width_ratio: str = "1", height_ratio: str = "1",
                          poll_interval_s: float = POLL_INTERVAL_S,
                          timeout_s: float = POLL_TIMEOUT_S,
                          on_poll: Optional[Callable[[int, float], None]] = None,
                          post_json: Callable = _post_json, get_json: Callable = _get_json,
                          sleep: Callable[[float], None] = time.sleep) -> GenerationResult:
    """Способ из урока: асинхронная операция и опрос статуса.

    На сентябрь 2026 отвечает отказом — модель `yandex-art` недоступна. Оставлен, чтобы
    показать разницу и проверить, не вернулась ли она.
    """
    api_key = api_key or art_api_key()
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
#
# Промпты намеренно короткие. Первая версия была втрое длиннее и с перечислением деталей —
# модель размывала их в декоративную картинку вместо знака (сравнение сохранено в
# results/v1-длинные-промпты/). Для логотипа работает другое правило: один объект,
# два цвета, слово «логотип» в начале и явный запрет надписей в конце.
PROMPTS: list[Prompt] = [
    Prompt(
        key="shield-check",
        title="Щит с галочкой — основной знак",
        # «галочка» без пояснения рисовалась треугольником — добавлен английский термин
        text=("Логотип: синий щит, внутри белая галочка как знак подтверждения check mark. "
              "Плоский минималистичный векторный знак по центру на белом фоне. Без текста."),
        tags=["минимализм", "доверие", "медицина"],
    ),
    Prompt(
        key="document-seal",
        title="Документ с печатью — знак соответствия",
        text=("Логотип: белый лист документа с синей круглой печатью. "
              "Плоский минималистичный векторный знак по центру на белом фоне. Без текста."),
        tags=["официальный", "сертификация"],
    ),
    Prompt(
        key="tech-pulse",
        title="Пульс в круге — ассистент по нормам",
        text=("Логотип: бирюзовая линия кардиограммы внутри синего круга. "
              "Плоский минималистичный векторный знак по центру на белом фоне. Без текста."),
        tags=["технологии", "медтех"],
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
    ap.add_argument("--seed", type=int, help="зерно генерации (учитывается только в --legacy)")
    ap.add_argument("--model", help=f"модель, по умолчанию {DEFAULT_MODEL}")
    ap.add_argument("--size", default=DEFAULT_SIZE, help=f"размер, по умолчанию {DEFAULT_SIZE}")
    ap.add_argument("--legacy", action="store_true",
                    help="способ из урока (асинхронная операция, модель yandex-art) — сейчас не работает")
    ap.add_argument("--ratio", default="1:1", help="соотношение сторон для --legacy, например 1:1")
    ap.add_argument("--out", default="results", help="папка для картинок")
    ap.add_argument("--list", action="store_true", help="показать заготовленные промпты и выйти")
    ap.add_argument("--dry-run", action="store_true", help="собрать запрос и показать его, не отправляя")
    ap.add_argument("--check", action="store_true",
                    help="проверить доступ и понять, чего не хватает (генерацию не запускает)")
    args = ap.parse_args(argv)

    if args.check:
        return check_access()

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
    model = args.model or model_name()
    print(f"Модель: {'yandex-art (способ из урока)' if args.legacy else model}\n")

    ok = 0
    for p in jobs:
        print(f"=== {p.key}: {p.title}")
        print(f"Промпт ({len(p.text)} символов): {p.text}")
        if len(p.text) > PROMPT_LIMIT:
            print(f"  ВНИМАНИЕ: у модели предел {PROMPT_LIMIT} символов, лишнее будет отброшено")
        try:
            if args.legacy:
                def report(n: int, elapsed: float) -> None:
                    print(f"  ожидание… проверка {n}, прошло {elapsed:.0f} с")

                res = generate_image_legacy(p.text, args.seed, width_ratio=width,
                                            height_ratio=height, on_poll=report)
            else:
                res = generate_image(p.text, model=model, size=args.size)
        except YandexArtError as exc:
            print(f"  ОШИБКА: {exc}\n", file=sys.stderr)
            continue
        path = save_image(res, out_dir, f"{stamp}_{p.key}")
        ok += 1
        print(f"  готово за {res.elapsed_s} с, {res.size_kb:.0f} КБ")
        print(f"  сохранено: {path.relative_to(HERE)}\n")

    print(f"Успешно сгенерировано: {ok} из {len(jobs)}")
    return 0 if ok == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
