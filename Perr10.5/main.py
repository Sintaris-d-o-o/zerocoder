"""Запуск автопостинга: шлём тему поста на вебхук n8n (задание 10.5).

Цепочка в n8n принимает POST с темой, генерирует текст моделью и публикует его в
Telegram-канал. Скрипт отправляет запрос и показывает, что вернулось.

Настройки берутся из .env в корне репозитория:
    N8N_POST_WEBHOOK  — адрес вебхука цепочки
    N8N_WEBHOOK_TOKEN — необязательный токен, если вебхук закрыт заголовком

Запуск:
    python main.py                          # тема по умолчанию
    python main.py "AI в маркетинге"        # своя тема
    python main.py --topic "..." --style деловой --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env")
load_dotenv(HERE / ".env")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_WEBHOOK = "http://localhost:5678/webhook/zerocoder-autopost"
DEFAULT_TOPIC = "Как проверить срок действия сертификата медицинского изделия"
TIMEOUT_S = 120   # генерация текста моделью занимает десятки секунд


def webhook_url() -> str:
    return os.getenv("N8N_POST_WEBHOOK", "") or DEFAULT_WEBHOOK


def post_topic(topic: str, *, style: str = "", url: Optional[str] = None,
               timeout: int = TIMEOUT_S, opener=urllib.request.urlopen) -> tuple[int, dict | str]:
    """Отправляет тему на вебхук. Возвращает код ответа и разобранное тело.

    `opener` вынесен в параметр — так его подменяют тесты, и логика проверяется без n8n.
    """
    url = url or webhook_url()
    payload: dict = {"topic": topic}
    if style:
        payload["style"] = style

    headers = {"Content-Type": "application/json"}
    token = os.getenv("N8N_WEBHOOK_TOKEN", "")
    if token:
        headers["X-Webhook-Secret"] = token

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers=headers, method="POST")
    try:
        with opener(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            code = getattr(resp, "status", 200)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:500]
    except urllib.error.URLError as exc:
        raise SystemExit(f"Не достучались до n8n по адресу {url}: {exc.reason}\n"
                         f"Проверьте, что контейнер запущен и цепочка активна.") from exc

    try:
        return code, json.loads(raw)
    except json.JSONDecodeError:
        return code, raw


def describe(code: int, body: dict | str) -> None:
    """Печатает результат так, чтобы по выводу было видно, дошёл ли пост до канала."""
    print(f"Код ответа: {code}", "✔ успех" if code == 200 else "✘ не 200")
    if isinstance(body, str):
        print("Ответ (текст):")
        print(body[:1000])
        return

    print("Ответ (JSON):")
    print(json.dumps(body, ensure_ascii=False, indent=2)[:1500])

    # Цепочка возвращает признак публикации и ссылку на сообщение
    if body.get("ok") or body.get("published"):
        link = body.get("url") or body.get("link")
        mid = body.get("message_id")
        print("\nПост опубликован" + (f", идентификатор сообщения {mid}" if mid else "")
              + (f"\nСсылка: {link}" if link else ""))
    elif body.get("error"):
        print(f"\nЦепочка вернула ошибку: {body['error']}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("topic", nargs="?", default=DEFAULT_TOPIC, help="тема поста")
    ap.add_argument("--topic", dest="topic_opt", help="тема поста (то же самое, но флагом)")
    ap.add_argument("--style", default="", help="подсказка по стилю, например «деловой»")
    ap.add_argument("--url", help=f"адрес вебхука, по умолчанию из .env или {DEFAULT_WEBHOOK}")
    ap.add_argument("--dry-run", action="store_true", help="показать запрос, не отправляя")
    args = ap.parse_args(argv)

    topic = args.topic_opt or args.topic
    url = args.url or webhook_url()

    print(f"Вебхук: {url}")
    print(f"Тема:   {topic}")
    if args.style:
        print(f"Стиль:  {args.style}")

    if args.dry_run:
        payload = {"topic": topic}
        if args.style:
            payload["style"] = args.style
        print("\nТело запроса:")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print("\n--dry-run: запрос не отправлялся.")
        return 0

    print("\nОтправляем запрос, генерация занимает до минуты…")
    started = time.perf_counter()
    code, body = post_topic(topic, style=args.style, url=url)
    print(f"Ответ пришёл за {time.perf_counter() - started:.1f} с\n")
    describe(code, body)
    return 0 if code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
