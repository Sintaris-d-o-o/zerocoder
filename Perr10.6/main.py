"""Запуск автопостинга с картинкой: шлём тему поста на защищённый вебхук n8n (задание 10.6).

Цепочка в n8n принимает POST с темой, пишет текст, придумывает задание художнику,
рисует картинку и публикует её в Telegram-канал подписью к фото.

Вебхук закрыт заголовком `Authorization: Bearer <токен>`. Без него n8n отвечает 403 —
это можно показать флагом `--no-token`.

Настройки берутся из .env в корне репозитория или рядом со скриптом:
    N8N_POST_WEBHOOK  — адрес вебхука цепочки
    N8N_WEBHOOK_TOKEN — токен, который уходит в заголовке Authorization

Запуск:
    python main.py                                  # тема по умолчанию
    python main.py "Маркировка CE"                  # своя тема
    python main.py --no-token "Маркировка CE"       # показать ошибку 403
    python main.py --dry-run "Маркировка CE"        # показать запрос, не отправляя
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

DEFAULT_WEBHOOK = "http://localhost:5681/webhook/zerocoder-autopost-image"
DEFAULT_TOPIC = "Что проверяет нотифицированный орган при сертификации медицинского изделия"
TIMEOUT_S = 240   # текст, задание художнику и сама картинка — до четырёх минут


def webhook_url() -> str:
    return os.getenv("N8N_POST_WEBHOOK", "") or DEFAULT_WEBHOOK


def post_topic(topic: str, *, style: str = "", url: Optional[str] = None,
               token: Optional[str] = None, send_token: bool = True,
               timeout: int = TIMEOUT_S,
               opener=urllib.request.urlopen) -> tuple[int, dict | str]:
    """Отправляет тему на вебхук. Возвращает код ответа и разобранное тело.

    `send_token=False` — намеренно уйти без заголовка: так проверяется, что защита
    работает и n8n отвечает 403.
    `opener` вынесен в параметр — его подменяют тесты, и логика проверяется без n8n.
    """
    url = url or webhook_url()
    payload: dict = {"topic": topic}
    if style:
        payload["style"] = style

    headers = {"Content-Type": "application/json"}
    if send_token:
        value = token if token is not None else os.getenv("N8N_WEBHOOK_TOKEN", "")
        if value:
            headers["Authorization"] = f"Bearer {value}"

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
    if code == 200:
        mark = "✔ успех"
    elif code in (401, 403):
        mark = "✔ доступ закрыт — защита сработала"
    else:
        mark = "✘ неожиданный код"
    print(f"Код ответа: {code}", mark)

    if isinstance(body, str):
        print("Ответ (текст):")
        print(body[:1000])
        return

    print("Ответ (JSON):")
    print(json.dumps(body, ensure_ascii=False, indent=2)[:1500])

    if body.get("ok") or body.get("published"):
        parts = [f"идентификатор сообщения {body['message_id']}"] if body.get("message_id") else []
        if body.get("with_image"):
            parts.append("с картинкой")
        if body.get("text_sent_separately"):
            parts.append("текст ушёл отдельным сообщением: не поместился в подпись")
        print("\nПост опубликован" + (", " + ", ".join(parts) if parts else ""))
        if body.get("url"):
            print(f"Ссылка: {body['url']}")
    elif body.get("error"):
        print(f"\nЦепочка вернула ошибку: {body['error']}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("topic", nargs="?", default=DEFAULT_TOPIC, help="тема поста")
    ap.add_argument("--topic", dest="topic_opt", help="тема поста (то же самое, но флагом)")
    ap.add_argument("--style", default="", help="подсказка по стилю, например «деловой»")
    ap.add_argument("--url", help=f"адрес вебхука, по умолчанию из .env или {DEFAULT_WEBHOOK}")
    ap.add_argument("--no-token", action="store_true",
                    help="отправить без заголовка Authorization — должен вернуться 403")
    ap.add_argument("--dry-run", action="store_true", help="показать запрос, не отправляя")
    args = ap.parse_args(argv)

    topic = args.topic_opt or args.topic
    url = args.url or webhook_url()
    token = os.getenv("N8N_WEBHOOK_TOKEN", "")

    print(f"Вебхук: {url}")
    print(f"Тема:   {topic}")
    if args.style:
        print(f"Стиль:  {args.style}")
    if args.no_token:
        print("Токен:  не отправляем (проверяем защиту)")
    else:
        print(f"Токен:  {'Authorization: Bearer …' if token else 'не задан в .env'}")

    if args.dry_run:
        payload = {"topic": topic}
        if args.style:
            payload["style"] = args.style
        print("\nТело запроса:")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print("\n--dry-run: запрос не отправлялся.")
        return 0

    print("\nОтправляем запрос. Текст, задание художнику и картинка — до нескольких минут…")
    started = time.perf_counter()
    code, body = post_topic(topic, style=args.style, url=url,
                            send_token=not args.no_token)
    print(f"Ответ пришёл за {time.perf_counter() - started:.1f} с\n")
    describe(code, body)

    # без токена правильный исход — именно 403, поэтому он считается успехом
    if args.no_token:
        return 0 if code in (401, 403) else 1
    return 0 if code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
