"""Загрузка цепочки 10.6 в n8n через его REST API.

Делает то же, что руками в редакторе, но без кликов:
  • заводит ключ OpenAI;
  • заводит учётные данные Header Auth (`Authorization: Bearer <токен>`) для защиты вебхука;
  • создаёт или обновляет рабочий процесс, привязывает к узлам и то и другое;
  • включает процесс.

Настройки берутся из окружения или .env рядом со скриптом:
    N8N_BASE_URL       адрес n8n, по умолчанию http://127.0.0.1:5681
    N8N_OWNER_EMAIL    почта владельца
    N8N_OWNER_PASSWORD пароль владельца
    OPENAI_API_KEY     ключ, который пропишется в учётные данные n8n
    N8N_WEBHOOK_TOKEN  токен для заголовка Authorization

Запуск:
    python deploy_workflow.py             # загрузить и включить
    python deploy_workflow.py --off       # только загрузить, не включать
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = os.getenv("N8N_BASE_URL", "http://127.0.0.1:5681").rstrip("/")
OPENAI_CRED_NAME = "OpenAI — Sintaris"
HEADER_CRED_NAME = "Zerocoder 10.6 — Bearer"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class Client:
    """Тонкая обёртка над REST API n8n.

    Куку авторизации храним сами: n8n помечает её Secure, и стандартный
    cookiejar не отдаёт её обратно, когда ходим на http://127.0.0.1.
    """

    def __init__(self, base: str):
        self.base = base
        self.cookie = None

    def call(self, path: str, data=None, method: str | None = None):
        headers = {"Content-Type": "application/json", "browser-id": "deploy-script"}
        if self.cookie:
            headers["Cookie"] = self.cookie
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(data).encode() if data is not None else None,
            headers=headers,
            method=method or ("POST" if data is not None else "GET"))
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.headers.get("Set-Cookie")
                if raw and "n8n-auth=" in raw:
                    self.cookie = raw.split(";")[0]
                return resp.status, json.loads(resp.read().decode() or "{}")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode()[:400]

    def login(self, email: str, password: str) -> None:
        code, _ = self.call("/rest/login",
                            {"emailOrLdapLoginId": email, "password": password})
        if code != 200 or not self.cookie:
            raise SystemExit(f"Не удалось войти в n8n ({code}). Проверьте логин и пароль.")


def existing_credentials(cli: Client) -> list[dict]:
    code, res = cli.call("/rest/credentials")
    return (res.get("data") or []) if isinstance(res, dict) else []


def ensure_openai(cli: Client, api_key: str) -> dict:
    found = next((c for c in existing_credentials(cli) if c.get("type") == "openAiApi"), None)
    if found:
        print(f"ключ OpenAI уже заведён: {found['name']}")
        return found
    code, res = cli.call("/rest/credentials",
                         {"name": OPENAI_CRED_NAME, "type": "openAiApi",
                          "data": {"apiKey": api_key}})
    if code >= 300:
        raise SystemExit(f"Не создать учётные данные OpenAI: {code} {res}")
    print(f"создан ключ OpenAI: {OPENAI_CRED_NAME}")
    return res["data"]


def ensure_header_auth(cli: Client, token: str) -> dict:
    """Учётные данные для защиты вебхука.

    n8n сравнивает заголовок целиком, поэтому в значение кладётся строка
    вместе со словом Bearer — ровно то, что пришлёт клиент.
    """
    found = next((c for c in existing_credentials(cli)
                  if c.get("name") == HEADER_CRED_NAME), None)
    if found:
        print(f"защита вебхука уже заведена: {found['name']}")
        return found
    code, res = cli.call("/rest/credentials",
                         {"name": HEADER_CRED_NAME, "type": "httpHeaderAuth",
                          "data": {"name": "Authorization", "value": f"Bearer {token}"}})
    if code >= 300:
        raise SystemExit(f"Не создать учётные данные Header Auth: {code} {res}")
    print(f"создана защита вебхука: {HEADER_CRED_NAME}")
    return res["data"]


def upsert_workflow(cli: Client, wf: dict, openai_cred: dict, header_cred: dict) -> str:
    for node in wf["nodes"]:
        # ключ OpenAI нужен и узлам модели, и обычному HTTP-узлу, который сам ходит
        # в /v1/images/generations с предустановленным типом учётных данных
        uses_openai = (node["type"].endswith("langchain.openAi")
                       or node["type"].endswith("base.openAi")
                       or node["parameters"].get("nodeCredentialType") == "openAiApi")
        if uses_openai:
            node["credentials"] = {"openAiApi": {"id": openai_cred["id"],
                                                 "name": openai_cred["name"]}}
        if node["type"].endswith("base.webhook"):
            node["credentials"] = {"httpHeaderAuth": {"id": header_cred["id"],
                                                      "name": header_cred["name"]}}

    code, res = cli.call("/rest/workflows")
    data = res.get("data") if isinstance(res, dict) else []
    existing = data.get("data", []) if isinstance(data, dict) else (data or [])
    old = next((w for w in existing if w.get("name") == wf["name"]), None)

    body = {"name": wf["name"], "nodes": wf["nodes"],
            "connections": wf["connections"], "settings": wf.get("settings", {})}
    if old:
        body["versionId"] = old.get("versionId")
        code, res = cli.call(f"/rest/workflows/{old['id']}", body, method="PATCH")
        if code >= 300:
            raise SystemExit(f"Не обновить цепочку: {code} {res}")
        print(f"цепочка обновлена: {wf['name']}")
        return old["id"]

    code, res = cli.call("/rest/workflows", body)
    if code >= 300:
        raise SystemExit(f"Не создать цепочку: {code} {res}")
    print(f"цепочка создана: {wf['name']}")
    return res["data"]["id"]


def activate(cli: Client, wf_id: str) -> bool:
    """В n8n 2.x включается не процесс целиком, а конкретная его версия."""
    code, res = cli.call(f"/rest/workflows/{wf_id}")
    version = res["data"]["versionId"]
    code, res = cli.call(f"/rest/workflows/{wf_id}/activate", {"versionId": version}, "POST")
    if code >= 300:
        raise SystemExit(f"Не включить цепочку: {code} {res}")
    code, res = cli.call(f"/rest/workflows/{wf_id}")
    return bool(res["data"].get("active"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=str(HERE / "workflow.json"), help="файл цепочки")
    ap.add_argument("--off", action="store_true", help="загрузить, но не включать")
    args = ap.parse_args(argv)

    email = os.getenv("N8N_OWNER_EMAIL")
    password = os.getenv("N8N_OWNER_PASSWORD")
    api_key = os.getenv("OPENAI_API_KEY")
    token = os.getenv("N8N_WEBHOOK_TOKEN")
    if not (email and password and api_key and token):
        raise SystemExit("Нужны N8N_OWNER_EMAIL, N8N_OWNER_PASSWORD, OPENAI_API_KEY "
                         "и N8N_WEBHOOK_TOKEN в окружении или .env")

    wf = json.loads(Path(args.file).read_text(encoding="utf-8"))
    cli = Client(BASE)
    cli.login(email, password)
    print(f"вошли в n8n: {BASE}")

    openai_cred = ensure_openai(cli, api_key)
    header_cred = ensure_header_auth(cli, token)
    wf_id = upsert_workflow(cli, wf, openai_cred, header_cred)

    if args.off:
        print("цепочка не включена (--off)")
    else:
        print("включена:", activate(cli, wf_id))

    webhook = next(n for n in wf["nodes"] if n["type"].endswith("webhook"))
    print(f"\nадрес вебхука: {BASE}/webhook/{webhook['parameters']['path']}")
    print("защита: заголовок Authorization: Bearer <токен из N8N_WEBHOOK_TOKEN>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
