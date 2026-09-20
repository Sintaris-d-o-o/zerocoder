"""Показывает, что произошло в последнем запуске цепочки (задание 10.5).

n8n хранит данные выполнения в формате flatted — это плоский массив, где вместо
вложенных объектов стоят номера элементов. Поэтому журнал не читается обычным
json.loads, и его приходится собирать обратно.

Запуск (переменные те же, что у deploy_workflow.py):
    python last_run.py
    python last_run.py --full     # показать текст поста целиком
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy_workflow import Client, BASE  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WORKFLOW_NAME = "Zerocoder 10.5 — Автопостинг в Telegram"


def unflatten(parsed: list):
    """Собирает обратно структуру, разложенную библиотекой flatted."""
    done: dict[int, object] = {}

    def walk(value):
        if isinstance(value, str) and value.isdigit() and int(value) < len(parsed):
            idx = int(value)
            if idx in done:
                return done[idx]
            done[idx] = None            # заглушка на случай ссылки на самого себя
            done[idx] = walk(parsed[idx])
            return done[idx]
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        return value

    return walk(parsed[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="печатать длинные значения целиком")
    args = ap.parse_args(argv)
    cut = 100000 if args.full else 300

    cli = Client(BASE)
    cli.login(os.environ["N8N_OWNER_EMAIL"], os.environ["N8N_OWNER_PASSWORD"])

    code, res = cli.call("/rest/workflows")
    data = res.get("data")
    items = data.get("data", []) if isinstance(data, dict) else (data or [])
    wf = next((w for w in items if w["name"] == WORKFLOW_NAME), None)
    if not wf:
        raise SystemExit(f"Цепочка «{WORKFLOW_NAME}» в этом n8n не найдена")

    flt = urllib.parse.quote(json.dumps({"workflowId": wf["id"]}))
    code, res = cli.call(f"/rest/executions?filter={flt}")
    data = res.get("data")
    runs = data.get("results", []) if isinstance(data, dict) else (data or [])
    if not runs:
        raise SystemExit("Запусков ещё не было")

    run = runs[0]
    print(f"Запуск {run['id']}: {run.get('status')}  ({run.get('startedAt')})\n")

    code, res = cli.call(f"/rest/executions/{run['id']}")
    raw = res["data"]["data"]
    result = unflatten(json.loads(raw))["resultData"] if isinstance(raw, str) else raw["resultData"]

    for name, node_runs in result.get("runData", {}).items():
        first = node_runs[0]
        error = (first.get("error") or {}).get("message")
        if error:
            print(f"✘ {name}\n    {str(error)[:cut]}")
            continue
        try:
            out = json.dumps(first["data"]["main"][0][0]["json"], ensure_ascii=False)
        except Exception:
            out = "(без данных)"
        print(f"✔ {name}\n    {out[:cut]}")

    print(f"\nостановились на: {result.get('lastNodeExecuted')}")
    err = (result.get("error") or {}).get("message")
    if err:
        print(f"ошибка: {str(err)[:cut]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
