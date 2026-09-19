"""Убирает из кэша аномально долгие вызовы, чтобы перезамерить их начисто.

Зачем: при прогоне на живой сети один запрос к gpt-5 провисел 7609 секунд (медиана — 36 с),
из-за чего суммарное время режима оказалось бессмысленным. Скрипт выбрасывает такие записи из
`results/cache.jsonl`; следующий запуск `run_experiment.py` повторит только их, всё остальное
возьмёт из кэша.

Запуск: python fix_outlier.py [--threshold 600] [--results results] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--threshold", type=float, default=600.0, help="секунд; дольше — считаем зависшим")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    path = HERE / args.results / "cache.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cloud = [r["result"]["latency_s"] for r in records if r["result"]["provider"] == "openai"]
    print(f"Всего записей: {len(records)}; облачных вызовов: {len(cloud)}, "
          f"медиана {statistics.median(cloud):.1f} с, максимум {max(cloud):.1f} с")

    slow = [r for r in records if r["result"]["latency_s"] > args.threshold]
    if not slow:
        print("Зависших вызовов нет — ничего не меняю.")
        return
    for r in slow:
        res = r["result"]
        print(f"  выброшен: роль {res['role']}, модель {res['model']}, {res['latency_s']:.0f} с, "
              f"выход {res['tokens_out']} токенов")
    if args.dry_run:
        print("--dry-run: файл не тронут.")
        return

    shutil.copy(path, path.with_suffix(".jsonl.bak"))
    keep = [r for r in records if r["result"]["latency_s"] <= args.threshold]
    with path.open("w", encoding="utf-8") as fh:
        for r in keep:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Удалено {len(slow)}; осталось {len(keep)}. Резервная копия: {path.name}.bak")
    print("Теперь запустите run_experiment.py заново — повторятся только удалённые вызовы.")


if __name__ == "__main__":
    main()
