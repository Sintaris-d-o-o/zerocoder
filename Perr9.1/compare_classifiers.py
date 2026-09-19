"""Сравнение локальных моделей в роли дежурного: точность и время классификации 24 вопросов.

Обе модели работают через Ollama (цена 0), поэтому сравнение бесплатное.
Результат: results/classifier_models.txt и results/classifier_models.json.

Запуск: python compare_classifiers.py gemma4:12b gemma4:e2b
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import agent as ag
from run_experiment import load_queries

HERE = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def evaluate(model: str, queries: list[dict], cache: ag.Cache) -> dict:
    roles = {"intent": {**ag.ROLES["intent"], "model": model}}
    agent = ag.Agent(cache=cache, roles=roles)
    rows = []
    for q in queries:
        tier, res = agent.classify(q["question"], namespace="B")
        rows.append({"id": q["id"], "expected": q["tier"], "predicted": tier, "raw": res.answer,
                     "latency_s": res.latency_s, "tokens_in": res.tokens_in})
        print(f"  {model:12s} {q['id']:3s} ожидали {q['tier']:6s} → {tier:6s} {'✓' if tier == q['tier'] else '✗'}  {res.latency_s:5.1f} с")
    acc = sum(r["expected"] == r["predicted"] for r in rows) / len(rows)
    return {"model": model, "accuracy": acc, "avg_latency_s": statistics.mean(r["latency_s"] for r in rows),
            "errors": [r for r in rows if r["expected"] != r["predicted"]], "rows": rows}


def main(models: list[str]) -> None:
    queries = load_queries()
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    cache = ag.Cache(results_dir / "cache.jsonl")   # тот же кэш, что у основного прогона
    out = []
    for m in models:
        print(f"Модель {m}:")
        out.append(evaluate(m, queries, cache))
    lines = ["Сравнение локальных моделей в роли дежурного (классификация 24 вопросов, Ollama, CPU)", ""]
    lines.append(f"{'модель':14s} {'точность':>9s} {'среднее время, с':>17s}  ошибки (id: ожидали→получили)")
    for r in out:
        errs = ", ".join(f"{e['id']}: {e['expected']}→{e['predicted']}" for e in r["errors"]) or "нет"
        lines.append(f"{r['model']:14s} {r['accuracy']:>9.0%} {r['avg_latency_s']:>17.1f}  {errs}")
    text = "\n".join(lines)
    print()
    print(text)
    (results_dir / "classifier_models.txt").write_text(text + "\n", encoding="utf-8")
    (results_dir / "classifier_models.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or ["gemma4:12b", "gemma4:e2b"])
