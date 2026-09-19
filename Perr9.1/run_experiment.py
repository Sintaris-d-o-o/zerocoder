"""Прогон эксперимента задания 9.1 по шагам урока.

Каждый шаг печатает свой вывод в консоль и одновременно сохраняет его в results/steps/NN_*.txt —
из этих файлов потом делаются PNG-«скриншоты» ключевых шагов (make_screenshots.py).

Запуск:
    python run_experiment.py            # полный прогон на всех вопросах
    python run_experiment.py --smoke    # быстрая проверка на трёх вопросах (S1, M1, H1)
    python run_experiment.py --no-judge # без экзаменатора
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

import agent as ag

HERE = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 70)

TOKENIZATION_PHRASES = [
    ("EN", "The certificate expires in 30 days."),
    ("RU", "Сертификат истекает через 30 дней."),
    ("DE", "Das Zertifikat läuft in 30 Tagen ab."),
    ("SL", "Certifikat poteče čez 30 dni."),
    ("RU translit", "Sertifikat istekaet cherez 30 dney."),
]
ANSWER_PREVIEW = 700   # сколько символов ответа показывать в выводе шага (полный текст — в answers.md)


# ----------------------------------------------------------------------------- инфраструктура

class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


class Context:
    """Всё состояние прогона: агент, вопросы, папка результатов, накопленные строки для CSV."""

    def __init__(self, results_dir: Path, queries: list[dict], use_cache: bool = True,
                 backends: Optional[ag.Backends] = None):
        self.results_dir = results_dir
        self.steps_dir = results_dir / "steps"
        self.steps_dir.mkdir(parents=True, exist_ok=True)
        self.queries = queries
        cache = ag.Cache(results_dir / "cache.jsonl" if use_cache else None)
        self.agent = ag.Agent(backends=backends, cache=cache)
        self.rows: list[dict] = []          # по одной строке на каждый вызов модели
        self.classification: dict[str, str] = {}
        self.mode_a: dict[str, ag.CallResult] = {}
        self.mode_b: dict[str, dict] = {}
        self.scores: dict[str, dict] = {}
        self.tokenization: list[dict] = []
        self.summary: dict = {}

    def record(self, step: str, mode: str, q: Optional[dict], res: ag.CallResult,
               predicted_tier: Optional[str] = None, extra: Optional[dict] = None) -> None:
        row = {
            "step": step, "mode": mode,
            "qid": q["id"] if q else "", "expected_tier": q["tier"] if q else "",
            "predicted_tier": predicted_tier or "",
            **res.to_dict(),
        }
        row.pop("answer")
        if extra:
            row.update(extra)
        self.rows.append(row)

    @contextmanager
    def step(self, number: int, slug: str, title: str):
        buf = io.StringIO()
        real = sys.stdout
        sys.stdout = _Tee(real, buf)
        header = f"ШАГ {number}. {title}"
        try:
            print("=" * 100)
            print(header)
            print("=" * 100)
            yield
        finally:
            sys.stdout = real
            path = self.steps_dir / f"{number:02d}_{slug}.txt"   # slug — только латиница, без ':<>\"/\\|?*'
            path.write_text(buf.getvalue(), encoding="utf-8")
            print(f"[сохранено: {path.relative_to(HERE)}]\n")


def load_queries(path: Path = HERE / "queries.json") -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["queries"]


def _fmt_usd(x: float) -> str:
    return f"${x:.5f}"


def _preview(text: str, n: int = ANSWER_PREVIEW) -> str:
    text = text.strip()
    return text if len(text) <= n else text[:n].rstrip() + f" … [ещё {len(text) - n} символов]"


# ----------------------------------------------------------------------------- шаги

def step0_tokenization(ctx: Context, with_ollama: bool = True) -> None:
    import tiktoken
    with ctx.step(0, "tokenization", "Токенизация: одна фраза на разных языках"):
        enc_new = tiktoken.get_encoding("o200k_base")    # токенизатор gpt-5 / gpt-4o
        enc_old = tiktoken.get_encoding("cl100k_base")   # токенизатор gpt-3.5 / gpt-4 (модели урока)
        rows = []
        for lang, phrase in TOKENIZATION_PHRASES:
            row = {"язык": lang, "фраза": phrase, "символов": len(phrase),
                   "токенов o200k (gpt-5)": len(enc_new.encode(phrase)),
                   "токенов cl100k (gpt-3.5/4)": len(enc_old.encode(phrase))}
            if with_ollama:
                try:
                    row[f"токенов {ag.OLLAMA_MODEL}"] = ctx.agent.backends.count_tokens_ollama(ag.OLLAMA_MODEL, phrase)
                except Exception as exc:  # noqa: BLE001 — шаг иллюстративный, не роняем прогон
                    row[f"токенов {ag.OLLAMA_MODEL}"] = f"ошибка: {type(exc).__name__}"
            rows.append(row)
        ctx.tokenization = rows
        print(pd.DataFrame(rows).to_string(index=False))
        print()
        print("Как токенизатор o200k режет фразы (каждый токен в |...|):")
        for lang, phrase in TOKENIZATION_PHRASES[:4]:
            pieces = [enc_new.decode([t]) for t in enc_new.encode(phrase)]
            print(f"  {lang:11s} " + "".join(f"|{p}|" for p in pieces))
        print()
        counts = [len(enc_new.encode(q["question"])) for q in ctx.queries]
        print(f"Тестовые вопросы ({len(counts)} шт., английский): "
              f"мин {min(counts)}, среднее {statistics.mean(counts):.1f}, макс {max(counts)} токенов; "
              f"всего {sum(counts)} токенов.")
        print("Вывод: русский, немецкий и словенский текст стоят в 1,5–3 раза больше токенов, чем английский;")
        print("новый токенизатор o200k экономнее старого cl100k на не-английских языках.")


def step1_config(ctx: Context) -> None:
    with ctx.step(1, "config", "Конфигурация ролей: модели, промпты, тарифы"):
        pr = ag.PRICING
        print(f"Тарифы: {pr.get('unit', '')}. Источник: {pr.get('source', '?')} (дата: {pr.get('retrieved', '?')})")
        rows = []
        for role, cfg in ctx.agent.roles.items():
            p = ag.price_for(cfg["model"])
            rows.append({"роль": role, "где работает": cfg["provider"], "модель": cfg["model"],
                         "лимит вывода": cfg["max_tokens"] or "—",
                         "reasoning": cfg["reasoning_effort"] or "по умолчанию",
                         "$/1M вход": p["input"], "$/1M выход": p["output"]})
        print(pd.DataFrame(rows).to_string(index=False))
        print()
        print("Системный промпт дежурного (intent):")
        print(ag.CLASSIFIER_PROMPT)
        print()
        print("Системный промпт отвечающих моделей:")
        print(ag.ANSWER_PROMPT)
        print()
        print(f"Ollama: {ag.OLLAMA_HOST}, модель {ag.OLLAMA_MODEL} (локально, цена 0)")


def step2_call_model(ctx: Context) -> None:
    with ctx.step(2, "call_model", "Функция вызова модели: пробные запросы (простой и сложный вопрос)"):
        by_id = {q["id"]: q for q in ctx.queries}
        demo = [qid for qid in ("S1", "H1") if qid in by_id] or [ctx.queries[0]["id"]]
        for qid in demo:
            q = by_id[qid]
            print(f"--- {qid} [{q['tier']}] {q['question']}")
            tier, intent = ctx.agent.classify(q["question"], namespace="B")
            ctx.record("2", "demo", q, intent, predicted_tier=tier)
            print(f"дежурный ({intent.model}, локально): '{intent.answer}' → {tier}   "
                  f"[{intent.latency_s:.1f} с, {intent.tokens_in}+{intent.tokens_out} токенов, {_fmt_usd(intent.cost_usd)}]")
            res = ctx.agent.call_model(q["question"], q["tier"], namespace="demo")
            ctx.record("2", "demo", q, res)
            print(f"ответ модели уровня '{q['tier']}' ({res.model}, {res.provider}):")
            print(_preview(res.answer))
            print(f"→ токены: вход {res.tokens_in}, выход {res.tokens_out} (из них рассуждения {res.reasoning_tokens}); "
                  f"стоимость {_fmt_usd(res.cost_usd)}; время {res.latency_s:.1f} с")
            print()


def step3_classifier(ctx: Context) -> None:
    with ctx.step(3, "classifier", "Проверка классификатора: корректно ли дежурный определяет уровень"):
        rows = []
        for q in ctx.queries:
            tier, res = ctx.agent.classify(q["question"], namespace="B")
            ctx.classification[q["id"]] = tier
            ctx.record("3", "B", q, res, predicted_tier=tier)
            rows.append({"id": q["id"], "ожидали": q["tier"], "дежурный ответил": res.answer,
                         "получили": tier, "верно": "✓" if tier == q["tier"] else "✗",
                         "время, с": res.latency_s, "вопрос": q["question"][:70]})
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        print()
        correct = int((df["ожидали"] == df["получили"]).sum())
        acc = correct / len(df)
        print(f"Точность: {correct} из {len(df)} = {acc:.0%}; среднее время классификации {df['время, с'].mean():.1f} с")
        print()
        print("Матрица ошибок (строки — ожидали, столбцы — получили):")
        print(pd.crosstab(df["ожидали"], df["получили"]).reindex(index=list(ag.TIERS), columns=list(ag.TIERS), fill_value=0))
        per_tier = {t: (df[df["ожидали"] == t]["получили"] == t).mean() for t in ag.TIERS if (df["ожидали"] == t).any()}
        print()
        print("Точность по уровням: " + ", ".join(f"{t}: {v:.0%}" for t, v in per_tier.items()))
        ctx.summary["classifier"] = {
            "accuracy": acc, "correct": correct, "total": len(df),
            "per_tier": {t: float((df[df['ожидали'] == t]['получили'] == t).mean()) for t in ag.TIERS if (df['ожидали'] == t).any()},
            "avg_latency_s": float(df["время, с"].mean()),
            "errors": [{"id": r["id"], "expected": r["ожидали"], "predicted": r["получили"]}
                       for r in rows if r["ожидали"] != r["получили"]],
        }


def _mode_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def step4_mode_a(ctx: Context) -> None:
    with ctx.step(4, "mode_A_strong_only", "Режим А: все вопросы — только в мощную модель (gpt-5)"):
        rows = []
        for q in ctx.queries:
            res = ctx.agent.call_model(q["question"], "hard", namespace="A")
            ctx.mode_a[q["id"]] = res
            ctx.record("4", "A", q, res)
            rows.append({"id": q["id"], "уровень": q["tier"], "модель": res.model,
                         "вход": res.tokens_in, "выход": res.tokens_out, "рассужд.": res.reasoning_tokens,
                         "стоимость $": round(res.cost_usd, 5), "время, с": res.latency_s})
        df = _mode_table(rows)
        print(df.to_string(index=False))
        print("-" * 100)
        print(f"ИТОГО режим А: стоимость {_fmt_usd(df['стоимость $'].sum())}, "
              f"время {df['время, с'].sum():.1f} с, токенов вход {df['вход'].sum()} / выход {df['выход'].sum()}")
        print()
        print(f"Пример ответа ({ctx.queries[0]['id']}):")
        print(_preview(ctx.mode_a[ctx.queries[0]["id"]].answer, 500))


def step5_mode_b(ctx: Context) -> None:
    with ctx.step(5, "mode_B_multimodel", "Режим Б: мультимодельность — дежурный + модель по уровню"):
        rows = []
        for q in ctx.queries:
            out = ctx.agent.route(q["question"], namespace="B")
            ctx.mode_b[q["id"]] = out
            ctx.record("5", "B", q, out["intent"], predicted_tier=out["tier"])
            ctx.record("5", "B", q, out["answer"], predicted_tier=out["tier"])
            a = out["answer"]
            rows.append({"id": q["id"], "ожидали": q["tier"], "дежурный": out["tier"],
                         "модель ответа": a.model, "вход": a.tokens_in, "выход": a.tokens_out,
                         "рассужд.": a.reasoning_tokens,
                         "стоимость $ (с дежурным)": round(out["cost_usd"], 5),
                         "время, с (с дежурным)": out["latency_s"]})
        df = _mode_table(rows)
        print(df.to_string(index=False))
        print("-" * 100)
        print(f"ИТОГО режим Б: стоимость {_fmt_usd(df['стоимость $ (с дежурным)'].sum())}, "
              f"время {df['время, с (с дежурным)'].sum():.1f} с, токенов вход {df['вход'].sum()} / выход {df['выход'].sum()}")
        by_model = df.groupby("модель ответа").agg(вопросов=("id", "count"),
                                                    стоимость=("стоимость $ (с дежурным)", "sum"),
                                                    время=("время, с (с дежурным)", "sum"))
        print()
        print("Распределение по моделям:")
        print(by_model.to_string())
        print()
        qid = next((q["id"] for q in ctx.queries if ctx.mode_b[q["id"]]["tier"] == "simple"), ctx.queries[0]["id"])
        print(f"Пример ответа локальной модели ({qid}, {ctx.mode_b[qid]['answer'].model}):")
        print(_preview(ctx.mode_b[qid]["answer"].answer, 500))


def step6_judge(ctx: Context) -> None:
    with ctx.step(6, "judge", "Экзаменатор: оценка качества ответов вслепую (1–5)"):
        rows = []
        for q in ctx.queries:
            sa, ra, res_a = ctx.agent.judge(q["question"], ctx.mode_a[q["id"]].answer, namespace="A")
            ctx.record("6", "A", q, res_a, extra={"score": sa})
            sb, rb, res_b = ctx.agent.judge(q["question"], ctx.mode_b[q["id"]]["answer"].answer, namespace="B")
            ctx.record("6", "B", q, res_b, extra={"score": sb})
            ctx.scores[q["id"]] = {"A": sa, "B": sb, "reason_A": ra, "reason_B": rb}
            rows.append({"id": q["id"], "уровень": q["tier"], "Б: модель": ctx.mode_b[q["id"]]["answer"].model,
                         "оценка А": sa, "оценка Б": sb, "комментарий к Б": (rb or "")[:60]})
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        print("-" * 100)
        missing = int(df["оценка А"].isna().sum() + df["оценка Б"].isna().sum())
        print(f"Средняя оценка: режим А {df['оценка А'].mean():.2f}, режим Б {df['оценка Б'].mean():.2f}"
              + (f"  (без оценки: {missing} ответов — экзаменатор не вернул число)" if missing else ""))
        print("По уровням (А / Б): " + ", ".join(
            f"{t}: {df[df['уровень'] == t]['оценка А'].mean():.2f} / {df[df['уровень'] == t]['оценка Б'].mean():.2f}"
            for t in ag.TIERS if (df["уровень"] == t).any()))
        judge_cost = sum(r["cost_usd"] for r in ctx.rows if r["step"] == "6")
        print(f"Стоимость работы экзаменатора (накладные расходы эксперимента, в сравнение А/Б не входит): {_fmt_usd(judge_cost)}")


def step7_summary(ctx: Context) -> dict:
    with ctx.step(7, "summary", "Сравнение: только мощная модель (А) против мультимодельности (Б)"):
        n = len(ctx.queries)
        a_cost = sum(r.cost_usd for r in ctx.mode_a.values())
        a_lat = sum(r.latency_s for r in ctx.mode_a.values())
        a_in = sum(r.tokens_in for r in ctx.mode_a.values())
        a_out = sum(r.tokens_out for r in ctx.mode_a.values())
        b_cost = sum(o["cost_usd"] for o in ctx.mode_b.values())
        b_lat = sum(o["latency_s"] for o in ctx.mode_b.values())
        b_in = sum(o["answer"].tokens_in + o["intent"].tokens_in for o in ctx.mode_b.values())
        b_out = sum(o["answer"].tokens_out + o["intent"].tokens_out for o in ctx.mode_b.values())
        a_list = [v["A"] for v in ctx.scores.values() if v["A"] is not None]
        b_list = [v["B"] for v in ctx.scores.values() if v["B"] is not None]
        has_scores = bool(a_list) and bool(b_list)
        a_score = statistics.mean(a_list) if a_list else None
        b_score = statistics.mean(b_list) if b_list else None

        table = pd.DataFrame([
            {"показатель": "стоимость, $", "А: только gpt-5": round(a_cost, 5), "Б: мультимодельность": round(b_cost, 5),
             "разница": f"{(b_cost - a_cost) / a_cost:+.0%}" if a_cost else "—"},
            {"показатель": "время, с (сумма)", "А: только gpt-5": round(a_lat, 1), "Б: мультимодельность": round(b_lat, 1),
             "разница": f"{(b_lat - a_lat) / a_lat:+.0%}" if a_lat else "—"},
            {"показатель": "время, с (среднее на вопрос)", "А: только gpt-5": round(a_lat / n, 1),
             "Б: мультимодельность": round(b_lat / n, 1), "разница": ""},
            {"показатель": "токенов вход / выход", "А: только gpt-5": f"{a_in} / {a_out}",
             "Б: мультимодельность": f"{b_in} / {b_out}", "разница": ""},
            {"показатель": "средняя оценка качества (1–5)", "А: только gpt-5": round(a_score, 2) if a_score else "—",
             "Б: мультимодельность": round(b_score, 2) if b_score else "—",
             "разница": f"{b_score - a_score:+.2f}" if has_scores else ""},
        ])
        print(table.to_string(index=False))
        print()
        by_model = {}
        for o in ctx.mode_b.values():
            m = o["answer"].model
            by_model.setdefault(m, {"вопросов": 0, "стоимость": 0.0, "время": 0.0})
            by_model[m]["вопросов"] += 1
            by_model[m]["стоимость"] += o["answer"].cost_usd
            by_model[m]["время"] += o["answer"].latency_s
        print("Режим Б — кто отвечал:")
        for m, v in by_model.items():
            print(f"  {m:12s} {v['вопросов']:2d} вопросов, {_fmt_usd(v['стоимость'])}, {v['время']:.0f} с")
        intent_lat = sum(o["intent"].latency_s for o in ctx.mode_b.values())
        print(f"  дежурный ({ag.OLLAMA_MODEL}): {n} классификаций, $0, {intent_lat:.0f} с")
        print()
        errs = [(qid, ctx.queries[i]["tier"], o["tier"]) for i, (qid, o) in enumerate(ctx.mode_b.items())
                if o["tier"] != ctx.queries[i]["tier"]]
        if errs:
            print("Ошибки маршрутизации (id: ожидали → получили; оценка А / Б):")
            for qid, exp, got in errs:
                s = ctx.scores.get(qid, {})
                print(f"  {qid}: {exp} → {got}; оценка {s.get('A', '—')} / {s.get('B', '—')}")
        else:
            print("Ошибок маршрутизации нет.")
        print()
        saving = (a_cost - b_cost) / a_cost if a_cost else 0
        verdict = (f"Мультимодельность дешевле на {saving:.0%}"
                   + (f" при среднем качестве {b_score:.2f} против {a_score:.2f}" if has_scores else "")
                   + f"; время {'меньше' if b_lat < a_lat else 'больше'} на {abs(b_lat - a_lat) / a_lat:.0%}"
                   + " (локальная модель работает на CPU ноутбука).")
        print("ВЫВОД: " + verdict)

        ctx.summary.update({
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "n_queries": n,
            "models": {r: ctx.agent.roles[r]["model"] for r in ctx.agent.roles},
            "pricing": ag.PRICING,
            "mode_A": {"cost_usd": a_cost, "latency_s": a_lat, "tokens_in": a_in, "tokens_out": a_out, "avg_score": a_score},
            "mode_B": {"cost_usd": b_cost, "latency_s": b_lat, "tokens_in": b_in, "tokens_out": b_out, "avg_score": b_score,
                       "by_model": by_model, "intent_latency_s": intent_lat,
                       "routing_errors": [{"id": e[0], "expected": e[1], "predicted": e[2]} for e in errs]},
            "saving_pct": saving * 100,
            "judge_cost_usd": sum(r["cost_usd"] for r in ctx.rows if r["step"] == "6"),
            "scores": ctx.scores,
            "tokenization": ctx.tokenization,
            "verdict": verdict,
        })
        return ctx.summary


def save_results(ctx: Context) -> None:
    (ctx.results_dir / "summary.json").write_text(json.dumps(ctx.summary, ensure_ascii=False, indent=2, default=str),
                                                  encoding="utf-8")
    pd.DataFrame(ctx.rows).to_csv(ctx.results_dir / "runs.csv", index=False, encoding="utf-8-sig")
    lines = ["# Все ответы: режим А (только gpt-5) и режим Б (мультимодельность)", ""]
    for q in ctx.queries:
        a = ctx.mode_a.get(q["id"])
        b = ctx.mode_b.get(q["id"])
        s = ctx.scores.get(q["id"], {})
        lines += [f"## {q['id']} [{q['tier']}] {q['question']}", ""]
        if a:
            lines += [f"**Режим А — {a.model}** (вход {a.tokens_in}, выход {a.tokens_out}, {_fmt_usd(a.cost_usd)}, "
                      f"{a.latency_s:.1f} с; оценка {s.get('A', '—')}: {s.get('reason_A', '')})", "", a.answer, ""]
        if b:
            ans = b["answer"]
            lines += [f"**Режим Б — дежурный сказал `{b['tier']}` → {ans.model}** (вход {ans.tokens_in}, выход {ans.tokens_out}, "
                      f"{_fmt_usd(b['cost_usd'])} с дежурным, {b['latency_s']:.1f} с; оценка {s.get('B', '—')}: "
                      f"{s.get('reason_B', '')})", "", ans.answer, ""]
    (ctx.results_dir / "answers.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Сохранено: {ctx.results_dir / 'summary.json'}, runs.csv, answers.md")


# ----------------------------------------------------------------------------- main

def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="только вопросы S1, M1, H1")
    ap.add_argument("--limit", type=int, default=0, help="взять первые N вопросов")
    ap.add_argument("--no-judge", action="store_true", help="пропустить экзаменатора")
    ap.add_argument("--no-cache", action="store_true", help="не использовать кэш повторных запросов")
    ap.add_argument("--no-ollama-tokens", action="store_true", help="в шаге 0 не считать токены через Ollama")
    ap.add_argument("--results", default="results", help="папка результатов (относительно папки задания)")
    args = ap.parse_args(argv)

    queries = load_queries()
    if args.smoke:
        queries = [q for q in queries if q["id"] in ("S1", "M1", "H1")]
    if args.limit:
        queries = queries[: args.limit]

    ctx = Context(HERE / args.results, queries, use_cache=not args.no_cache)
    print(f"Вопросов: {len(queries)}; результаты → {ctx.results_dir}\n")
    step0_tokenization(ctx, with_ollama=not args.no_ollama_tokens)
    step1_config(ctx)
    step2_call_model(ctx)
    step3_classifier(ctx)
    step4_mode_a(ctx)
    step5_mode_b(ctx)
    if not args.no_judge:
        step6_judge(ctx)
    step7_summary(ctx)
    save_results(ctx)


if __name__ == "__main__":
    main()
