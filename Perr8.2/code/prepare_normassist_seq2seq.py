# -*- coding: utf-8 -*-
"""Готовит датасет задания 8.1 к формату seq2seq для T5-Small.

Вход:  raw_rows.jsonl из задания 8.1 (поля q, a, uc, lang, kind, ctx).
Выход: JSONL с полями document / summary — теми же именами, что у XSum,
       чтобы дальше работал ровно тот же код препроцессинга, что и в уроке.

Отбор строк идёт **по реальному алфавиту текста, а не по метке языка**.
Причина: словарь t5-small не содержит кириллицы (измерено в
step0_tokenizer_check.py), а в датасете 8.1 у строки с меткой lang="en"
контекст вполне может быть русским — нормативные выдержки ISO есть только
в русском переводе. Метка языка описывает язык *ответа*, а не язык входа.
"""
import json, argparse, random, collections
from pathlib import Path

DEFAULT_SRC = r"J:/My Drive/Colab Notebooks/sintaris/normassist_out/data/raw_rows.jsonl"

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=DEFAULT_SRC)
ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "normassist_seq2seq"))
ap.add_argument("--max-cyrillic", type=float, default=0.01,
                help="максимальная доля кириллицы в тексте, при которой строку ещё берём")
ap.add_argument("--drop-kinds", default="json",
                help="типы строк, которые не берём (у json-строк эталон — JSON-объект, ROUGE мерил бы скобки)")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

drop = set(a.drop_kinds.split(",")) if a.drop_kinds else set()


def cyrillic_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "\u0400" <= c <= "\u04ff") / len(letters)


rows = [json.loads(l) for l in Path(a.src).open(encoding="utf-8")]
print(f"Прочитано строк из датасета 8.1: {len(rows)}")
print(f"  по метке языка: {dict(collections.Counter(r['lang'] for r in rows))}")

kept, stats = [], collections.Counter()
for r in rows:
    ctx = "\n".join(f"[{i}] {c['text']}" for i, c in enumerate(r["ctx"], 1))
    document = f"QUESTION: {r['q']}\nCONTEXT:\n{ctx}"
    if r["kind"] in drop:
        stats["отброшено: тип строки"] += 1
        continue
    if cyrillic_share(document) > a.max_cyrillic or cyrillic_share(r["a"]) > a.max_cyrillic:
        stats["отброшено: кириллица в тексте"] += 1
        continue
    stats["взято"] += 1
    kept.append({"document": document, "summary": r["a"],
                 "uc": r["uc"], "kind": r["kind"], "lang": r["lang"]})

print("\nОтбор:")
for k, v in stats.most_common():
    print(f"  {k:<32} {v:>5}  ({v / len(rows):.0%})")
print(f"\nОсталось строк: {len(kept)} из {len(rows)} — это и есть та часть нашего датасета,")
print("которую англоязычный токенизатор T5 вообще способен прочитать без потерь.")
print(f"  по метке языка: {dict(collections.Counter(x['lang'] for x in kept))}")
print(f"  по типу:        {dict(collections.Counter(x['kind'] for x in kept))}")

random.Random(a.seed).shuffle(kept)
n_val = max(1, round(len(kept) * 0.1))
splits = {"validation": kept[:n_val], "train": kept[n_val:]}

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for name, part in splits.items():
    p = out / f"{name}.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for x in part:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"{name:<12} {len(part):>5} строк -> {p}")

s = splits["train"][0]
print("\nПример пары:")
print("  document:", s["document"][:220].replace("\n", " / "), "...")
print("  summary :", s["summary"][:180], "...")
