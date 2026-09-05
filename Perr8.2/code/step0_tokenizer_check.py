# -*- coding: utf-8 -*-
"""Проверка токенизатора T5-Small на 4 языках датасета из задания 8.1.

Урок 8.2 начинается с тезиса: «Важно использовать токенизатор, который
соответствует исходному датасету». Этот скрипт превращает тезис в измерение:
сколько текста каждого языка токенизатор T5 просто не понимает (<unk>)
и во сколько раз раздувается длина последовательности.
"""
import json, sys, collections
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results"
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"J:/My Drive/Colab Notebooks/sintaris/normassist_out/data/raw_rows.jsonl")

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("t5-small")
unk_id = tok.unk_token_id
print(f"Токенизатор: t5-small | размер словаря: {tok.vocab_size} | <unk> id: {unk_id}")

rows = [json.loads(l) for l in DATA.open(encoding="utf-8")]
per_lang = collections.defaultdict(lambda: {"rows": 0, "chars": 0, "tokens": 0, "unk": 0})

for r in rows:
    text = r["q"] + " " + r["a"]
    ids = tok(text, add_special_tokens=False)["input_ids"]
    s = per_lang[r["lang"]]
    s["rows"] += 1
    s["chars"] += len(text)
    s["tokens"] += len(ids)
    s["unk"] += sum(1 for i in ids if i == unk_id)

print(f"\n{'язык':<6}{'строк':>8}{'токенов':>10}{'<unk>':>9}{'доля <unk>':>12}{'токенов/100 симв.':>20}")
print("-" * 65)
report = {}
for lang in ("en", "de", "sl", "ru"):
    s = per_lang[lang]
    unk_share = s["unk"] / s["tokens"]
    density = s["tokens"] / s["chars"] * 100
    report[lang] = {"rows": s["rows"], "tokens": s["tokens"], "unk": s["unk"],
                    "unk_share": round(unk_share, 4), "tokens_per_100_chars": round(density, 2)}
    print(f"{lang:<6}{s['rows']:>8}{s['tokens']:>10}{s['unk']:>9}{unk_share:>11.1%}{density:>20.1f}")

print("\nЧто это значит:")
print("  EN — родной язык модели, эталон плотности токенов.")
for lang, name in (("de", "немецкий"), ("sl", "словенский"), ("ru", "русский")):
    d = report[lang]
    ratio = d["tokens_per_100_chars"] / report["en"]["tokens_per_100_chars"]
    print(f"  {lang.upper()} ({name}): <unk> {d['unk_share']:.1%}, "
          f"длина последовательности x{ratio:.1f} относительно английского.")

# Наглядный пример: как выглядит русская фраза глазами токенизатора T5
ru = next(r for r in rows if r["lang"] == "ru")
sample = ru["q"][:90]
print(f"\nПример RU-текста:\n  {sample}")
print(f"  -> токены T5: {tok.tokenize(sample)[:20]} ...")
print(f"  -> обратное декодирование: {tok.decode(tok(sample, add_special_tokens=False)['input_ids'])[:90]!r}")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "tokenizer_check.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nСохранено: {OUT / 'tokenizer_check.json'}")
