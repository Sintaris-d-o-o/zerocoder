# -*- coding: utf-8 -*-
"""Собирает markdown-таблицы метрик прямо из JSON-результатов прогонов.

Нужен, чтобы числа в отчёте не переписывались руками: results/tables.md
вставляется в результаты.md как есть.
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
OUT = R / "tables.md"
lines = []


def load(name):
    p = R / f"{name}_metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def block(name, title):
    d = load(name)
    if not d:
        lines.append(f"\n_Результатов для `{name}` нет._\n")
        return
    cfg, base, fin = d["config"], d.get("baseline") or {}, d["final"]

    lines.append(f"\n### {title}\n")
    lines.append("| Параметр прогона | Значение |")
    lines.append("|---|---|")
    total = f"{d['params_total']:,}".replace(",", " ")
    trainable = f"{d['params_trainable']:,}".replace(",", " ")
    lines.append(f"| Модель | `{d['model']}`, {total} параметров |")
    lines.append(f"| Обучаемых параметров | {trainable} (100 %, полный fine-tuning) |")
    lines.append(f"| Строк: обучение / валидация | {d['rows']['train']} / {d['rows']['validation']} |")
    lines.append(f"| Эпох | {cfg['epochs']:g} |")
    lines.append(f"| Batch size | {cfg['batch_size']} |")
    lines.append(f"| Learning rate | {cfg['lr']:g} |")
    lines.append(f"| Длина входа / выхода (токенов) | {cfg['max_input']} / {cfg['max_target']} |")
    lines.append(f"| Устройство | {d['device'].upper()} |")
    lines.append(f"| Время обучения | {d['train_time_sec'] / 60:.1f} мин |")
    lines.append(f"| Скорость | {d['train_samples_per_second']:.2f} примеров/с |")
    lines.append(f"| Память | {d['memory']} |")
    lines.append(f"| Итоговый training loss | {d['train_loss']:.4f} |")

    lines.append("\n| Метрика | До дообучения | После дообучения | Изменение |")
    lines.append("|---|---:|---:|---:|")
    rows = [("validation loss", "loss", 4, "ниже — лучше"),
            ("ROUGE-1", "rouge1", 2, None), ("ROUGE-2", "rouge2", 2, None),
            ("ROUGE-L", "rougeL", 2, None), ("ROUGE-Lsum", "rougeLsum", 2, None),
            ("gen_len (своя метрика)", "gen_len", 1, None)]
    for label, key, nd, note in rows:
        b, f = base.get(f"base_{key}"), fin.get(f"eval_{key}")
        if f is None:
            continue
        delta = f"{f - b:+.{nd}f}" if b is not None else "—"
        lines.append(f"| {label} | {fmt(b, nd)} | {fmt(f, nd)} | **{delta}** |")

    ev = [h for h in d["log_history"] if "eval_loss" in h]
    if len(ev) > 1:
        lines.append("\n| Эпоха | validation loss | ROUGE-1 | ROUGE-2 | ROUGE-L | gen_len |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for h in ev:
            lines.append(
                f"| {h['epoch']:.0f} | {h['eval_loss']:.4f} | {h.get('eval_rouge1', 0):.2f} | "
                f"{h.get('eval_rouge2', 0):.2f} | {h.get('eval_rougeL', 0):.2f} | {h.get('eval_gen_len', 0):.1f} |")


lines.append("<!-- Сгенерировано code/make_report_tables.py — не править руками -->")
block("xsum", "Эксперимент A — XSum (эталон задания)")
block("normassist", "Эксперимент B — собственный датасет из задания 8.1")

tok = R / "tokenizer_check.json"
if tok.exists():
    d = json.loads(tok.read_text(encoding="utf-8"))
    lines.append("\n### Проверка токенизатора T5-Small на 4 языках датасета 8.1\n")
    lines.append("| Язык | Строк | Токенов | Из них `<unk>` | Доля `<unk>` | Токенов на 100 символов |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    names = {"en": "английский", "de": "немецкий", "sl": "словенский", "ru": "русский"}
    base_density = d["en"]["tokens_per_100_chars"]
    for lang in ("en", "de", "sl", "ru"):
        x = d[lang]
        ratio = x["tokens_per_100_chars"] / base_density
        toks = f"{x['tokens']:,}".replace(",", " ")
        unks = f"{x['unk']:,}".replace(",", " ")
        lines.append(f"| {names[lang]} | {x['rows']} | {toks} | {unks} | **{x['unk_share']:.1%}** | "
                     f"{x['tokens_per_100_chars']:.0f} (×{ratio:.1f}) |")

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Готово: {OUT} ({len(lines)} строк)")
