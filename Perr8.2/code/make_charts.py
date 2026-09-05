# -*- coding: utf-8 -*-
"""Графики по результатам задания 8.2. Читает JSON, оставленный finetune_t5.py."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dedcd6"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 130,
})


def tidy(ax, ylabel=None):
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    if ylabel:
        ax.set_ylabel(ylabel)


def load(name):
    p = R / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ---------------------------------------------------------------- токенизатор
def chart_tokenizer():
    d = load("tokenizer_check.json")
    if not d:
        return
    langs = ["en", "de", "sl", "ru"]
    names = {"en": "английский", "de": "немецкий", "sl": "словенский", "ru": "русский"}
    unk = [d[l]["unk_share"] * 100 for l in langs]
    dens = [d[l]["tokens_per_100_chars"] for l in langs]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    fig.suptitle("Токенизатор t5-small на данных из задания 8.1: чем дальше от английского, тем хуже",
                 fontsize=12, fontweight="bold", y=0.99)

    ax = axes[0]
    colors = [AQUA if l != "ru" else ORANGE for l in langs]
    bars = ax.bar([names[l] for l in langs], unk, color=colors, width=0.62)
    for b, v in zip(bars, unk):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", color=INK, fontsize=10)
    ax.set_title("Доля потерянных символов (<unk>)", loc="left")
    ax.set_ylim(0, max(unk) * 1.25)
    tidy(ax, "% токенов, ставших <unk>")

    ax = axes[1]
    base = dens[0]
    bars = ax.bar([names[l] for l in langs], dens, color=[BLUE] * 4, width=0.62)
    for b, v in zip(bars, dens):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}  (x{v / base:.1f})",
                ha="center", color=INK, fontsize=9.5)
    ax.set_title("Во что обходится текст: токенов на 100 символов", loc="left")
    ax.set_ylim(0, max(dens) * 1.25)
    tidy(ax, "токенов / 100 символов")

    fig.text(0.012, 0.015,
             "Метка языка в датасете 8.1 описывает язык ответа: у части «английских» строк контекст —"
             " русские выдержки ISO, отсюда 3,8 % <unk> даже у EN.",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    out = R / "tokenizer_unk.png"
    fig.savefig(out); plt.close(fig)
    print("  ", out.name)


# ---------------------------------------------------------------- обучение
def chart_training(dataset, title):
    d = load(f"{dataset}_metrics.json")
    if not d:
        return
    hist = d["log_history"]
    tr = [(h["epoch"], h["loss"]) for h in hist if "loss" in h and "eval_loss" not in h]
    ev = [(h["epoch"], h["eval_loss"]) for h in hist if "eval_loss" in h]
    rg = [(h["epoch"], h.get("eval_rouge1"), h.get("eval_rouge2"), h.get("eval_rougeL"))
          for h in hist if "eval_rouge1" in h]
    if not tr:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.99)

    ax = axes[0]
    ax.plot(*zip(*tr), color=BLUE, lw=2, label="обучающая выборка")
    if ev:
        ax.plot(*zip(*ev), color=ORANGE, lw=2, marker="o", ms=7, label="валидационная выборка")
        for e, v in ev:
            ax.annotate(f"{v:.3f}", (e, v), textcoords="offset points", xytext=(0, 9),
                        ha="center", color=INK, fontsize=9)
    ax.set_title("Функция потерь (loss) — чем ниже, тем лучше", loc="left")
    ax.set_xlabel("эпоха")
    ax.legend(frameon=False, loc="upper right")
    tidy(ax, "loss")

    ax = axes[1]
    if rg:
        for vals, color, name in ((1, BLUE, "ROUGE-1"), (2, ORANGE, "ROUGE-2"), (3, AQUA, "ROUGE-L")):
            xs = [r[0] for r in rg]
            ys = [r[vals] for r in rg]
            ax.plot(xs, ys, color=color, lw=2, marker="o", ms=7, label=name)
            ax.annotate(f"{ys[-1]:.1f}", (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(8, -3), color=INK, fontsize=9)
        ax.legend(frameon=False)
        ax.set_xticks([r[0] for r in rg])
    ax.set_title("ROUGE на валидации — чем выше, тем лучше", loc="left")
    ax.set_xlabel("эпоха")
    tidy(ax, "ROUGE, %")

    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    out = R / f"{dataset}_training.png"
    fig.savefig(out); plt.close(fig)
    print("  ", out.name)


# ---------------------------------------------------------------- до / после
def chart_before_after():
    sets = [(n, load(f"{n}_metrics.json"), t) for n, t in
            (("xsum", "XSum (эталон задания)"), ("normassist", "Наши данные из 8.1"))]
    sets = [(n, d, t) for n, d, t in sets if d and d.get("baseline")]
    if not sets:
        return
    metrics = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    fig, axes = plt.subplots(1, len(sets), figsize=(5.6 * len(sets), 4.2), squeeze=False)
    fig.suptitle("ROUGE до и после дообучения T5-Small", fontsize=12, fontweight="bold", y=0.99)

    for ax, (name, d, title) in zip(axes[0], sets):
        before = [d["baseline"].get(f"base_{m}", 0) for m in metrics]
        after = [d["final"].get(f"eval_{m}", 0) for m in metrics]
        x = range(len(metrics))
        w = 0.38
        b1 = ax.bar([i - w / 2 - 0.01 for i in x], before, w, color=BLUE, label="до дообучения")
        b2 = ax.bar([i + w / 2 + 0.01 for i in x], after, w, color=ORANGE, label="после дообучения")
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4,
                        f"{b.get_height():.1f}", ha="center", color=INK, fontsize=9)
        ax.set_xticks(list(x)); ax.set_xticklabels(["ROUGE-1", "ROUGE-2", "ROUGE-L", "ROUGE-Lsum"])
        ax.set_ylim(0, max(before + after) * 1.22)
        ax.set_title(title, loc="left")
        ax.legend(frameon=False, loc="upper right")
        tidy(ax, "ROUGE, %")

    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    out = R / "rouge_before_after.png"
    fig.savefig(out); plt.close(fig)
    print("  ", out.name)


if __name__ == "__main__":
    print("Строю графики:")
    chart_tokenizer()
    chart_training("xsum", "Дообучение T5-Small на XSum (эталон задания)")
    chart_training("normassist", "Дообучение T5-Small на собственных данных из задания 8.1")
    chart_before_after()
    print("Готово.")
