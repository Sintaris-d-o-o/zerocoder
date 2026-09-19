"""«Скриншоты» ключевых шагов и графики сравнения по результатам run_experiment.py.

- results/steps/NN_*.txt  →  results/steps/NN_*.png  (вывод шага, отрисованный как окно терминала)
- results/summary.json + runs.csv  →  results/comparison.png, cost_per_question.png,
  classifier_confusion.png, tokenization.png

Запуск: python make_screenshots.py [--results results]
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

HERE = Path(__file__).resolve().parent

# Палитра (валидированная категориальная палитра из skill dataviz, light mode)
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
MODEL_COLORS = {"gpt-5": C_BLUE, "gpt-5-mini": C_ORANGE, "local": C_AQUA, "intent": C_YELLOW}


def model_color(model: str) -> str:
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    return MODEL_COLORS["local"]   # gemma4:* и другие локальные


# ----------------------------------------------------------------------------- текст → PNG

def _font(size: int) -> ImageFont.FreeTypeFont:
    for cand in [Path("C:/Windows/Fonts/consola.ttf"), Path("C:/Windows/Fonts/cour.ttf")]:
        if cand.exists():
            return ImageFont.truetype(str(cand), size)
    from matplotlib import font_manager
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans Mono"), size)


def render_text_png(txt_path: Path, max_cols: int = 150, max_lines: int = 110) -> list[Path]:
    """Рисует текстовый вывод шага как окно терминала. Длинные шаги режутся на страницы."""
    raw = txt_path.read_text(encoding="utf-8").splitlines()
    lines: list[str] = []
    for ln in raw:
        lines.extend(textwrap.wrap(ln, max_cols, replace_whitespace=False, drop_whitespace=False) or [""])
    pages = [lines[i:i + max_lines] for i in range(0, len(lines), max_lines)] or [[""]]
    font, title_font = _font(17), _font(15)
    ch_w = font.getbbox("M")[2]
    line_h = 22
    outputs = []
    for p_idx, page in enumerate(pages):
        width = ch_w * min(max(len(ln) for ln in page) + 2, max_cols + 2) + 40
        width = max(width, 900)
        height = 44 + line_h * len(page) + 30
        img = Image.new("RGB", (width, height), "#1e1e1e")
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, width, 36], fill="#2d2d2d")
        for i, col in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
            d.ellipse([14 + i * 22, 11, 28 + i * 22, 25], fill=col)
        suffix = f"  (стр. {p_idx + 1}/{len(pages)})" if len(pages) > 1 else ""
        d.text((90, 9), f"python run_experiment.py — {txt_path.stem}{suffix}", font=title_font, fill="#cfcfcf")
        y = 50
        for ln in page:
            color = "#d4d4d4"
            if ln.startswith("ШАГ ") or ln.startswith("ИТОГО") or ln.startswith("ВЫВОД"):
                color = "#ffd166"
            elif ln.startswith("=") or ln.startswith("-"):
                color = "#6b6b6b"
            elif ln.startswith("[сохранено"):
                color = "#7fbf7f"
            d.text((20, y), ln, font=font, fill=color)
            y += line_h
        out = txt_path.with_suffix("").with_name(txt_path.stem + (f"_p{p_idx + 1}" if len(pages) > 1 else "") + ".png")
        img.save(out)
        outputs.append(out)
    return outputs


# ----------------------------------------------------------------------------- графики

def _style(ax, title: str, ylabel: str = ""):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart_comparison(summary: dict, out: Path) -> None:
    A, B = summary["mode_A"], summary["mode_B"]
    n = summary["n_queries"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=SURFACE)
    fig.suptitle(f"Только gpt-5 (А) против мультимодельности (Б), {n} вопросов", x=0.01, ha="left",
                 fontsize=14, color=INK)

    # --- стоимость, $
    ax = axes[0]
    ax.bar(["А"], [A["cost_usd"]], color=C_BLUE, width=0.55, label="gpt-5")
    bottom = 0.0
    for model, v in B["by_model"].items():
        ax.bar(["Б"], [v["стоимость"]], bottom=bottom, color=model_color(model), width=0.55,
               label=model if model != "gpt-5" else None, edgecolor=SURFACE, linewidth=2)
        bottom += v["стоимость"]
    ax.text(0, A["cost_usd"], f"${A['cost_usd']:.3f}", ha="center", va="bottom", color=INK)
    ax.text(1, B["cost_usd"], f"${B['cost_usd']:.3f}  (−{summary['saving_pct']:.0f} %)", ha="center", va="bottom", color=INK)
    _style(ax, "Стоимость всех ответов, $")
    ax.set_ylim(0, max(A["cost_usd"], B["cost_usd"]) * 1.25)
    ax.legend(frameon=False, fontsize=9)

    # --- время, с
    ax = axes[1]
    ax.bar(["А"], [A["latency_s"]], color=C_BLUE, width=0.55, label="gpt-5")
    bottom = 0.0
    for model, v in B["by_model"].items():
        ax.bar(["Б"], [v["время"]], bottom=bottom, color=model_color(model), width=0.55,
               label=model if model != "gpt-5" else None, edgecolor=SURFACE, linewidth=2)
        bottom += v["время"]
    ax.bar(["Б"], [B["intent_latency_s"]], bottom=bottom, color=C_YELLOW, width=0.55, label="дежурный (локально)",
           edgecolor=SURFACE, linewidth=2)
    ax.text(0, A["latency_s"], f"{A['latency_s']:.0f} с", ha="center", va="bottom", color=INK)
    ax.text(1, B["latency_s"], f"{B['latency_s']:.0f} с", ha="center", va="bottom", color=INK)
    _style(ax, "Суммарное время ответов, с")
    ax.set_ylim(0, max(A["latency_s"], B["latency_s"]) * 1.25)
    ax.legend(frameon=False, fontsize=9)

    # --- качество
    ax = axes[2]
    scores = summary.get("scores") or {}
    tiers = ["simple", "medium", "hard"]
    qtier = {}
    for q in json.loads((HERE / "queries.json").read_text(encoding="utf-8"))["queries"]:
        qtier[q["id"]] = q["tier"]
    groups, a_vals, b_vals = [], [], []
    for g in ["все"] + tiers:
        ids = [i for i in scores if g == "все" or qtier.get(i) == g]
        a_list = [scores[i]["A"] for i in ids if scores[i]["A"] is not None]
        b_list = [scores[i]["B"] for i in ids if scores[i]["B"] is not None]
        if not a_list and not b_list:
            continue   # группа без оценок — не рисуем пустой столбик
        groups.append(g)
        a_vals.append(sum(a_list) / len(a_list) if a_list else 0)
        b_vals.append(sum(b_list) / len(b_list) if b_list else 0)
    x = range(len(groups))
    ax.bar([i - 0.18 for i in x], a_vals, width=0.34, color=C_BLUE, label="А: только gpt-5")
    ax.bar([i + 0.18 for i in x], b_vals, width=0.34, color=C_VIOLET, label="Б: мультимодельность")
    for i, (a, b) in enumerate(zip(a_vals, b_vals)):
        ax.text(i - 0.18, a + 0.05, f"{a:.2f}", ha="center", fontsize=9, color=INK)
        ax.text(i + 0.18, b + 0.05, f"{b:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(list(x), groups)
    ax.set_ylim(0, 6.6)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    _style(ax, "Средняя оценка экзаменатора (1–5)")
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=2)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_cost_per_question(runs: pd.DataFrame, out: Path) -> None:
    a = runs[(runs["step"] == "4")].set_index("qid")["cost_usd"]
    b_rows = runs[(runs["step"] == "5") & (runs["role"] != "intent")].set_index("qid")
    ids = list(a.index)
    fig, ax = plt.subplots(figsize=(14, 4.8), facecolor=SURFACE)
    x = range(len(ids))
    ax.bar([i - 0.2 for i in x], [a[q] for q in ids], width=0.4, color=C_BLUE, label="А: gpt-5")
    seen = set()
    for i, q in enumerate(ids):
        m = b_rows.loc[q, "model"]
        ax.bar([i + 0.2], [b_rows.loc[q, "cost_usd"]], width=0.4, color=model_color(m), hatch="//",
               edgecolor=SURFACE, label=(f"Б: {m}" if m not in seen else None))
        seen.add(m)
    ax.set_xticks(list(x), ids, rotation=0, fontsize=9)
    _style(ax, "Стоимость ответа на каждый вопрос, $ (S — простые, M — средние, H — сложные)")
    ax.legend(frameon=False, fontsize=9, ncol=4)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_confusion(runs: pd.DataFrame, out: Path) -> None:
    cls = runs[runs["step"] == "3"]
    tiers = ["simple", "medium", "hard"]
    ct = pd.crosstab(cls["expected_tier"], cls["predicted_tier"]).reindex(index=tiers, columns=tiers, fill_value=0)
    fig, ax = plt.subplots(figsize=(5.6, 4.8), facecolor=SURFACE)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#0d366b"])
    ax.imshow(ct.values, cmap=cmap)
    for i in range(3):
        for j in range(3):
            v = int(ct.values[i, j])
            ax.text(j, i, str(v), ha="center", va="center", fontsize=14,
                    color="white" if v > ct.values.max() / 2 else INK)
    ax.set_xticks(range(3), tiers)
    ax.set_yticks(range(3), tiers)
    ax.set_xlabel("дежурный ответил", color=INK2)
    ax.set_ylabel("ожидали", color=INK2)
    acc = (cls["expected_tier"] == cls["predicted_tier"]).mean()
    ax.set_title(f"Классификатор {cls['model'].iloc[0]}: точность {acc:.0%}", loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_tokenization(summary: dict, out: Path) -> None:
    rows = summary.get("tokenization") or []
    if not rows:
        return
    langs = [r["язык"] for r in rows]
    series = [(k, c) for k, c in [("токенов o200k (gpt-5)", C_BLUE), ("токенов cl100k (gpt-3.5/4)", C_ORANGE)]]
    extra = [k for k in rows[0] if k.startswith("токенов gemma")]
    if extra and all(isinstance(r[extra[0]], int) for r in rows):
        series.append((extra[0], C_AQUA))
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
    w = 0.8 / len(series)
    for k, (name, color) in enumerate(series):
        xs = [i + (k - (len(series) - 1) / 2) * w for i in range(len(langs))]
        vals = [r[name] for r in rows]
        ax.bar(xs, vals, width=w * 0.95, color=color, label=name.replace("токенов ", ""))
        for xx, v in zip(xs, vals):
            ax.text(xx, v + 0.15, str(v), ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(langs)), langs)
    _style(ax, "Одна и та же фраза «Сертификат истекает через 30 дней» — токенов по языкам", "токенов")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)
    rdir = HERE / args.results
    made = []
    for txt in sorted((rdir / "steps").glob("*.txt")):
        made += render_text_png(txt)
    summary = json.loads((rdir / "summary.json").read_text(encoding="utf-8"))
    runs = pd.read_csv(rdir / "runs.csv", dtype={"step": str})
    chart_comparison(summary, rdir / "comparison.png"); made.append(rdir / "comparison.png")
    chart_cost_per_question(runs, rdir / "cost_per_question.png"); made.append(rdir / "cost_per_question.png")
    chart_confusion(runs, rdir / "classifier_confusion.png"); made.append(rdir / "classifier_confusion.png")
    chart_tokenization(summary, rdir / "tokenization.png"); made.append(rdir / "tokenization.png")
    for p in made:
        print("→", p.relative_to(HERE))


if __name__ == "__main__":
    main()
