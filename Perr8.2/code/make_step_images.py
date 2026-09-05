# -*- coding: utf-8 -*-
"""Собирает «скриншоты этапов» из реального вывода запуска.

Задание требует показать выполнение каждого этапа снимками экрана. Здесь
снимок делается не фотоаппаратом, а отрисовкой настоящего текста, который
шаг напечатал во время прогона (results/steps/*.txt) — ничего не дописывается
и не редактируется вручную.
"""
import sys, textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
STEPS = ROOT / "results" / "steps"

BG, FG, BAR, BAR_FG = "#1e1e2e", "#e6e6ea", "#313244", "#a6adc8"
MAX_COLS, MAX_LINES = 118, 46


def render(txt_path: Path) -> Path:
    raw = txt_path.read_text(encoding="utf-8").rstrip("\n").split("\n")

    lines = []
    for line in raw:
        if len(line) <= MAX_COLS:
            lines.append(line)
        else:
            lines.extend(textwrap.wrap(line, MAX_COLS, subsequent_indent="    ",
                                       break_long_words=True, break_on_hyphens=False) or [""])
    if len(lines) > MAX_LINES:
        head, tail = lines[: MAX_LINES - 9], lines[-8:]
        lines = head + ["", f"   … пропущено {len(lines) - len(head) - len(tail)} строк вывода …", ""] + tail

    fig_w = 12.0
    fig_h = max(2.2, 0.205 * (len(lines) + 3))
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=110, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set_facecolor(BG)

    bar_h = 0.30 / fig_h * 3
    ax.add_patch(plt.Rectangle((0, 1 - bar_h), 1, bar_h, color=BAR, transform=ax.transAxes, zorder=1))
    for i, c in enumerate(("#f38ba8", "#f9e2af", "#a6e3a1")):
        ax.add_patch(plt.Circle((0.012 + i * 0.016, 1 - bar_h / 2), 0.0055,
                                color=c, transform=ax.transAxes, zorder=2))
    ax.text(0.075, 1 - bar_h / 2, txt_path.stem, color=BAR_FG, family="monospace",
            fontsize=9, va="center", transform=ax.transAxes, zorder=2)

    ax.text(0.012, 1 - bar_h - 0.012, "\n".join(lines), color=FG, family="monospace",
            fontsize=8.2, va="top", ha="left", linespacing=1.35, transform=ax.transAxes, zorder=2)

    out = txt_path.with_suffix(".png")
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    return out


if __name__ == "__main__":
    pattern = sys.argv[1] if len(sys.argv) > 1 else "*.txt"
    files = sorted(STEPS.glob(pattern))
    if not files:
        sys.exit(f"Нет файлов по шаблону {pattern} в {STEPS}")
    for f in files:
        print(f"  {f.name:<50} -> {render(f).name}")
    print(f"\nГотово: {len(files)} снимков в {STEPS}")
