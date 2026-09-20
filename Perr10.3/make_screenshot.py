"""Рисует «скриншот» терминала из сохранённого вывода прогона (задание 10.3).

Задание просит скриншот терминала с прогресс-баром и статусами. Прогресс-бар
перерисовывает одну и ту же строку символом возврата каретки, поэтому в файле лога он
лежит в одну длинную строку. Скрипт разворачивает её обратно по шагам и отрисовывает
как окно терминала — получается тот же вывод, что видел человек на экране.

Запуск: python make_screenshot.py [results/terminal.log]
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def unfold(raw: str) -> list[str]:
    """Разворачивает перерисовку строки: каждый шаг прогресса становится своей строкой."""
    lines: list[str] = []
    for chunk in raw.splitlines():
        parts = [p for p in chunk.split("\r") if p.strip()]
        lines.extend(parts or [chunk])
    return lines


def _font(size: int) -> ImageFont.FreeTypeFont:
    for cand in (Path("C:/Windows/Fonts/consola.ttf"), Path("C:/Windows/Fonts/cour.ttf")):
        if cand.exists():
            return ImageFont.truetype(str(cand), size)
    from matplotlib import font_manager
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans Mono"), size)


def render(lines: list[str], out: Path, title: str, max_cols: int = 118) -> Path:
    wrapped: list[str] = []
    for ln in lines:
        wrapped.extend(textwrap.wrap(ln, max_cols, replace_whitespace=False,
                                     drop_whitespace=False) or [""])
    font, title_font = _font(16), _font(14)
    ch_w = font.getbbox("M")[2]
    line_h = 21
    width = max(900, ch_w * (min(max(len(l) for l in wrapped) + 2, max_cols + 2)) + 40)
    height = 44 + line_h * len(wrapped) + 26

    img = Image.new("RGB", (width, height), "#1e1e1e")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, width, 36], fill="#2d2d2d")
    for i, col in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
        d.ellipse([14 + i * 22, 11, 28 + i * 22, 25], fill=col)
    d.text((90, 10), title, font=title_font, fill="#cfcfcf")

    y = 50
    for ln in wrapped:
        color = "#d4d4d4"
        if "100%" in ln or ln.startswith("Готово") or ln.startswith("Видео сохранено"):
            color = "#7fbf7f"
        elif "█" in ln:
            color = "#6fb3d2"
        elif ln.startswith(("Сервис:", "Длительность:", "Промпт:")):
            color = "#ffd166"
        elif ln.startswith("ОШИБКА"):
            color = "#ff6b6b"
        d.text((20, y), ln, font=font, fill=color)
        y += line_h
    img.save(out)
    return out


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    log = HERE / (argv[0] if argv else "results/terminal.log")
    if not log.exists():
        print(f"Нет файла {log}. Сначала запустите генерацию и сохраните вывод:", file=sys.stderr)
        print("  python video_generator.py | tee results/terminal.log", file=sys.stderr)
        return 1
    lines = unfold(log.read_text(encoding="utf-8"))
    out = render(lines, log.parent / "01_терминал.png",
                 "python video_generator.py — генерация видео через RouterAI")
    print(f"→ {out.relative_to(HERE)} ({len(lines)} строк)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
