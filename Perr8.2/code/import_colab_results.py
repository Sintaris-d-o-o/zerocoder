# -*- coding: utf-8 -*-
"""Забирает результаты прогона из Colab (через Google Drive) в папку задания.

Ноутбук на шаге 12 складывает метрики, отчёт, график и характеристики GPU в
`MyDrive/Colab Notebooks/sintaris/perr8.2/`. Диск смонтирован в системе как `J:`,
поэтому копировать вручную ничего не нужно — достаточно запустить этот скрипт.

    python code/import_colab_results.py

Дополнительно: если рядом положить скачанный из Colab ноутбук с выводами
(File -> Download -> Download .ipynb), он тоже будет перенесён — это главный
документ-доказательство, в нём сохранён вывод каждой ячейки.
"""
import argparse, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = Path(r"J:/My Drive/Colab Notebooks/sintaris/perr8.2")

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=str(DEFAULT_SRC), help="папка на Google Drive, куда писал ноутбук")
ap.add_argument("--dst", default=str(ROOT / "results" / "colab"))
a = ap.parse_args()

src, dst = Path(a.src), Path(a.dst)
if not src.exists():
    raise SystemExit(
        f"Папки {src} нет.\n"
        "Значит, шаг 12 ноутбука ещё не выполнялся либо Google Drive не смонтирован в системе.\n"
        "Проверь путь: My Drive -> Colab Notebooks -> sintaris -> perr8.2"
    )

dst.mkdir(parents=True, exist_ok=True)
patterns = ("*.json", "*.txt", "*.png", "*.ipynb", "*.csv")
copied = []
for pat in patterns:
    for f in sorted(src.glob(pat)):
        target = dst / f.name
        if target.exists() and target.stat().st_mtime >= f.stat().st_mtime and target.stat().st_size == f.stat().st_size:
            continue
        shutil.copy2(f, target)
        copied.append(target)

if not copied:
    print(f"Новых файлов нет — в {dst} уже лежит всё, что есть на диске.")
else:
    print(f"Перенесено файлов: {len(copied)}")
    for f in copied:
        print(f"  {f.name:<40} {f.stat().st_size / 1024:>8.0f} КБ")

print(f"\nВсего в {dst}:")
for f in sorted(dst.iterdir()):
    print(f"  {f.name:<40} {f.stat().st_size / 1024:>8.0f} КБ")
