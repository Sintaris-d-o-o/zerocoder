# -*- coding: utf-8 -*-
"""Ячейка для УЖЕ ЗАПУЩЕННОЙ сессии Colab: сохранить результаты прогона.

Не часть обычного порядка ячеек. Нужна, когда обучение уже прошло в текущей
сессии, а сохранять результаты нечем — например, ноутбук был запущен из версии
без шага 12. Скопировать целиком в пустую ячейку Colab и выполнить.

Ничего не переобучает и не перезагружает: только читает то, что уже лежит в
памяти сессии (`trainer`, `model`, `tokenizer`), и раскладывает по файлам.

Требуется: выполненный `trainer.train()` в этой же сессии.
"""

# ======================= КОПИРОВАТЬ ОТСЮДА =======================
import json, subprocess, shutil
from pathlib import Path

assert "trainer" in globals(), "В сессии нет объекта trainer — обучение в ней не запускалось."

# 1. Куда складывать: Google Drive (тогда файлы сами окажутся на компьютере в J:)
try:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = Path("/content/drive/MyDrive/Colab Notebooks/sintaris/perr8.2")
    on_drive = True
except Exception as e:
    print("Google Drive недоступен, сохраняю локально в сессию:", e)
    OUT = Path("/content/perr8.2_results")
    on_drive = False
OUT.mkdir(parents=True, exist_ok=True)

# 2. Собираем то, что есть в памяти сессии (переменных может не хватать — не падаем)
DATASET = globals().get("DATASET", "run")
tag = f"{DATASET}_colab"
hist = list(trainer.state.log_history)
baseline = globals().get("baseline") or {}
final = globals().get("final") or {}
if not final:
    print("Финального замера в сессии нет — делаю его сейчас...")
    final = trainer.evaluate()

meta = next((h for h in reversed(hist) if "train_runtime" in h), {})
train_runtime = meta.get("train_runtime")
train_loss = meta.get("train_loss")

has_gpu = False
gpu_name, peak = "CPU", "—"
try:
    import torch
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        peak = f"{torch.cuda.max_memory_allocated() / 1e9:.2f} ГБ"
except Exception:
    pass

# 3. Текстовый отчёт — ровно то, что просит пункт 11 задания
lines = [
    f"Задание 8.2 — fine-tuning {globals().get('model_checkpoint', 't5-small')} на датасете {DATASET}",
    f"Оборудование: {gpu_name} | пик VRAM: {peak}",
    f"Строк: train {len(trainer.train_dataset)}, validation {len(trainer.eval_dataset)}",
    f"Эпох: {trainer.args.num_train_epochs} | batch: {trainer.args.per_device_train_batch_size} "
    f"| lr: {trainer.args.learning_rate} | fp16: {trainer.args.fp16}",
]
if train_runtime:
    lines.append(f"Время обучения: {train_runtime / 60:.1f} мин | итоговый training loss: {train_loss:.4f}")
lines += ["", f"{'эпоха':>6} {'val loss':>10} {'rouge1':>8} {'rouge2':>8} {'rougeL':>8} {'gen_len':>8}"]
for h in hist:
    if "eval_loss" in h:
        lines.append(
            f"{h['epoch']:>6.2f} {h['eval_loss']:>10.4f} {h.get('eval_rouge1', 0):>8.2f} "
            f"{h.get('eval_rouge2', 0):>8.2f} {h.get('eval_rougeL', 0):>8.2f} {h.get('eval_gen_len', 0):>8.1f}"
        )
if baseline:
    lines += ["", "ДО обучения -> ПОСЛЕ обучения:"]
    for m in ("loss", "rouge1", "rouge2", "rougeL", "gen_len"):
        b, f = baseline.get(f"base_{m}"), final.get(f"eval_{m}")
        if b is not None and f is not None:
            lines.append(f"  {m:<10}{b:>9.3f} -> {f:>9.3f}   ({f - b:+.3f})")
else:
    lines += ["", "Замера до обучения в сессии не было — сравнивать не с чем."]

with open(OUT / f"report_{tag}.txt", "w", encoding="utf-8") as fh:
    for line in lines:
        print(line, file=fh)
print(*lines, sep="\n")

# 4. Все числа целиком
with open(OUT / f"metrics_{tag}.json", "w", encoding="utf-8") as fh:
    json.dump({"dataset": DATASET, "gpu": gpu_name, "peak_vram": peak,
               "train_runtime_sec": train_runtime, "train_loss": train_loss,
               "baseline": baseline, "final": final, "log_history": hist},
              fh, ensure_ascii=False, indent=2, default=str)

# 5. График обучения
try:
    import matplotlib.pyplot as plt
    tr = [(h["epoch"], h["loss"]) for h in hist if "loss" in h and "eval_loss" not in h]
    ev = [(h["epoch"], h["eval_loss"]) for h in hist if "eval_loss" in h]
    rg = [(h["epoch"], h.get("eval_rouge1"), h.get("eval_rouge2"), h.get("eval_rougeL"))
          for h in hist if "eval_rouge1" in h]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    if tr: ax[0].plot(*zip(*tr), label="train")
    if ev: ax[0].plot(*zip(*ev), marker="o", label="validation")
    ax[0].set_title("loss"); ax[0].set_xlabel("эпоха"); ax[0].legend()
    for i, nm in ((1, "ROUGE-1"), (2, "ROUGE-2"), (3, "ROUGE-L")):
        if rg: ax[1].plot([r[0] for r in rg], [r[i] for r in rg], marker="o", label=nm)
    ax[1].set_title("ROUGE на валидации"); ax[1].set_xlabel("эпоха"); ax[1].legend()
    fig.suptitle(f"{DATASET} · {gpu_name}")
    fig.tight_layout(); fig.savefig(OUT / f"training_{tag}.png", dpi=130)
    print("график сохранён")
except Exception as e:
    print("график не построен:", e)

# 6. Подтверждение, что обучение шло на GPU
if has_gpu:
    with open(OUT / f"gpu_{tag}.txt", "w", encoding="utf-8") as fh:
        fh.write(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

print(f"\nСохранено в {OUT}:")
for f in sorted(OUT.glob(f"*{tag}*")):
    print(f"   {f.name:<40} {f.stat().st_size / 1024:>7.0f} КБ")

# 7. Если Drive не подключился — забираем архивом через браузер
if not on_drive:
    shutil.make_archive("/content/perr8_2_results", "zip", OUT)
    from google.colab import files
    files.download("/content/perr8_2_results.zip")
# ======================= КОПИРОВАТЬ ДОСЮДА =======================
