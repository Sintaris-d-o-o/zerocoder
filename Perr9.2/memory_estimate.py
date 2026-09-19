"""Расчёт памяти, которая нужна модели для инференса и для обучения (задание 9.2).

Считает по формуле урока: память под веса = число параметров × размер одного веса.
Дополнительно считает KV-кэш (память под контекст), потому что в нашем Норм-ассистенте
запрос приходит не пустым, а с найденными фрагментами регламентов.

Запуск: python memory_estimate.py
Результат: results/memory_models.md, results/memory_models.json, results/kv_cache.md
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

GB = 1024 ** 3   # считаем в гибибайтах (как показывают ОС и nvidia-smi)

# Размер одного веса в байтах для каждого формата хранения
FORMATS = {"FP32": 4.0, "FP16/BF16": 2.0, "INT8": 1.0, "INT4": 0.5}

# Множитель памяти для обучения относительно инференса.
# Урок называет 5–6х: сами веса + градиенты + два состояния оптимизатора Adam + активации.
TRAIN_MULTIPLIER = 5.0
# Служебный запас на активации и фрагментацию при инференсе (эмпирически ~20 %).
INFERENCE_OVERHEAD = 1.2


@dataclass
class Model:
    name: str
    params_b: float          # млрд параметров (физических, а не «эффективных»)
    layers: int              # число слоёв — нужно для KV-кэша
    kv_heads: int            # число KV-голов (grouped-query attention)
    head_dim: int            # размерность одной головы
    note: str = ""
    real_file_gb: float | None = None   # фактический размер файла весов, если известен
    generative: bool = True  # False для энкодеров (эмбеддинги) — они не генерируют, KV-кэш не нужен
    active_b: float | None = None       # активных параметров, млрд (у E2B/E4B меньше, чем всего)
    measured_ram_gb: float | None = None  # сколько модель реально заняла в памяти (замер)


# Данные по архитектуре — из карточек моделей на Hugging Face и вывода `ollama show`.
# Активные параметры (active_b) отличаются от всех у моделей вида E2B/E4B: скорость генерации
# зависит от активных, а память — от всех.
MODELS = [
    Model("Gemma 4 E2B (certtaris, normassist:v1)", 5.1, 30, 2, 256,
          "«E2B» = 2 млрд активных параметров, физически 5,1 млрд; замер: 7,7–7,8 ГБ в памяти "
          "при окне 16 384 токена", real_file_gb=7.2, active_b=2.0, measured_ram_gb=7.75),
    Model("Gemma 4 E4B", 8.0, 35, 4, 256,
          "второй вариант для certtaris; замер: 10,0 ГБ в памяти", real_file_gb=9.6,
          active_b=4.0, measured_ram_gb=10.0),
    Model("Gemma 4 12B", 11.9, 48, 8, 256,
          "дежурный мультимодельного агента из задания 9.1", real_file_gb=7.6, active_b=11.9),
    Model("MiniLM-L12-v2 (эмбеддинги certtaris)", 0.118, 12, 12, 32,
          "384 измерения; энкодер — превращает фрагмент в вектор и ничего не генерирует, "
          "поэтому кэш контекста ему не нужен", generative=False),
    Model("GPT-3 175B", 175.0, 96, 96, 128,
          "пример из текста задания, для масштаба", active_b=175.0),
]

# Замеренная на SintAItion эффективная пропускная способность памяти, ГБ/с (см. раздел 6 отчёта).
MEASURED_BANDWIDTH_GBS = 38.0
# Паспортная пропускная способность вариантов дискретных видеокарт, ГБ/с.
GPU_BANDWIDTH = {
    "Radeon 890M (встроенная, замер)": MEASURED_BANDWIDTH_GBS,
    "RTX 4060 Ti 16 ГБ": 288.0,
    "RTX 3060 12 ГБ": 360.0,
    "RTX 3090 24 ГБ": 936.0,
    "RTX 4090 24 ГБ": 1008.0,
}
GPU_EFFICIENCY = 0.5   # осторожная оценка: половина паспортной величины

CONTEXTS = {
    "короткий вопрос (500 токенов)": 500,
    "рабочее окно certtaris (16 384)": 16_384,
    "длинный документ (32 000)": 32_000,
    "максимальное окно (262 144)": 262_144,
}


def weights_gb(params_b: float, bytes_per_param: float) -> float:
    return params_b * 1e9 * bytes_per_param / GB


def kv_cache_gb(m: Model, ctx_tokens: int, bytes_per_elem: float = 2.0) -> float:
    """Память под контекст: 2 (ключи и значения) × слои × KV-головы × размер головы × токены."""
    return 2 * m.layers * m.kv_heads * m.head_dim * ctx_tokens * bytes_per_elem / GB


def main() -> None:
    OUT.mkdir(exist_ok=True)
    data = {"models": [], "contexts": CONTEXTS, "train_multiplier": TRAIN_MULTIPLIER}

    lines = ["# Память под веса модели", "",
             "Формула урока: память = число параметров × размер одного веса.",
             "Числа в гибибайтах (ГиБ) — так же их показывают операционная система и `nvidia-smi`.", ""]
    header = "| Модель | Параметров, млрд | " + " | ".join(FORMATS) + " | Реальный файл | Замер в памяти |"
    lines += [header, "|" + "---|" * (len(FORMATS) + 4)]
    for m in MODELS:
        cells = [f"{weights_gb(m.params_b, b):.1f}" for b in FORMATS.values()]
        real = f"{m.real_file_gb:.1f} ГБ (Q4_K_M)" if m.real_file_gb else "—"
        meas = f"{m.measured_ram_gb:.1f} ГБ" if m.measured_ram_gb else "—"
        lines.append(f"| {m.name} | {m.params_b:g} | " + " | ".join(cells) + f" | {real} | {meas} |")
        data["models"].append({
            "name": m.name, "params_b": m.params_b, "active_b": m.active_b, "note": m.note,
            "weights_gb": {f: round(weights_gb(m.params_b, b), 2) for f, b in FORMATS.items()},
            "real_file_gb": m.real_file_gb, "measured_ram_gb": m.measured_ram_gb,
            "inference_gb_int4": round(weights_gb(m.params_b, 0.5) * INFERENCE_OVERHEAD, 2),
            "training_gb_fp16": round(weights_gb(m.params_b, 2.0) * TRAIN_MULTIPLIER, 2),
        })
    lines += ["", "Примечания:", ""]
    for m in MODELS:
        if m.note:
            lines.append(f"- **{m.name}** — {m.note}.")
    lines += ["",
              "Почему реальный файл не совпадает с колонкой INT4: формат Q4_K_M — не «всё в 4 бита». "
              "Часть слоёв (внимание, эмбеддинги словаря) хранится точнее, поэтому файл получается "
              "крупнее расчёта. Для Gemma 4 12B расчёт даёт "
              f"{weights_gb(MODELS[1].params_b, 0.5):.1f} ГиБ, фактический файл — {MODELS[1].real_file_gb} ГБ. "
              "Планировать память надо по фактическому размеру файла, а колонку INT4 использовать "
              "как нижнюю границу."]

    lines += ["", "## Инференс против обучения", "",
              f"Урок: обучение требует в 5–6 раз больше памяти, чем инференс. Берём множитель {TRAIN_MULTIPLIER:g}× "
              f"(веса + градиенты + два состояния оптимизатора Adam + активации) и запас "
              f"{round((INFERENCE_OVERHEAD - 1) * 100)} % на активации при инференсе.", "",
              "| Модель | Инференс в INT4, ГиБ | Инференс в FP16, ГиБ | Обучение в FP16, ГиБ |",
              "|---|---|---|---|"]
    for m in MODELS:
        lines.append(f"| {m.name} | {weights_gb(m.params_b, 0.5) * INFERENCE_OVERHEAD:.1f} | "
                     f"{weights_gb(m.params_b, 2.0) * INFERENCE_OVERHEAD:.1f} | "
                     f"{weights_gb(m.params_b, 2.0) * TRAIN_MULTIPLIER:.1f} |")
    (OUT / "memory_models.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- KV-кэш ---
    kv = ["# Память под контекст (KV-кэш)", "",
          "Веса — не вся память. Пока модель отвечает, она держит в памяти «кэш внимания» по всем",
          "прочитанным токенам. Для Норм-ассистента это важно: вопрос приходит вместе с найденными",
          "фрагментами регламентов, то есть контекст всегда длинный.", "",
          "Формула: 2 (ключи и значения) × слои × KV-головы × размер головы × токены × 2 байта (FP16).", "",
          "| Модель | " + " | ".join(CONTEXTS) + " |", "|" + "---|" * (len(CONTEXTS) + 1)]
    kv_data = {}
    for m in MODELS:
        if not m.generative:
            kv.append(f"| {m.name} | " + " | ".join(["не применяется"] * len(CONTEXTS)) + " |")
            continue
        cells = []
        kv_data[m.name] = {}
        for label, n in CONTEXTS.items():
            v = kv_cache_gb(m, n)
            kv_data[m.name][label] = round(v, 3)
            cells.append(f"{v:.2f} ГиБ" if v >= 0.01 else "<0,01 ГиБ")
        kv.append(f"| {m.name} | " + " | ".join(cells) + " |")
    data["kv_cache_gb"] = kv_data

    m12 = next(m for m in MODELS if m.name.startswith("Gemma 4 12B"))
    e2b = MODELS[0]
    kv += ["", "Что это значит на практике:", "",
           f"- в рабочем режиме certtaris (окно 16 384 токена) формула даёт для "
           f"{e2b.name.split(' (')[0]} {kv_cache_gb(e2b, 16384):.2f} ГиБ на контекст. Замер показывает "
           f"меньше: файл 7,2 ГБ → 7,7–7,8 ГБ в памяти, то есть около 0,5 ГБ. Формула — **верхняя "
           f"оценка**: Gemma в большинстве слоёв использует скользящее окно внимания и хранит не весь "
           f"контекст целиком;",
           f"- у более крупной {m12.name} тот же контекст стоил бы {kv_cache_gb(m12, 16384):.2f} ГиБ, а полное "
           f"окно (262 144 токена) — {kv_cache_gb(m12, 262144):.0f} ГиБ, больше, чем у любой игровой видеокарты;",
           "- отсюда практическое правило: считать надо не «веса», а «веса + контекст того размера, "
           "который реально приходит от поиска по базе норм».", "",
           "Проверено замером: уменьшать окно ради экономии памяти бессмысленно — переход с 16 384 на "
           "8 192 токена освобождает около 0,3 ГБ и не ускоряет ответ (27,6 с против 27,9 с)."]
    (OUT / "kv_cache.md").write_text("\n".join(kv) + "\n", encoding="utf-8")

    # --- скорость генерации: пропускная способность памяти / объём активных весов ---
    sp = ["# Скорость генерации: откуда она берётся", "",
          "Чтобы выдать один токен, модель обязана прочитать все свои активные веса. Значит:", "",
          "> скорость генерации ≈ пропускная способность памяти ÷ объём активных весов", "",
          "Проверка формулы на замерах SintAItion (встроенная графика Radeon 890M):", "",
          "| Замер | Активных весов на токен | Скорость, ток/с | Получается пропускная способность |",
          "|---|---:|---:|---:|"]
    checks = [("Gemma 4 E2B", 2.0, 39.1), ("Gemma 4 E4B", 4.0, 20.8), ("Gemma 4 E2B, прогретая", 2.0, 41.9)]
    for name, act, toks in checks:
        gbt = act * 1e9 * 0.5 / GB
        sp.append(f"| {name} | {gbt:.2f} ГБ | {toks:.1f} | **{gbt * toks:.1f} ГБ/с** |")
    sp += ["", f"Три независимых замера дают одно и то же число — около {MEASURED_BANDWIDTH_GBS:.0f} ГБ/с. "
           "Формула работает, значит по ней можно оценить любую другую видеокарту.", "",
           "## Что даст дискретная видеокарта", "",
           f"Считаем осторожно: {GPU_EFFICIENCY:.0%} паспортной пропускной способности.", "",
           "| Вариант | Пропускная способность | Ожидаемая скорость Gemma 4 E2B | Во сколько раз быстрее |",
           "|---|---:|---:|---:|"]
    base = None
    speed_data = {}
    gb_per_token = 2.0 * 1e9 * 0.5 / GB   # активные веса E2B в Q4
    for name, bw in GPU_BANDWIDTH.items():
        eff = bw if "замер" in name else bw * GPU_EFFICIENCY
        speed = eff / gb_per_token
        base = base or speed
        speed_data[name] = {"bandwidth_gbs": bw, "effective_gbs": round(eff, 1),
                            "tokens_per_s": round(speed), "speedup": round(speed / base, 1)}
        sp.append(f"| {name} | {bw:.0f} ГБ/с | ~{speed:.0f} ток/с | {speed / base:.1f}× |")
    sp += ["", "Это **верхняя граница**: на таких скоростях узким местом становятся уже вычисления, "
           "а не память, поэтому реально стоит ожидать ускорения в три-пять раз."]
    data["speed"] = speed_data
    (OUT / "speed_estimate.md").write_text("\n".join(sp) + "\n", encoding="utf-8")

    (OUT / "memory_models.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for f in ("memory_models.md", "kv_cache.md", "speed_estimate.md"):
        print((OUT / f).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
