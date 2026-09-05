# -*- coding: utf-8 -*-
"""Задание 8.2 — fine-tuning T5-Small: все 11 шагов домашнего задания.

Один и тот же пайплайн для двух датасетов:
  --dataset xsum        — эталон задания (новости BBC -> саммари в одно предложение)
  --dataset normassist  — собственный датасет из задания 8.1 (контекст -> ответ со ссылками)

Вывод каждого шага дублируется в results/steps/NN_*.txt — из этих файлов
затем собираются «скриншоты этапов» (make_step_images.py).
"""
import argparse, io, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STEPS = RESULTS / "steps"

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["xsum", "normassist"], default="xsum")
ap.add_argument("--train-rows", type=int, default=2000)
ap.add_argument("--val-rows", type=int, default=200)
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--batch-size", type=int, default=8)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--max-input", type=int, default=512)
ap.add_argument("--max-target", type=int, default=64)
ap.add_argument("--steps-prefix", default="")
ap.add_argument("--skip-baseline", action="store_true")
ARGS = ap.parse_args()

STEPS.mkdir(parents=True, exist_ok=True)
PFX = ARGS.steps_prefix or ARGS.dataset


class _Tee:
    def __init__(self, *targets):
        self.targets = targets

    def write(self, s):
        for t in self.targets:
            t.write(s)
        return len(s)

    def flush(self):
        for t in self.targets:
            t.flush()


class Step:
    """Печатает баннер шага и параллельно пишет весь вывод шага в файл."""

    def __init__(self, n, title):
        self.n, self.title = n, title

    def __enter__(self):
        self.buf = io.StringIO()
        self._old = sys.stdout
        sys.stdout = _Tee(self._old, self.buf)
        print("=" * 78)
        print(f"ШАГ {self.n} — {self.title}")
        print("=" * 78)
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        print(f"\n[шаг {self.n} занял {time.time() - self.t0:.1f} с]")
        sys.stdout = self._old
        # Из имени файла убираем символы, запрещённые в путях Windows. Особенно двоеточие:
        # без этого "шаг 11 — Отслеживаем метрики: loss..." уходит в альтернативный поток
        # NTFS (file:stream) вместо обычного файла, и текст шага молча теряется.
        slug = self.title.split("(")[0].strip().lower()
        for ch in r':<>"/\|?*,':
            slug = slug.replace(ch, "")
        slug = "_".join(slug.split())[:40]
        (STEPS / f"{PFX}_{self.n:02d}_{slug}.txt").write_text(self.buf.getvalue(), encoding="utf-8")
        return False


# ----------------------------------------------------------------- ШАГ 1
with Step(1, "Установка библиотек и фиксация версий"):
    import torch, transformers, datasets, evaluate
    import numpy as np

    versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "evaluate": evaluate.__version__,
    }
    for k, v in versions.items():
        print(f"  {k:<15} {v}")
    HAS_GPU = torch.cuda.is_available()
    device = "cuda" if HAS_GPU else "cpu"
    if HAS_GPU:
        print(f"\n  Устройство: GPU — {torch.cuda.get_device_name(0)}")
    else:
        print(f"\n  Устройство: CPU ({os.cpu_count()} ядер, GPU недоступен)")
    print(f"  fp16 доступен: {HAS_GPU} (на CPU обучение идёт в fp32)")

# ----------------------------------------------------------------- ШАГ 2
with Step(2, "Определяем модель T5-Small и датасет"):
    model_checkpoint = "t5-small"
    prefix = "summarize: "
    print(f"  Модель:  {model_checkpoint}")
    if ARGS.dataset == "xsum":
        dataset_name = "EdinburghNLP/xsum"
        print(f"  Датасет: {dataset_name} — экстремальная суммаризация новостей BBC")
    else:
        dataset_name = "normassist (собственный, из задания 8.1)"
        print(f"  Датасет: {dataset_name}")
        print("           вход = вопрос + нормативный контекст, выход = краткий ответ со ссылками [n]")
    print(f"  Префикс задачи для T5: {prefix!r}")

# ----------------------------------------------------------------- ШАГ 3
with Step(3, "Загружаем датасет и метрику ROUGE, смотрим структуру"):
    from datasets import load_dataset

    t0 = time.time()
    if ARGS.dataset == "xsum":
        raw_datasets = load_dataset("EdinburghNLP/xsum")
    else:
        d = ROOT / "data" / "normassist_seq2seq"
        raw_datasets = load_dataset(
            "json",
            data_files={"train": str(d / "train.jsonl"), "validation": str(d / "validation.jsonl")},
        )
    metric = evaluate.load("rouge")
    print(f"  Загружено за {time.time() - t0:.1f} с\n")
    print("  Структура датасета:")
    print("  " + str(raw_datasets).replace("\n", "\n  "))
    ex = raw_datasets["train"][0]
    print(f"\n  Пример (document, первые 400 символов):\n    {ex['document'][:400]!r}")
    print(f"\n  Пример (summary):\n    {ex['summary'][:300]!r}")
    print(f"\n  Метрика ROUGE загружена: {metric.name}")

    full = {k: len(v) for k, v in raw_datasets.items()}
    n_tr = min(ARGS.train_rows, len(raw_datasets["train"]))
    n_va = min(ARGS.val_rows, len(raw_datasets["validation"]))
    raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42).select(range(n_tr))
    raw_datasets["validation"] = raw_datasets["validation"].shuffle(seed=42).select(range(n_va))
    print(f"\n  Уменьшаем выборку (полный размер {full}):")
    print(f"    train      {n_tr}")
    print(f"    validation {n_va}")

# ----------------------------------------------------------------- ШАГ 4
with Step(4, "Токенизация данных"):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    print(f"  Токенизатор от той же модели: {model_checkpoint}, словарь {tokenizer.vocab_size} токенов")
    probe = tokenizer("Hello, this one sentence!")
    print(f"  Проверка: tokenizer('Hello, this one sentence!') -> {probe['input_ids']}")
    print(f"            обратно: {tokenizer.decode(probe['input_ids'])!r}")

    max_input_length, max_target_length = ARGS.max_input, ARGS.max_target

    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["document"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        labels = tokenizer(text_target=examples["summary"], max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print(f"\n  Ограничения длины: вход {max_input_length} токенов, выход {max_target_length}")
    demo = preprocess_function(raw_datasets["train"][:2])
    print("  Пример после препроцессинга (2 строки):")
    print(f"    input_ids[0][:24] = {demo['input_ids'][0][:24]}")
    print(f"    labels[0][:24]    = {demo['labels'][0][:24]}")
    print(f"    длины входов: {[len(x) for x in demo['input_ids']]}, длины меток: {[len(x) for x in demo['labels']]}")

    unk = tokenizer.unk_token_id
    all_ids = [i for x in demo["input_ids"] for i in x]
    print(f"    доля <unk> в этих примерах: {sum(1 for i in all_ids if i == unk) / len(all_ids):.2%}")

    t0 = time.time()
    # remove_columns обязателен: иначе в батч попадут исходные текстовые колонки
    # (document/summary), и упаковщик упадёт при попытке сделать из строк тензор.
    tokenized_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names
    )
    sizes = {k: len(v) for k, v in tokenized_datasets.items()}
    print(f"  Колонки после токенизации: {tokenized_datasets['train'].column_names}")
    print(f"\n  Токенизировано за {time.time() - t0:.1f} с: {sizes}")

# ----------------------------------------------------------------- ШАГ 5
with Step(5, "Инициализируем модель T5-Small"):
    from transformers import (
        AutoModelForSeq2SeqLM,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    total = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Класс модели: {type(model).__name__} (encoder-decoder, seq2seq)")
    print(f"  Параметров всего:     {total:,}")
    print(f"  Обучаемых параметров: {train_p:,} ({train_p / total:.0%} — полный fine-tuning, без LoRA)")
    print("  Для сравнения, в задании 8.1: Gemma 4 E2B, ~2 000 000 000 параметров, LoRA обучала <1%")

# ----------------------------------------------------------------- ШАГ 6
with Step(6, "Задаём гиперпараметры обучения"):
    batch_size = ARGS.batch_size
    model_name = model_checkpoint.split("/")[-1]
    out_dir = ROOT / "runs" / f"{model_name}-finetuned-{ARGS.dataset}"
    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=25,
        learning_rate=ARGS.lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=ARGS.epochs,
        predict_with_generate=True,
        generation_max_length=ARGS.max_target,
        generation_num_beams=1,
        fp16=HAS_GPU,
        push_to_hub=False,
        report_to=[],
    )
    for k in (
        "output_dir", "learning_rate", "num_train_epochs", "per_device_train_batch_size",
        "weight_decay", "eval_strategy", "predict_with_generate", "fp16", "push_to_hub",
    ):
        print(f"  {k:<30} {getattr(args, k)}")
    print("\n  push_to_hub=False и report_to=[] — модель никуда не публикуется,")
    print("  поэтому токен Hugging Face и ключ wandb в этом проекте не нужны.")

# ----------------------------------------------------------------- ШАГ 7
with Step(7, "Создаём упаковщик данных (data collator)"):
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    print(f"  {type(data_collator).__name__}: дополняет примеры до одной длины внутри батча")
    n_demo = min(4, len(tokenized_datasets["train"]))
    batch = data_collator([tokenized_datasets["train"][i] for i in range(n_demo)])
    shapes = ", ".join(f"{k}={tuple(v.shape)}" for k, v in batch.items())
    print(f"  Пробный батч из {n_demo} примеров: {shapes}")
    n_ignored = int((batch["labels"] == -100).sum())
    print(f"  Заглушки в метках помечены -100 (в функцию потерь не идут): {n_ignored} из {batch['labels'].numel()}")

# ----------------------------------------------------------------- ШАГ 8
with Step(8, "Функция вычисления метрик ROUGE + своя метрика длины"):
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        for pkg in ("punkt_tab", "punkt"):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass

    def split_sentences(text):
        try:
            return nltk.sent_tokenize(text)
        except Exception:
            t = text.replace("! ", ".\n").replace("? ", ".\n").replace(". ", ".\n")
            return [s for s in t.split("\n") if s]

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = ["\n".join(split_sentences(p.strip())) for p in decoded_preds]
        decoded_labels = ["\n".join(split_sentences(l.strip())) for l in decoded_labels]
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: v * 100 for k, v in result.items()}
        pred_lens = [np.count_nonzero(p != tokenizer.pad_token_id) for p in predictions]
        result["gen_len"] = float(np.mean(pred_lens))
        return {k: round(float(v), 4) for k, v in result.items()}

    toy_pred = np.array(tokenizer(["the cat sat on the mat"], max_length=16, padding="max_length")["input_ids"])
    toy_ref = np.array(tokenizer(["a cat was sitting on the mat"], max_length=16, padding="max_length")["input_ids"])
    print("  Метрики: rouge1/rouge2/rougeL/rougeLsum (перекрытие n-грамм)")
    print("           + gen_len — своя (кастомная) метрика: средняя длина предсказания")
    print("  Проверка на игрушечном примере ('the cat sat on the mat' vs 'a cat was sitting on the mat'):")
    print(f"    {compute_metrics((toy_pred, toy_ref))}")

# ----------------------------------------------------------------- ШАГ 9
with Step(9, "Передаём всё в Seq2SeqTrainer"):
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    steps_per_epoch = max(1, len(tokenized_datasets["train"]) // batch_size)
    print(f"  {type(trainer).__name__} собран.")
    print(f"  train {len(tokenized_datasets['train'])} строк / batch {batch_size} = ~{steps_per_epoch} шагов на эпоху")
    print(f"  всего шагов: ~{int(steps_per_epoch * ARGS.epochs)}")

# ----------------------------------------------------------------- ШАГ 10
with Step(10, "Запускаем обучение"):
    baseline = None
    if not ARGS.skip_baseline:
        print("  Сначала — замер ДО обучения (базовая линия, без неё непонятно, что дало дообучение):")
        t0 = time.time()
        baseline = trainer.evaluate(metric_key_prefix="base")
        shown = {k: v for k, v in baseline.items() if "rouge" in k or "loss" in k or "gen_len" in k}
        print(f"    {json.dumps(shown, indent=6, ensure_ascii=False)}")
        print(f"    (замер занял {time.time() - t0:.0f} с)\n")

    print("  Обучение:")
    t0 = time.time()
    train_result = trainer.train()
    train_time = time.time() - t0
    print(f"\n  Обучение заняло {train_time / 60:.1f} мин ({train_time:.0f} с)")
    print(f"  Итоговый training loss: {train_result.training_loss:.4f}")

# ----------------------------------------------------------------- ШАГ 11
with Step(11, "Отслеживаем метрики: loss, ROUGE, длина, время, ресурсы"):
    # Историю снимаем ДО финального evaluate: иначе он добавит в неё ещё одну
    # запись и последняя эпоха попадёт в таблицу дважды.
    hist = list(trainer.state.log_history)
    final = trainer.evaluate(metric_key_prefix="eval")

    print("  Метрики по эпохам:")
    header = f"    {'эпоха':>6} {'train loss':>11} {'val loss':>10} {'rouge1':>8} {'rouge2':>8} {'rougeL':>8} {'gen_len':>8}"
    print(header)
    tr_points = [(h["epoch"], h["loss"]) for h in hist if "loss" in h and "eval_loss" not in h]
    for h in hist:
        if "eval_loss" not in h:
            continue
        ep = h["epoch"]
        near = [l for e, l in tr_points if e <= ep + 1e-6]
        tl = near[-1] if near else float("nan")
        print(
            f"    {ep:>6.2f} {tl:>11.4f} {h['eval_loss']:>10.4f} {h.get('eval_rouge1', 0):>8.2f} "
            f"{h.get('eval_rouge2', 0):>8.2f} {h.get('eval_rougeL', 0):>8.2f} {h.get('eval_gen_len', 0):>8.1f}"
        )

    if baseline:
        print("\n  ДО обучения vs ПОСЛЕ обучения:")
        print(f"    {'метрика':<12}{'до':>10}{'после':>10}{'прирост':>12}")
        for m in ("loss", "rouge1", "rouge2", "rougeL", "rougeLsum", "gen_len"):
            b, f_ = baseline.get(f"base_{m}"), final.get(f"eval_{m}")
            if b is None or f_ is None:
                continue
            print(f"    {m:<12}{b:>10.3f}{f_:>10.3f}{f_ - b:>+12.3f}")

    try:
        import psutil

        mem = f"{psutil.Process().memory_info().rss / 1e9:.2f} ГБ (оперативная память процесса)"
    except Exception:
        mem = "не измерено"
    if HAS_GPU:
        mem = f"{torch.cuda.max_memory_allocated() / 1e9:.2f} ГБ (пик VRAM), " + mem
    sps = train_result.metrics.get("train_samples_per_second", 0)
    print(f"\n  Время обучения: {train_time / 60:.1f} мин на {device.upper()}")
    print(f"  Память:         {mem}")
    print(f"  Скорость:       {sps:.2f} примеров/с")

    print("\n  Примеры генерации после обучения:")
    model.eval()
    for i in range(min(2, len(raw_datasets["validation"]))):
        row = raw_datasets["validation"][i]
        enc = tokenizer(
            prefix + row["document"], max_length=ARGS.max_input, truncation=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_length=ARGS.max_target, num_beams=1)
        print(f"    --- пример {i + 1} ---")
        print(f"    вход (начало):  {row['document'][:150]!r}")
        print(f"    эталон:         {row['summary'][:150]!r}")
        print(f"    модель выдала:  {tokenizer.decode(out[0], skip_special_tokens=True)[:150]!r}")

    payload = {
        "dataset": ARGS.dataset,
        "model": model_checkpoint,
        "device": device,
        "versions": versions,
        "config": vars(ARGS),
        "rows": sizes,
        "params_total": total,
        "params_trainable": train_p,
        "train_time_sec": round(train_time, 1),
        "train_loss": round(train_result.training_loss, 4),
        "train_samples_per_second": sps,
        "memory": mem,
        "baseline": baseline,
        "final": final,
        "log_history": hist,
    }
    p = RESULTS / f"{ARGS.dataset}_metrics.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Все метрики сохранены: {p}")
