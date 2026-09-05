# Демонстрация задания 8.2

## Что показывать проверяющему

1. [результаты.md](результаты.md) — отчёт: что сделано по каждому из 11 пунктов задания,
   таблицы метрик, выводы.
2. [results/steps/](results/steps/) — **«скриншоты» всех 11 этапов**: реальный вывод запуска,
   отрисованный в PNG (`xsum_01_*.png` … `xsum_11_*.png` и то же для `normassist_*`).
   Рядом лежат исходные `.txt` — тот же текст, если нужно скопировать.
3. [results/](results/) — графики: `tokenizer_unk.png`, `xsum_training.png`,
   `normassist_training.png`, `rouge_before_after.png` и JSON со всеми числами.
4. [PE8.2_finetuning_T5_colab.ipynb](PE8.2_finetuning_T5_colab.ipynb) — ноутбук для Colab
   со всеми 11 шагами: можно открыть и перезапустить.
5. [results/colab/](results/colab/) — **выполненный прогон на GPU Tesla T4**:
   `PE8.2_colab_run_T4_xsum.ipynb` (ноутбук с сохранённым выводом каждой ячейки),
   `report_xsum_colab.txt`, `metrics_xsum_colab.json`, `training_xsum_colab.png`
   и `gpu_xsum_colab.txt` с выводом `nvidia-smi`.

## Как повторить у себя

### Вариант 1 — Google Colab (с GPU, как в уроке)

1. Открыть [PE8.2_finetuning_T5_colab.ipynb](PE8.2_finetuning_T5_colab.ipynb) в Colab
   (загрузить файл или открыть с Google Drive).
2. Runtime → Change runtime type → **T4 GPU** → Save.
3. Выполнить ячейки сверху вниз. Токен Hugging Face и ключ wandb **не нужны**:
   в ноутбуке `push_to_hub=False` и `report_to=[]`.
4. Для эксперимента на собственных данных: поставить `DATASET = "normassist"` и
   загрузить в Colab (Files → Upload) файлы `train.jsonl` и `validation.jsonl`,
   полученные скриптом подготовки данных (см. ниже).

### Как забрать результаты прогона из Colab (доказательства для сдачи)

Шаг 12 ноутбука сам складывает всё нужное на Google Drive, в папку
`My Drive/Colab Notebooks/sintaris/perr8.2/`:

| Файл | Что доказывает |
|---|---|
| `metrics_<датасет>_colab.json` | все числа прогона целиком, включая историю по шагам |
| `report_<датасет>_colab.txt` | пункт 11 задания: loss, ROUGE, gen_len по эпохам, «до и после», время обучения |
| `training_<датасет>_colab.png` | графики loss и ROUGE |
| `gpu_<датасет>_colab.txt` | вывод `nvidia-smi` — подтверждение, что обучение шло на GPU |

Плюс **скачать сам ноутбук с выводами**: в Colab File → Download → Download .ipynb.
Это главный документ — в нём сохранён вывод каждой ячейки, то есть «выполнение каждого
этапа» в исходном виде, а не пересказ. Скачанный файл положить в ту же папку на Drive.

Дальше на компьютере (Google Drive смонтирован как `J:`) одной командой:

```bash
python code/import_colab_results.py
```

Скрипт перенесёт всё в `results/colab/` — оттуда файлы попадут в репозиторий вместе с
остальными материалами задания.

### Вариант 2 — локально (то, как получены результаты в этой папке)

```bash
cd Perr8.2
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m pip install "transformers>=4.44" "datasets>=3.0" "evaluate>=0.4" \
    rouge-score nltk sentencepiece accelerate matplotlib

# проверка токенизатора на 4 языках (нужен доступ к данным задания 8.1)
./.venv/Scripts/python.exe code/step0_tokenizer_check.py

# эксперимент A — эталон задания, XSum
./.venv/Scripts/python.exe code/finetune_t5.py --dataset xsum \
    --train-rows 2000 --val-rows 200 --epochs 2 --batch-size 8

# эксперимент B — собственные данные из задания 8.1
./.venv/Scripts/python.exe code/prepare_normassist_seq2seq.py
./.venv/Scripts/python.exe code/finetune_t5.py --dataset normassist \
    --train-rows 400 --val-rows 40 --epochs 4 --batch-size 4 --max-target 128

# графики и «скриншоты» этапов
./.venv/Scripts/python.exe code/make_charts.py
./.venv/Scripts/python.exe code/make_step_images.py
```

Эксперимент A (XSum) повторяется **полностью** и без каких-либо доступов — датасет
скачивается с Hugging Face автоматически.

## Почему подготовленных данных эксперимента B нет в репозитории

Файлы `data/normassist_seq2seq/*.jsonl` намеренно исключены из репозитория
(`.gitignore`). Причина — не размер, а авторское право: 104 из 396 строк содержат в
контексте короткие выдержки из стандартов **ISO 13485 / ISO 14971**. Эти стандарты
лицензированные, и правило проекта, откуда взяты данные (задание 8.1), запрещает
класть их текст в публичный репозиторий даже фрагментами.

Кто имеет доступ к данным задания 8.1 — получит те же файлы одной командой
`python code/prepare_normassist_seq2seq.py` (отбор строк детерминированный, `seed=42`).
Все результаты, метрики и «скриншоты» эксперимента B в папке есть — не публикуется
только сам обучающий текст.

## Что не публикуется ещё

- `.venv/` — локальное окружение Python (~2 ГБ);
- `runs/` — служебные каталоги обучения;
- презентация эксперта `.pptx` (4 МБ) — ссылка на оригинал есть в [00-задание.md](00-задание.md).
