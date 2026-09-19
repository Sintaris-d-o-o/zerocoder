"""Веб-приложение «Генератор логотипов» на Flask (задание 10.2).

Доработка консольного генератора из задания 10.1: форма в браузере, выбор стиля,
сборка промпта из частей, галерея сгенерированного и скачивание файла.

Что доработано по сравнению с примером эксперта:
  * генерация вынесена в отдельный поток, страница не «висит» и показывает прогресс;
  * стили — не только для логотипов: есть иконка, баннер и обложка с разным соотношением сторон;
  * результат сохраняется на диск вместе с описанием (какой промпт, какой seed, сколько заняло),
    поэтому удачный логотип можно повторить;
  * галерея ранее сгенерированного с возможностью повторить с тем же зерном;
  * понятные сообщения об ошибках вместо кода HTTP;
  * режим демонстрации без ключа (--demo) — интерфейс можно показать и проверить бесплатно.

Запуск:
    python app.py                 # обычный режим, нужен ключ в .env
    python app.py --demo          # без ключа: вместо картинок рисуются заглушки
    python app.py --port 5001
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template, request, send_from_directory

HERE = Path(__file__).resolve().parent
# Генератор живёт в папке задания 10.1 — это его прямое продолжение, копия кода не нужна:
# одна правка должна действовать в обоих заданиях (урок из задания 8.2).
sys.path.insert(0, str(HERE.parent / "Perr10.1"))

import yandex_art as ya  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RESULTS_DIR = HERE / "results" / "logos"
GALLERY_FILE = HERE / "results" / "gallery.json"

# Стили: к запросу пользователя подмешивается описание с меньшим весом, чем сам запрос.
STYLES: dict[str, dict] = {
    "minimal": {
        "title": "Минимализм",
        "text": "минималистичный плоский векторный знак, простые геометрические формы, "
                "максимум два цвета, много воздуха, без мелких деталей",
    },
    "corporate": {
        "title": "Деловой",
        "text": "строгий корпоративный стиль, сдержанная палитра синего и серого, "
                "симметричная композиция, ощущение надёжности и официальности",
    },
    "medtech": {
        "title": "Медтех",
        "text": "медицинская тематика, чистые линии, бирюзовый и синий, "
                "ассоциации с точностью, безопасностью и заботой о здоровье",
    },
    "tech": {
        "title": "Технологичный",
        "text": "технологичный футуристичный стиль, тонкие светящиеся линии, "
                "градиент, тёмный фон, ощущение цифрового продукта",
    },
    "document": {
        "title": "Документ и печать",
        "text": "мотив документа, печати или подписи, официальный знак соответствия, "
                "тонкая линия, сдержанные цвета",
    },
    "vintage": {
        "title": "Винтаж",
        "text": "винтажный стиль, классическая эмблема, тонкая штриховка, "
                "приглушённая охристая палитра",
    },
}

FORMATS: dict[str, dict] = {
    "logo":   {"title": "Логотип 1:1", "w": "1", "h": "1"},
    "icon":   {"title": "Иконка 1:1", "w": "1", "h": "1"},
    "banner": {"title": "Баннер 16:9", "w": "16", "h": "9"},
    "story":  {"title": "Вертикальный 9:16", "w": "9", "h": "16"},
}

# Хвост промпта: для знака важно запретить надписи — модели склонны рисовать нечитаемые буквы.
NO_TEXT_TAIL = "без текста, без букв и цифр, на белом фоне, чёткие края"


@dataclass
class Job:
    """Одна задача генерации. Живёт в памяти, пока приложение запущено."""
    job_id: str
    prompt: str
    company: str
    style: str
    fmt: str
    seed: Optional[int]
    status: str = "queued"          # queued → running → done | error
    message: str = ""
    filename: str = ""
    elapsed_s: float = 0.0
    polls: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def build_prompt(company: str, style: str, extra: str, fmt: str) -> str:
    """Собирает итоговый промпт из частей: что рисуем, в каком стиле, что запрещено."""
    what = {
        "logo": f'Логотип компании «{company}»',
        "icon": f'Иконка приложения «{company}»',
        "banner": f'Широкий баннер для сайта компании «{company}»',
        "story": f'Вертикальная обложка для компании «{company}»',
    }.get(fmt, f'Логотип компании «{company}»')

    parts = [what + "."]
    if style in STYLES:
        parts.append(STYLES[style]["text"] + ".")
    if extra.strip():
        parts.append(extra.strip().rstrip(".") + ".")
    parts.append(NO_TEXT_TAIL + ".")
    return " ".join(parts)


def _demo_image(prompt: str) -> bytes:
    """Заглушка для режима демонстрации: картинка-плашка без обращения к платному API."""
    from PIL import Image, ImageDraw
    import textwrap
    img = Image.new("RGB", (512, 512), "#0f2b46")
    d = ImageDraw.Draw(img)
    d.ellipse([146, 116, 366, 336], outline="#4fd1c5", width=6)
    d.line([196, 226, 236, 276, 316, 176], fill="#4fd1c5", width=10)
    y = 380
    for line in textwrap.wrap(prompt, 46)[:4]:
        d.text((24, y), line, fill="#c9d6e4")
        y += 18
    d.text((24, 470), "РЕЖИМ ДЕМОНСТРАЦИИ — обращения к API не было", fill="#eda100")
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def run_job(job: Job, demo: bool) -> None:
    """Выполняется в отдельном потоке, чтобы браузер не ждал ответа минуту."""
    with JOBS_LOCK:
        job.status = "running"
    started = datetime.now()
    try:
        if demo:
            import time as _t
            _t.sleep(1.5)
            image_bytes = _demo_image(job.prompt)
            elapsed, polls, seed = 1.5, 1, job.seed
        else:
            f = FORMATS.get(job.fmt, FORMATS["logo"])

            def on_poll(n: int, _elapsed: float) -> None:
                with JOBS_LOCK:
                    job.polls = n

            res = ya.generate_image(job.prompt, job.seed, width_ratio=f["w"], height_ratio=f["h"],
                                    on_poll=on_poll)
            image_bytes, elapsed, polls, seed = res.image_bytes, res.elapsed_s, res.polls, res.seed

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = started.strftime("%Y%m%d-%H%M%S")
        safe_company = "".join("_" if ch in ':<>"/\\|?* ' else ch for ch in job.company)[:40]
        filename = f"{stamp}_{job.style}_{safe_company or 'logo'}.jpeg"
        (RESULTS_DIR / filename).write_bytes(image_bytes)

        with JOBS_LOCK:
            job.status = "done"
            job.filename = filename
            job.elapsed_s = elapsed
            job.polls = polls
            job.seed = seed
        append_gallery(job)
    except ya.YandexArtError as exc:
        with JOBS_LOCK:
            job.status, job.message = "error", str(exc)
    except Exception as exc:  # noqa: BLE001 — в веб-интерфейсе любая ошибка должна стать текстом
        with JOBS_LOCK:
            job.status, job.message = "error", f"Непредвиденная ошибка: {exc}"


def append_gallery(job: Job) -> None:
    """Запоминает удачный результат вместе с промптом и зерном, чтобы его можно было повторить."""
    GALLERY_FILE.parent.mkdir(parents=True, exist_ok=True)
    items = load_gallery()
    items.insert(0, asdict(job))
    GALLERY_FILE.write_text(json.dumps(items[:60], ensure_ascii=False, indent=2), encoding="utf-8")


def load_gallery() -> list[dict]:
    if not GALLERY_FILE.exists():
        return []
    try:
        return json.loads(GALLERY_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def create_app(demo: bool = False) -> Flask:
    app = Flask(__name__)
    app.config["DEMO"] = demo

    @app.route("/")
    def index():
        return render_template("index.html", styles=STYLES, formats=FORMATS,
                               demo=demo, gallery=load_gallery()[:12])

    @app.post("/generate")
    def generate():
        data = request.get_json(silent=True) or {}
        company = (data.get("company") or "").strip()
        if not company:
            return jsonify({"error": "Введите название компании"}), 400
        if len(company) > 80:
            return jsonify({"error": "Название слишком длинное (максимум 80 символов)"}), 400

        style = data.get("style") or "minimal"
        fmt = data.get("format") or "logo"
        extra = (data.get("extra") or "").strip()
        seed_raw = data.get("seed")
        try:
            seed = int(seed_raw) if str(seed_raw).strip() not in ("", "None", "null") else None
        except (TypeError, ValueError):
            return jsonify({"error": "Зерно должно быть целым числом"}), 400

        prompt = build_prompt(company, style, extra, fmt)
        job = Job(job_id=uuid.uuid4().hex[:12], prompt=prompt, company=company,
                  style=style, fmt=fmt, seed=seed)
        with JOBS_LOCK:
            JOBS[job.job_id] = job
        threading.Thread(target=run_job, args=(job, app.config["DEMO"]), daemon=True).start()
        return jsonify({"job_id": job.job_id, "prompt": prompt})

    @app.get("/status/<job_id>")
    def status(job_id: str):
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is None:
                return jsonify({"error": "Задача не найдена"}), 404
            return jsonify(asdict(job))

    @app.get("/image/<path:filename>")
    def image(filename: str):
        return send_from_directory(RESULTS_DIR, filename)

    @app.get("/download/<path:filename>")
    def download(filename: str):
        return send_from_directory(RESULTS_DIR, filename, as_attachment=True)

    @app.get("/health")
    def health():
        import os
        return jsonify({
            "demo": app.config["DEMO"],
            "api_key_set": bool(os.getenv("YANDEX_API_KEY")),
            "folder_set": bool(os.getenv("YANDEX_FOLDER_ID") or os.getenv("YANDEX_CLOUD_ID")),
            "styles": len(STYLES), "formats": len(FORMATS), "gallery": len(load_gallery()),
        })

    return app


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="без ключа: рисовать заглушки вместо генерации")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)

    import os
    if not args.demo and not os.getenv("YANDEX_API_KEY"):
        print("YANDEX_API_KEY не найден в .env.", file=sys.stderr)
        print("Запустите с --demo, чтобы посмотреть интерфейс без ключа.", file=sys.stderr)
        return 2

    app = create_app(demo=args.demo)
    режим = "ДЕМОНСТРАЦИЯ (без обращения к API)" if args.demo else "рабочий"
    print(f"Генератор логотипов запущен в режиме: {режим}")
    print(f"Откройте http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
