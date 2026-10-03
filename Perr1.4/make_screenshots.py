"""Делает скриншоты окна терминала из настоящих протоколов команд (папка протокол/).

Протоколы записаны при выполнении задания в Git Bash; скрипт только оформляет
их в виде окна терминала и снимает картинку через Edge без открытия окна.

Запуск: python make_screenshots.py
"""

import html
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "протокол"
OUT = ROOT / "screenshots"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

PROMPT = re.compile(r"^(\(\.venv\) )?(\S+) \$ (.*)$")
LINE_PX = 20
WIDTH = 1000

# (файл картинки, заголовок, протокол, с какой команды начать)
SHOTS = [
    ("1-структура-проекта.png", "Структура проекта: mkdir, touch, pwd, ls -a, ls -R",
     "1-структура.txt", None),
    ("2-запуск-скрипта.png", "Виртуальное окружение и запуск main.py",
     "2-запуск.txt", None),
    ("3-содержимое-app-log.png", "Содержимое logs/app.log",
     "3-анализ-лога.txt", "cat logs/app.log"),
]


def colorize(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\b(WARNING)\b", r'<span class="warn">\1</span>', text)
    text = re.sub(r"\b(ERROR|ZeroDivisionError)\b", r'<span class="err">\1</span>', text)
    text = re.sub(r"\b(INFO)\b", r'<span class="info">\1</span>', text)
    return text


def render(lines: list[str]) -> tuple[str, int]:
    out, count = [], 0
    for line in lines:
        m = PROMPT.match(line)
        if m:
            venv, path, cmd = m.groups()
            if out:
                out.append("")
                count += 1
            out.append(
                (f'<span class="venv">{venv}</span>' if venv else "")
                + '<span class="user">souls</span> <span class="host">MINGW64</span> '
                + f'<span class="path">{html.escape(path)}</span>'
            )
            out.append(f'<span class="dollar">$</span> {html.escape(cmd)}')
            count += 2
        else:
            out.append(colorize(line))
            count += 1
    return "\n".join(out), count


def page(title: str, body: str) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;background:#2b2b2b;font-family:Consolas,'Cascadia Mono',monospace}}
.bar{{background:#3c3c3c;color:#ddd;font:13px 'Segoe UI',sans-serif;padding:7px 12px}}
pre{{margin:0;padding:10px 14px;color:#e6e6e6;font-size:15px;line-height:{LINE_PX}px;
white-space:pre-wrap;word-break:break-all;background:#0c0c0c}}
.user{{color:#3fbf3f}}.host{{color:#c05fc0}}.path{{color:#d7c440}}.venv{{color:#8ab4f8}}
.dollar{{color:#e6e6e6}}.info{{color:#5fb3f0}}.warn{{color:#f0c050}}.err{{color:#ff6b6b}}
</style></head><body><div class="bar">MINGW64:/c/Users/souls/cli_practice — {html.escape(title)}</div>
<pre>{body}</pre></body></html>"""


def main() -> None:
    OUT.mkdir(exist_ok=True)
    tmp = Path(tempfile.gettempdir())
    for name, title, src, start in SHOTS:
        lines = (PROTOCOL / src).read_text(encoding="utf-8").splitlines()
        if start:
            lines = lines[next(i for i, l in enumerate(lines) if l.endswith("$ " + start)):]
        body, count = render(lines)
        # Edge плохо пишет в пути с кириллицей — снимаем во временную папку и переносим.
        html_file, png = tmp / "terminal_shot.html", tmp / "terminal_shot.png"
        html_file.write_text(page(title, body), encoding="utf-8")
        height = count * LINE_PX + 60
        subprocess.run([str(EDGE), "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        f"--window-size={WIDTH},{height}", f"--screenshot={png}",
                        html_file.as_uri()], check=True, capture_output=True)
        (OUT / name).write_bytes(png.read_bytes())
        print(f"{name}: {count} строк, {WIDTH}x{height}")


if __name__ == "__main__":
    main()
