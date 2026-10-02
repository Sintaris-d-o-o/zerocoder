"""Учебный скрипт: логирование в файл logs/app.log (задание Zerocoder 1.4)."""

import logging
import sys
from pathlib import Path

# Путь к журналу считаем от расположения скрипта, а не от текущей папки:
# так журнал попадёт в cli_practice/logs/, откуда бы скрипт ни запустили.
LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "app.log"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(module)s | %(message)s"
THRESHOLD = 100

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    # Терминал Windows по умолчанию может выводить в cp1252, где нет кириллицы.
    sys.stdout.reconfigure(encoding="utf-8")
    LOG_FILE.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def check_value(x: int) -> None:
    if x > THRESHOLD:
        logger.warning("Значение x=%s выше порога %s", x, THRESHOLD)
    else:
        logger.info("Значение x=%s в норме", x)


def divide(a: float, b: float) -> float | None:
    try:
        return a / b
    except ZeroDivisionError:
        logger.exception("Ошибка деления: %s / %s", a, b)
        return None


def main() -> None:
    setup_logging()
    logger.info("Приложение запущено")
    check_value(150)
    divide(10, 0)
    logger.info("Приложение завершено")


if __name__ == "__main__":
    main()
