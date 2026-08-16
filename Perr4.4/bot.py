"""
Telegram-бот — принимает заявки на консультацию и сохраняет их в общую
таблицу requests.csv (см. website/website_app.py — сайт-версия того же
ассистента, использует тот же "мозг" из assistant.py и ту же таблицу).
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from assistant import next_step
from storage import Lead, save_lead

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
)
# httpx/telegram логируют полный URL запроса, а он содержит токен бота —
# поднимаем уровень, чтобы токен не оседал в лог-файлах.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logger = logging.getLogger("leads-bot")

WELCOME = (
    "Здравствуйте! Я помогу оформить заявку на консультацию.\n\n"
    "Расскажите, пожалуйста, что вас интересует?"
)

# user_id -> история диалога (список {"role", "content"})
conversations: dict[int, list[dict]] = {}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    conversations[update.effective_user.id] = []
    await update.message.reply_text(WELCOME)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    history = conversations.setdefault(user_id, [])
    history.append({"role": "user", "content": update.message.text})

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )
    try:
        result = await next_step(history)
    except Exception:
        logger.exception("Ошибка обращения к ассистенту")
        await update.message.reply_text(
            "Сервис временно недоступен, попробуйте ещё раз чуть позже."
        )
        return

    history.append({"role": "assistant", "content": result["reply"]})
    await update.message.reply_text(result["reply"])

    lead_data = result["lead"]
    if lead_data:
        lead_id = save_lead(
            Lead(
                name=lead_data.get("name", ""),
                phone=lead_data.get("phone", ""),
                comment=lead_data.get("comment", ""),
                consultation_type=lead_data.get("consultation_type", ""),
                source="Telegram-бот",
            )
        )
        logger.info("Заявка #%s сохранена (Telegram)", lead_id)
        # Заявка оформлена — начинаем новый диалог на случай следующего обращения.
        conversations[user_id] = []


def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    logger.info("Бот запущен, ожидаю сообщения...")
    app.run_polling()


if __name__ == "__main__":
    main()
