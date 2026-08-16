"""
Мультимодальный ассистент для описания товаров по фото.

Пользователь присылает боту в Telegram фото товара и голосовое (или текстовое)
сообщение с деталями. Бот распознаёт речь, анализирует фото, генерирует
продающее описание и озвучивает готовый ответ.

Пайплайн (см. 01-концепция-и-план.md):
  Вход -> STT (Whisper) + Vision (GPT-4o) -> LLM (продающий текст) ->
  пост-обработка (структурированный отчёт + TTS) -> Вывод (текст + голос в Telegram)
"""

import io
import logging
from pathlib import Path

from dotenv import load_dotenv
import os

from openai import AsyncOpenAI
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# .env лежит в zerocoder/ — на уровень выше папки этого задания.
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

VISION_MODEL = "gpt-4o-mini"
TEXT_MODEL = "gpt-4o-mini"
STT_MODEL = "whisper-1"
TTS_MODEL = "tts-1"
TTS_VOICE = "alloy"

PROMPT_ANALYZE_PHOTO = (
    "Ты — эксперт по описанию товаров для маркетплейса. Внимательно изучи фото "
    "товара и перечисли фактами: что это за товар, категория, цвет, материал "
    "(если видно), видимое состояние (новое / б/у, есть ли дефекты), заметные "
    "особенности. Пиши только то, что реально видно на фото, без домыслов. "
    "Формат — короткий список фактов."
)

PROMPT_SELLING_TEXT = (
    "На основе фактов о товаре с фото и заметок продавца напиши продающее "
    "описание товара для интернет-площадки. Стиль — дружелюбный и конкретный, "
    "без канцелярита. Структура: заголовок; 3-5 пунктов с ключевыми "
    "характеристиками; короткий призыв к действию. Не добавляй факты, которых "
    "нет в исходных данных — если чего-то не хватает (например, цены), явно "
    "напиши, что это нужно уточнить у продавца.\n\n"
    "Факты с фото:\n{facts}\n\nЗаметки продавца (голос/текст):\n{details}"
)

PROMPT_FINAL_REPORT = (
    "Оформи ответ для продавца в виде короткого структурированного отчёта:\n"
    "1) Готовое описание товара — то, что можно сразу опубликовать;\n"
    "2) Ключевые факты, на которых оно основано;\n"
    "3) Что стоит уточнить или переснять/наговорить заново, если данных не "
    "хватило.\n\n"
    "Продающий текст:\n{selling_text}\n\nФакты с фото:\n{facts}"
)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
)
# httpx/telegram логируют полный URL запроса, а он содержит токен бота —
# поднимаем уровень, чтобы токен не оседал в лог-файлах.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logger = logging.getLogger("multimodal-assistant")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# user_id -> байты последнего присланного фото, ожидающего голос/текст с деталями
pending_photos: dict[int, bytes] = {}


async def transcribe_voice(voice_bytes: bytes) -> str:
    audio_file = io.BytesIO(voice_bytes)
    audio_file.name = "voice.ogg"
    transcript = await client.audio.transcriptions.create(
        model=STT_MODEL, file=audio_file
    )
    return transcript.text


async def analyze_photo(photo_bytes: bytes) -> str:
    import base64

    b64 = base64.b64encode(photo_bytes).decode("utf-8")
    response = await client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_ANALYZE_PHOTO},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content


async def generate_selling_text(facts: str, details: str) -> str:
    response = await client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {
                "role": "user",
                "content": PROMPT_SELLING_TEXT.format(facts=facts, details=details),
            }
        ],
    )
    return response.choices[0].message.content


async def format_final_report(selling_text: str, facts: str) -> str:
    response = await client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {
                "role": "user",
                "content": PROMPT_FINAL_REPORT.format(
                    selling_text=selling_text, facts=facts
                ),
            }
        ],
    )
    return response.choices[0].message.content


async def synthesize_speech(text: str) -> bytes:
    response = await client.audio.speech.create(
        model=TTS_MODEL, voice=TTS_VOICE, input=text
    )
    return response.content


async def process_item(photo_bytes: bytes, details_text: str) -> tuple[str, bytes]:
    facts = await analyze_photo(photo_bytes)
    selling_text = await generate_selling_text(facts, details_text)
    report = await format_final_report(selling_text, facts)
    audio = await synthesize_speech(selling_text)
    return report, audio


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я помогу быстро составить продающее описание товара.\n\n"
        "1. Пришлите фото товара.\n"
        "2. Затем голосовое сообщение (или текст) с деталями: материал, размер, "
        "состояние, цена, особенности.\n\n"
        "Через несколько секунд пришлю готовое текстовое описание и озвучу его."
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    photo = update.message.photo[-1]
    file = await photo.get_file()
    photo_bytes = bytes(await file.download_as_bytearray())
    pending_photos[user_id] = photo_bytes
    await update.message.reply_text(
        "Фото получил. Теперь пришлите голосовое сообщение или текст с деталями "
        "товара (материал, размер, состояние, цена, особенности)."
    )


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if user_id not in pending_photos:
        await update.message.reply_text(
            "Сначала пришлите фото товара, а затем голосовое сообщение с деталями."
        )
        return

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )
    try:
        voice = update.message.voice
        file = await voice.get_file()
        voice_bytes = bytes(await file.download_as_bytearray())
        details_text = await transcribe_voice(voice_bytes)
    except Exception:
        logger.exception("Ошибка распознавания голоса")
        await update.message.reply_text(
            "Не удалось распознать голосовое сообщение. Попробуйте ещё раз или "
            "пришлите детали текстом."
        )
        return

    await _run_pipeline(update, context, user_id, details_text)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if user_id not in pending_photos:
        await update.message.reply_text(
            "Сначала пришлите фото товара, а затем текст (или голосовое "
            "сообщение) с деталями."
        )
        return

    await _run_pipeline(update, context, user_id, update.message.text)


async def _run_pipeline(
    update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int, details_text: str
) -> None:
    photo_bytes = pending_photos.pop(user_id)
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )
    try:
        report, audio_bytes = await process_item(photo_bytes, details_text)
    except Exception:
        logger.exception("Ошибка обработки товара")
        await update.message.reply_text(
            "Не получилось обработать запрос — сервис анализа временно "
            "недоступен. Попробуйте ещё раз чуть позже."
        )
        return

    await update.message.reply_text(report)

    # Отправка голосового ответа отдельно оборачивается в повтор: на практике
    # именно этот сетевой запрос (самый "тяжёлый" по размеру) иногда обрывается
    # по тайм-ауту при нестабильном интернете, хотя сам ответ уже сгенерирован.
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "описание.mp3"
    try:
        await update.message.reply_audio(audio=audio_file)
    except Exception:
        logger.warning("Не удалось отправить аудио с первого раза, повторяю попытку")
        audio_file.seek(0)
        try:
            await update.message.reply_audio(audio=audio_file)
        except Exception:
            logger.exception("Не удалось отправить голосовой ответ")
            await update.message.reply_text(
                "Текст готов (см. выше), а вот голосовое отправить не удалось "
                "из-за сетевого сбоя. Текстовое описание уже можно использовать."
            )


def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    logger.info("Бот запущен, ожидаю сообщения...")
    app.run_polling()


if __name__ == "__main__":
    main()
