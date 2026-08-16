"""
Общий «мозг» ассистента — используется и Telegram-ботом (bot.py), и сайтом
(website/website_app.py), чтобы оба канала вели диалог одинаково.

Ведёт свободный разговор и, когда собраны все данные, сам решает вызвать
функцию save_lead (механизм function calling в OpenAI) — тогда next_step()
возвращает не только текстовый ответ собеседнику, но и готовые данные
заявки для сохранения в общую таблицу (storage.py).
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

# .env лежит в zerocoder/ — на уровень выше папки этого задания.
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "Ты — дружелюбный консультант сервиса для экспертов и авторов онлайн-курсов "
    "(объединяет общение с учениками, обучение и CRM в одном месте). Твоя задача — "
    "в свободном разговоре узнать у собеседника: (1) имя, (2) телефон для связи, "
    "(3) что именно его интересует — коротко одним-двумя предложениями "
    "(комментарий), (4) к какому типу относится его вопрос: демо платформы, "
    "вопросы по CRM, цены и тарифы, или другое. Спрашивай по одному-два пункта "
    "за раз, живым языком, не как анкету. Когда соберёшь все четыре пункта — "
    "вызови функцию save_lead и вежливо сообщи собеседнику, что заявка оформлена "
    "и с ним свяжутся. Не выдумывай данные, которых человек не называл. Общайся "
    "на русском языке, коротко и по-человечески."
)

SAVE_LEAD_TOOL = {
    "type": "function",
    "function": {
        "name": "save_lead",
        "description": (
            "Сохранить оформленную заявку на консультацию, когда все данные "
            "собраны и подтверждены собеседником."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Имя собеседника"},
                "phone": {"type": "string", "description": "Телефон для связи"},
                "comment": {
                    "type": "string",
                    "description": "Суть вопроса собеседника, коротко",
                },
                "consultation_type": {
                    "type": "string",
                    "description": "Тип консультации",
                    "enum": [
                        "Демо платформы",
                        "Вопросы по CRM",
                        "Цены и тарифы",
                        "Другое",
                    ],
                },
            },
            "required": ["name", "phone", "comment", "consultation_type"],
        },
    },
}

client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def next_step(history: list[dict]) -> dict:
    """
    history — список сообщений диалога без системного промпта:
    [{"role": "user"/"assistant", "content": "..."}, ...].

    Возвращает {"reply": текст для собеседника, "lead": dict | None} —
    lead заполнен, только если модель в этом шаге вызвала save_lead.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=[SAVE_LEAD_TOOL],
    )
    message = response.choices[0].message

    if not message.tool_calls:
        return {"reply": message.content, "lead": None}

    call = message.tool_calls[0]
    lead = json.loads(call.function.arguments)

    # Модель вызвала функцию вместо ответа собеседнику — просим её отдельным
    # запросом сформулировать подтверждение, подставив результат вызова.
    assistant_message = {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
        ],
    }
    follow_up = await client.chat.completions.create(
        model=MODEL,
        messages=[
            *messages,
            assistant_message,
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": "Заявка успешно сохранена.",
            },
        ],
    )
    reply = follow_up.choices[0].message.content
    return {"reply": reply, "lead": lead}
