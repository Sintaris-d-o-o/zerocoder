"""
Локальный сайт-чат — тот же ассистент, что и в Telegram-боте (bot.py),
но в виде веб-страницы. Использует общий "мозг" из assistant.py и пишет
заявки в ту же общую таблицу requests.csv (storage.py), с источником «Сайт».

Запуск (из папки Perr4.4): python website/website_app.py
Затем открыть в браузере: http://localhost:8000
"""

import logging
import sys
from pathlib import Path

# assistant.py и storage.py лежат на уровень выше (в Perr4.4/), а не внутри
# website/ — добавляем родительскую папку в sys.path, чтобы их можно было
# импортировать как обычные модули.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from assistant import next_step
from storage import Lead, save_lead

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
)
logger = logging.getLogger("leads-site")

app = FastAPI(title="Ассистент — заявки на консультацию")
INDEX_HTML = Path(__file__).resolve().parent / "index.html"

# session_id -> история диалога (список {"role", "content"})
sessions: dict[str, list[dict]] = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    lead_saved: bool


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    history = sessions.setdefault(payload.session_id, [])
    history.append({"role": "user", "content": payload.message})

    try:
        result = await next_step(history)
    except Exception:
        logger.exception("Ошибка обращения к ассистенту")
        return ChatResponse(
            reply="Сервис временно недоступен, попробуйте ещё раз чуть позже.",
            lead_saved=False,
        )

    history.append({"role": "assistant", "content": result["reply"]})

    lead_data = result["lead"]
    lead_saved = False
    if lead_data:
        lead_id = save_lead(
            Lead(
                name=lead_data.get("name", ""),
                phone=lead_data.get("phone", ""),
                comment=lead_data.get("comment", ""),
                consultation_type=lead_data.get("consultation_type", ""),
                source="Сайт",
            )
        )
        logger.info("Заявка #%s сохранена (сайт)", lead_id)
        # Заявка оформлена — начинаем новый диалог на случай следующего обращения.
        sessions[payload.session_id] = []
        lead_saved = True

    return ChatResponse(reply=result["reply"], lead_saved=lead_saved)


if __name__ == "__main__":
    logger.info("Запускаю сайт на http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
