"""Тесты цепочки и скрипта запуска без обращения к n8n (задание 10.6)."""
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402

WF = json.loads((Path(__file__).resolve().parents[1] / "workflow.json")
                .read_text(encoding="utf-8"))
NODES = {n["name"]: n for n in WF["nodes"]}


class FakeResponse(io.BytesIO):
    """Похож на то, что возвращает urlopen: контекстный менеджер с .read() и .status."""

    def __init__(self, payload, status=200):
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        super().__init__(data)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def opener_returning(payload, status=200):
    sent = {}

    def opener(req, timeout=None):
        sent["url"] = req.full_url
        sent["method"] = req.method
        sent["headers"] = dict(req.headers)
        sent["body"] = json.loads(req.data.decode())
        return FakeResponse(payload, status)

    opener.sent = sent
    return opener


@pytest.fixture()
def clean_env(monkeypatch):
    monkeypatch.delenv("N8N_POST_WEBHOOK", raising=False)
    monkeypatch.setenv("N8N_WEBHOOK_TOKEN", "тестовый-токен")


# --------------------------------------------------------------- запрос

def test_sends_topic_as_json_post(clean_env):
    op = opener_returning({"ok": True, "message_id": 42})
    code, body = main.post_topic("Маркировка CE", opener=op)
    assert code == 200 and body["ok"] is True
    assert op.sent["method"] == "POST"
    assert op.sent["headers"]["Content-type"] == "application/json"
    assert op.sent["body"] == {"topic": "Маркировка CE"}


def test_token_goes_in_authorization_header(clean_env):
    op = opener_returning({"ok": True})
    main.post_topic("тема", opener=op)
    assert op.sent["headers"]["Authorization"] == "Bearer тестовый-токен"


def test_no_token_mode_sends_no_header(clean_env):
    """Флагом --no-token проверяется, что защита действительно закрывает вебхук."""
    op = opener_returning({"message": "Authorization failed"}, status=403)
    main.post_topic("тема", send_token=False, opener=op)
    assert "Authorization" not in op.sent["headers"]


def test_style_is_optional(clean_env):
    op = opener_returning({"ok": True})
    main.post_topic("тема", style="деловой", opener=op)
    assert op.sent["body"]["style"] == "деловой"

    op2 = opener_returning({"ok": True})
    main.post_topic("тема", opener=op2)
    assert "style" not in op2.sent["body"]


def test_forbidden_is_reported_not_raised(clean_env):
    def opener(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {},
                                     io.BytesIO(b'{"message":"Authorization failed"}'))

    code, body = main.post_topic("тема", opener=opener)
    assert code == 403 and "Authorization failed" in body


def test_unreachable_n8n_explains_what_to_check(clean_env):
    def opener(req, timeout=None):
        raise urllib.error.URLError("Connection refused")

    with pytest.raises(SystemExit, match="контейнер запущен"):
        main.post_topic("тема", opener=opener)


# --------------------------------------------------------------- вывод

def test_describe_mentions_image_and_followup(capsys):
    main.describe(200, {"ok": True, "message_id": 7, "with_image": True,
                        "text_sent_separately": True,
                        "url": "https://t.me/ch/7"})
    out = capsys.readouterr().out
    assert "Пост опубликован" in out and "с картинкой" in out
    assert "не поместился в подпись" in out


def test_describe_treats_403_as_working_protection(capsys):
    main.describe(403, {"message": "Authorization failed"})
    assert "защита сработала" in capsys.readouterr().out


# --------------------------------------------------------------- командная строка

def test_cli_dry_run_sends_nothing(clean_env, capsys):
    assert main.main(["--dry-run", "Маркировка CE"]) == 0
    out = capsys.readouterr().out
    assert '"topic": "Маркировка CE"' in out and "не отправлялся" in out


def test_cli_no_token_counts_403_as_success(clean_env, monkeypatch):
    monkeypatch.setattr(main, "post_topic", lambda *a, **k: (403, "Authorization failed"))
    assert main.main(["--no-token", "тема"]) == 0      # 403 здесь — ожидаемый исход
    monkeypatch.setattr(main, "post_topic", lambda *a, **k: (200, {"ok": True}))
    assert main.main(["--no-token", "тема"]) == 1      # а 200 означает, что защиты нет


# --------------------------------------------------------------- защита вебхука

def test_webhook_requires_header_auth():
    """Часть 2 задания: вебхук закрыт заголовком, иначе он открыт всему интернету."""
    wh = NODES["Webhook"]
    assert wh["parameters"]["httpMethod"] == "POST"
    assert wh["parameters"]["authentication"] == "headerAuth"


def test_workflow_file_carries_no_secrets():
    raw = json.dumps(WF, ensure_ascii=False)
    assert "$env.TELEGRAM_BOT_TOKEN" in raw          # токен берётся из окружения
    assert not any(s in raw for s in ("sk-", "bot7", "bot8", "Bearer "))


# --------------------------------------------------------------- картинка

def test_image_prompt_is_a_separate_node():
    """Задание требует отдельный узел, который придумывает промпт для картинки."""
    node = NODES["AI — промпт картинки"]
    assert node["type"] == "@n8n/n8n-nodes-langchain.openAi"
    # узел читает готовый текст поста, а не исходную тему
    user_msg = node["parameters"]["messages"]["values"][-1]["content"]
    assert "$json.text" in user_msg


def test_image_node_uses_the_cheap_model():
    """Для учебной цепочки берём дешёвую модель; в продукте остаётся gpt-image-1.

    Запрос собирается вручную, а не готовым узлом OpenAI: тот на любой модели, кроме
    gpt-image-1, добавляет в тело response_format, которого модели серии gpt-image
    не принимают — живой запуск отвечал «Bad request».
    """
    node = NODES["Generate Image"]
    assert node["type"] == "n8n-nodes-base.httpRequest"
    assert node["parameters"]["nodeCredentialType"] == "openAiApi"
    body = node["parameters"]["jsonBody"]
    assert "gpt-image-1-mini" in body
    assert "quality: 'low'" in body
    assert "response_format" not in body


def test_base64_is_turned_into_a_file():
    """Модель отдаёт картинку строкой base64 — без превращения в файл Telegram
    получил бы текст вместо изображения."""
    node = NODES["Image to Binary"]
    assert node["type"] == "n8n-nodes-base.convertToFile"
    assert node["parameters"]["operation"] == "toBinary"
    assert node["parameters"]["sourceProperty"] == "data[0].b64_json"
    assert node["parameters"]["binaryPropertyName"] == "data"
    assert node["parameters"]["options"]["mimeType"] == "image/png"


def test_photo_is_sent_as_binary_not_as_link():
    """Задание требует корректной передачи двоичных данных."""
    node = NODES["Send Photo — Telegram API"]
    assert "sendPhoto" in node["parameters"]["url"]
    assert node["parameters"]["contentType"] == "multipart-form-data"
    params = node["parameters"]["bodyParameters"]["parameters"]
    by_name = {p["name"]: p for p in params}
    assert by_name["photo"]["parameterType"] == "formBinaryData"
    assert by_name["photo"]["inputDataFieldName"] == "data"
    for field in ("chat_id", "caption", "parse_mode"):
        assert field in by_name
    assert by_name["parse_mode"]["value"] == "HTML"


# --------------------------------------------------------------- ограничения

def test_caption_limit_is_1024_not_4096():
    """У подписи к фото предел вчетверо ниже, чем у обычного сообщения.

    Если положить в caption текст длиной 4000 знаков, Telegram ответит ошибкой, и
    публикация упадёт уже после того, как картинка нарисована и оплачена.
    """
    code = NODES["Parse Response"]["parameters"]["jsCode"]
    assert "TG_CAPTION_LIMIT = 1024" in code
    assert "TG_TEXT_LIMIT = 4096" in code
    assert "needsFollowup" in code


def test_image_prompt_is_trimmed():
    """Модели рисования обрезают промпт — укорачиваем его заранее."""
    code = NODES["Prepare Image Brief"]["parameters"]["jsCode"]
    assert "IMAGE_PROMPT_LIMIT = 900" in code


def test_token_limit_is_set_on_the_text_node():
    """Задание прямо требует ограничить количество токенов."""
    options = NODES["AI — текст поста"]["parameters"]["options"]
    assert options["maxTokens"] == 400
    assert options["simplify"] is False     # иначе в ответе не будет structure choices


# --------------------------------------------------------------- порядок узлов

def test_chain_order_matches_the_task():
    conn = WF["connections"]
    order = ["Webhook", "Prepare Prompt", "AI — текст поста", "Parse Response",
             "AI — промпт картинки", "Prepare Image Brief", "Generate Image",
             "Image to Binary", "Send Photo — Telegram API"]
    for src, dst in zip(order, order[1:]):
        targets = [c["node"] for lst in conn[src]["main"] for c in lst]
        assert targets == [dst], f"{src} должен вести в {dst}, а ведёт в {targets}"


def test_long_text_branch_sends_a_second_message():
    """Условный узел: текст не поместился — уходит отдельным сообщением."""
    branches = WF["connections"]["Текст не поместился в подпись?"]["main"]
    assert [c["node"] for c in branches[0]] == ["Send Text — Telegram API"]
    assert [c["node"] for c in branches[1]] == ["Build Result"]
    assert "sendMessage" in NODES["Send Text — Telegram API"]["parameters"]["url"]


# --------------------------------------------------------------- совместимость версий

# Версии узлов, которые поддерживает установленный n8n. Список снят с работающего
# 2.39.8 — у каждого типа узла своя история версий, и угадывать её нельзя.
KNOWN_GOOD_VERSIONS = {
    "n8n-nodes-base.webhook": {1, 1.1, 2, 2.1},
    "n8n-nodes-base.code": {1, 2},
    "n8n-nodes-base.httpRequest": {3, 4, 4.1, 4.2, 4.3, 4.4, 4.5},
    "n8n-nodes-base.if": {1, 2, 2.1, 2.2, 2.3},
    "n8n-nodes-base.respondToWebhook": {1, 1.1, 1.2, 1.3, 1.4, 1.5},
    "n8n-nodes-base.convertToFile": {1, 1.1},
    "@n8n/n8n-nodes-langchain.openAi": {1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8},
}


def test_node_versions_are_supported_by_installed_n8n():
    for node in WF["nodes"]:
        allowed = KNOWN_GOOD_VERSIONS.get(node["type"])
        assert allowed, f"узел {node['type']} не проверен на совместимость"
        assert node["typeVersion"] in allowed, (
            f"{node['name']}: typeVersion={node['typeVersion']}, "
            f"а установленный n8n знает только {sorted(allowed)}")


def test_versions_stay_on_the_conservative_side():
    """Берём младшую версию из современной серии, а не самую свежую.

    Цепочку могут открыть и на более старом n8n — на VPS и на машине в лаборатории
    версии отстают. Младшая версия серии есть везде, самая свежая — только здесь.
    """
    assert NODES["Текст не поместился в подпись?"]["typeVersion"] == 2
    assert NODES["Send Photo — Telegram API"]["typeVersion"] == 4
    assert NODES["AI — промпт картинки"]["typeVersion"] == 1.8
