"""Тесты скрипта запуска автопостинга без обращения к n8n (задание 10.5)."""
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402


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
    monkeypatch.delenv("N8N_WEBHOOK_TOKEN", raising=False)


# --------------------------------------------------------------- запрос

def test_sends_topic_as_json_post(clean_env):
    op = opener_returning({"ok": True, "message_id": 42})
    code, body = main.post_topic("AI в маркетинге", opener=op)
    assert code == 200 and body["ok"] is True
    assert op.sent["method"] == "POST"
    assert op.sent["headers"]["Content-type"] == "application/json"
    assert op.sent["body"] == {"topic": "AI в маркетинге"}   # параметр topic, как требует задание


def test_style_is_optional(clean_env):
    op = opener_returning({"ok": True})
    main.post_topic("тема", style="деловой", opener=op)
    assert op.sent["body"]["style"] == "деловой"

    op2 = opener_returning({"ok": True})
    main.post_topic("тема", opener=op2)
    assert "style" not in op2.sent["body"]      # пустой стиль не уходит в запрос


def test_webhook_address_comes_from_env(monkeypatch):
    monkeypatch.setenv("N8N_POST_WEBHOOK", "http://example.test/webhook/x")
    assert main.webhook_url() == "http://example.test/webhook/x"
    monkeypatch.delenv("N8N_POST_WEBHOOK")
    assert main.webhook_url() == main.DEFAULT_WEBHOOK


def test_token_is_sent_as_header(monkeypatch, clean_env):
    monkeypatch.setenv("N8N_WEBHOOK_TOKEN", "секрет")
    op = opener_returning({"ok": True})
    main.post_topic("тема", opener=op)
    assert op.sent["headers"]["X-webhook-secret"] == "секрет"


def test_explicit_url_wins(clean_env):
    op = opener_returning({"ok": True})
    main.post_topic("тема", url="http://other.test/hook", opener=op)
    assert op.sent["url"] == "http://other.test/hook"


# --------------------------------------------------------------- ответы

def test_non_json_answer_is_returned_as_text(clean_env):
    op = opener_returning(b"Workflow was started")
    code, body = main.post_topic("тема", opener=op)
    assert code == 200 and body == "Workflow was started"


def test_http_error_is_reported_not_raised(clean_env):
    def opener(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {},
                                     io.BytesIO(b'{"message":"webhook not registered"}'))

    code, body = main.post_topic("тема", opener=opener)
    assert code == 404
    assert "not registered" in body          # текст ошибки виден целиком


def test_unreachable_n8n_explains_what_to_check(clean_env):
    def opener(req, timeout=None):
        raise urllib.error.URLError("Connection refused")

    with pytest.raises(SystemExit, match="контейнер запущен"):
        main.post_topic("тема", opener=opener)


# --------------------------------------------------------------- вывод

def test_describe_shows_link_and_id(capsys):
    main.describe(200, {"ok": True, "message_id": 7, "url": "https://t.me/ch/7"})
    out = capsys.readouterr().out
    assert "успех" in out and "Пост опубликован" in out
    assert "https://t.me/ch/7" in out and "7" in out


def test_describe_shows_error_from_workflow(capsys):
    main.describe(200, {"ok": False, "error": "chat not found"})
    out = capsys.readouterr().out
    assert "chat not found" in out


def test_describe_marks_non_200(capsys):
    main.describe(500, "Internal Server Error")
    assert "не 200" in capsys.readouterr().out


# --------------------------------------------------------------- командная строка

def test_cli_dry_run_sends_nothing(clean_env, capsys):
    assert main.main(["--dry-run", "AI в маркетинге"]) == 0
    out = capsys.readouterr().out
    assert '"topic": "AI в маркетинге"' in out
    assert "не отправлялся" in out


def test_cli_topic_can_be_positional_or_flag(clean_env, capsys):
    main.main(["--dry-run", "позиционная"])
    assert "позиционная" in capsys.readouterr().out
    main.main(["--dry-run", "--topic", "флагом"])
    assert "флагом" in capsys.readouterr().out


def test_cli_returns_error_code_on_failure(clean_env, monkeypatch, capsys):
    monkeypatch.setattr(main, "post_topic", lambda *a, **k: (502, "Bad Gateway"))
    assert main.main(["тема"]) == 1


# --------------------------------------------------------------- цепочка n8n

def test_workflow_file_matches_the_task():
    """Проверяем, что цепочка состоит из узлов, которые требует задание."""
    wf = json.loads((Path(__file__).resolve().parents[1] / "workflow.json")
                    .read_text(encoding="utf-8"))
    types = [n["type"] for n in wf["nodes"]]
    assert "n8n-nodes-base.webhook" in types          # вебхук
    assert "@n8n/n8n-nodes-langchain.openAi" in types  # модель
    assert "n8n-nodes-base.httpRequest" in types      # запрос к Telegram API

    webhook = next(n for n in wf["nodes"] if n["type"].endswith("webhook"))
    assert webhook["parameters"]["httpMethod"] == "POST"     # метод POST, как требует задание

    http = next(n for n in wf["nodes"] if n["type"].endswith("httpRequest"))
    assert "api.telegram.org" in http["parameters"]["url"]
    assert "sendMessage" in http["parameters"]["url"]
    body = http["parameters"]["jsonBody"]
    for field in ("chat_id", "text", "parse_mode"):
        assert field in body                                  # три обязательных поля
    assert "HTML" in body

    # ни один узел не должен содержать ключей в открытом виде
    raw = json.dumps(wf, ensure_ascii=False)
    assert "$env.TELEGRAM_BOT_TOKEN" in raw                  # токен берётся из окружения
    assert not any(s in raw for s in ("sk-", "bot7", "bot8"))


def test_workflow_is_a_single_chain():
    wf = json.loads((Path(__file__).resolve().parents[1] / "workflow.json")
                    .read_text(encoding="utf-8"))
    conn = wf["connections"]
    order = ["Webhook", "Prepare Prompt", "AI — Message a model", "Parse Response",
             "HTTP Request — Telegram API", "Build Result"]
    for src, dst in zip(order, order[1:] + ["Respond to Webhook"]):
        targets = [c["node"] for lst in conn[src]["main"] for c in lst]
        assert targets == [dst], f"{src} должен вести в {dst}, а ведёт в {targets}"


def test_parse_node_uses_index_zero():
    """Задание требует обратиться к вложенной структуре ответа с индексом [0]."""
    wf = json.loads((Path(__file__).resolve().parents[1] / "workflow.json")
                    .read_text(encoding="utf-8"))
    parse = next(n for n in wf["nodes"] if n["name"] == "Parse Response")
    assert "choices?.[0]" in parse["parameters"]["jsCode"]


# --------------------------------------------------------------- совместимость версий

# Версии узлов, которые поддерживает установленный n8n (проверено на 2.39.8
# командой grep по описаниям узлов внутри контейнера) и которые уже используются
# в рабочих процессах Taris.
KNOWN_GOOD_VERSIONS = {
    "n8n-nodes-base.webhook": {2},
    "n8n-nodes-base.code": {2},
    "n8n-nodes-base.httpRequest": {4, 4.1, 4.2},
    "n8n-nodes-base.respondToWebhook": {1, 1.1},
    # «Message a Model» — это узел из набора langchain, а не старый n8n-nodes-base.openAi.
    # Установленный n8n знает версии 1…1.8 и 2…2.3, по умолчанию ставит 2.3.
    # Берём 1.8: она есть и в новом n8n, и в более старых, где серии 2.x ещё нет.
    "@n8n/n8n-nodes-langchain.openAi": {1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8},
    # старый узел — оставлен в таблице, потому что он используется в Taris
    "n8n-nodes-base.openAi": {1, 1.1},
}


def test_node_versions_are_supported_by_installed_n8n():
    """Версия узла, которой нет в установленном n8n, не загрузится при импорте.

    Ошибка ловится здесь, а не после развёртывания: узел OpenAI в n8n 2.17.7
    поддерживает только версии 1 и 1.1, хотя у других узлов номера ушли далеко вперёд.
    """
    wf = json.loads((Path(__file__).resolve().parents[1] / "workflow.json")
                    .read_text(encoding="utf-8"))
    for node in wf["nodes"]:
        allowed = KNOWN_GOOD_VERSIONS.get(node["type"])
        assert allowed, f"узел {node['type']} не проверен на совместимость"
        assert node["typeVersion"] in allowed, (
            f"{node['name']}: typeVersion={node['typeVersion']}, "
            f"а установленный n8n знает только {sorted(allowed)}")


def test_model_node_is_not_the_legacy_completion_node():
    """Узел n8n-nodes-base.openAi — не тот, который просит задание.

    У него версии 1 и 1.1, он существует и импортируется без ошибок, но по умолчанию
    обращается к старому endpoint завершения текста: системная роль и заданный промпт
    игнорируются, а на выходе приходит посторонний текст вместо ответа на тему.
    Проверено на живом запуске: модель вернула рекламный отрывок про «DAILY DEALS».
    Операция «Message a Model» живёт в узле @n8n/n8n-nodes-langchain.openAi.
    """
    wf = json.loads((Path(__file__).resolve().parents[1] / "workflow.json")
                    .read_text(encoding="utf-8"))
    model = next(n for n in wf["nodes"] if n["name"] == "AI — Message a model")
    assert model["type"] == "@n8n/n8n-nodes-langchain.openAi"
    assert model["parameters"]["options"]["simplify"] is False, (
        "без simplify=False узел отдаёт уже разобранный текст, "
        "и обращаться к choices[0], как требует задание, будет не к чему")


def test_versions_match_the_taris_workflows():
    """Общие с Taris узлы должны стоять тех же версий.

    Если продукт переедет на новую версию n8n, ломаться будут обе цепочки разом,
    а не по очереди — и чинить придётся один раз. Узел модели намеренно другой:
    в Taris он исторический (n8n-nodes-base.openAi), у нас — тот, что требует урок.
    """
    wf = json.loads((Path(__file__).resolve().parents[1] / "workflow.json")
                    .read_text(encoding="utf-8"))
    ours = {n["type"]: n["typeVersion"] for n in wf["nodes"]}
    taris = {                      # из src/n8n/workflows/Taris - Content Generate.json
        "n8n-nodes-base.webhook": 2,
        "n8n-nodes-base.code": 2,
        "n8n-nodes-base.respondToWebhook": 1,
    }
    for node_type, version in taris.items():
        assert ours.get(node_type) == version, (
            f"{node_type}: у нас {ours.get(node_type)}, у Taris {version}")
