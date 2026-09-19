"""Тесты логики агента без обращения к моделям (подменённые бэкенды)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent as ag  # noqa: E402


class FakeBackends(ag.Backends):
    """Отвечает заранее заданными строками и считает вызовы по моделям."""

    def __init__(self, intent_reply="simple"):
        super().__init__()
        self.intent_reply = intent_reply
        self.calls = []

    def chat_openai(self, model, system, user, max_tokens, reasoning_effort):
        self.calls.append(("openai", model))
        if system == ag.JUDGE_PROMPT:
            return 'Sure. {"score": 4, "reason": "ok"} ', {"tokens_in": 100, "tokens_out": 20,
                                                          "reasoning_tokens": 10, "cached_tokens": 0}
        return f"answer from {model}", {"tokens_in": 50, "tokens_out": 200, "reasoning_tokens": 120,
                                        "cached_tokens": 0}

    def chat_ollama(self, model, system, user, max_tokens, reasoning_effort):
        self.calls.append(("ollama", model))
        if system == ag.CLASSIFIER_PROMPT:
            return self.intent_reply, {"tokens_in": 60, "tokens_out": 2}
        return "local answer", {"tokens_in": 40, "tokens_out": 80}


@pytest.fixture()
def pricing(monkeypatch):
    monkeypatch.setattr(ag, "PRICING", {"models": {
        "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.0},
        "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.0},
        "gemma4": {"input": 0, "cached_input": 0, "output": 0},
    }})


@pytest.mark.parametrize("raw,expected", [
    ("simple", "simple"), ("Hard.", "hard"), ("  medium\n", "medium"),
    ("The tier is: hard", "hard"), ("I think this is Simple!", "simple"),
    ("", "hard"), ("не знаю", "hard"),          # непонятный ответ → самая сильная модель
])
def test_parse_tier(raw, expected):
    assert ag.parse_tier(raw) == expected


def test_parse_judge_json_and_fallback():
    assert ag.parse_judge('{"score": 5, "reason": "great"}') == (5, "great")
    assert ag.parse_judge('text before {"score": "3", "reason": "meh"} after')[0] == 3
    assert ag.parse_judge("Score: 2 out of 5")[0] == 2
    assert ag.parse_judge("no number here")[0] is None
    assert ag.parse_judge('{"score": 9}')[0] is None


def test_compute_cost(pricing):
    # 1000 входных + 500 выходных токенов gpt-5: 1000*1.25/1e6 + 500*10/1e6
    assert ag.compute_cost("gpt-5", 1000, 500) == pytest.approx(0.00125 + 0.005)
    # кэшированные входные токены дешевле
    assert ag.compute_cost("gpt-5", 1000, 0, cached_tokens=1000) == pytest.approx(0.000125)
    assert ag.compute_cost("gemma4:12b", 5000, 5000) == 0
    with pytest.raises(KeyError):
        ag.compute_cost("unknown-model", 1, 1)


@pytest.mark.parametrize("intent_reply,expected_provider,expected_model", [
    ("simple", "ollama", ag.OLLAMA_MODEL),
    ("medium", "openai", "gpt-5-mini"),
    ("hard", "openai", "gpt-5"),
    ("???", "openai", "gpt-5"),
])
def test_route_sends_to_right_model(pricing, intent_reply, expected_provider, expected_model):
    fb = FakeBackends(intent_reply=intent_reply)
    a = ag.Agent(backends=fb)
    out = a.route("What is a notified body?")
    assert out["answer"].provider == expected_provider
    assert out["answer"].model == expected_model
    assert fb.calls[0] == ("ollama", ag.OLLAMA_MODEL)            # первым всегда дежурный
    assert out["cost_usd"] == pytest.approx(out["intent"].cost_usd + out["answer"].cost_usd)
    assert out["latency_s"] == pytest.approx(out["intent"].latency_s + out["answer"].latency_s, abs=1e-3)


def test_cost_uses_all_completion_tokens_including_reasoning(pricing):
    a = ag.Agent(backends=FakeBackends())
    res = a.call_model("q", "hard")
    assert res.reasoning_tokens == 120
    assert res.cost_usd == pytest.approx((50 * 1.25 + 200 * 10) / 1e6)


def test_judge_parses_score(pricing):
    a = ag.Agent(backends=FakeBackends())
    score, reason, res = a.judge("q", "a")
    assert score == 4 and reason == "ok" and res.role == "judge"


def test_cache_roundtrip(pricing, tmp_path):
    path = tmp_path / "cache.jsonl"
    fb = FakeBackends()
    a = ag.Agent(backends=fb, cache=ag.Cache(path))
    first = a.call_model("q", "hard", namespace="A")
    again = a.call_model("q", "hard", namespace="A")
    other_ns = a.call_model("q", "hard", namespace="B")
    assert first.cached_call is False and again.cached_call is True and other_ns.cached_call is False
    assert len([c for c in fb.calls if c[1] == "gpt-5"]) == 2   # второй одинаковый вызов не ушёл в сеть
    # перечитываем с диска
    b = ag.Agent(backends=FakeBackends(), cache=ag.Cache(path))
    assert b.call_model("q", "hard", namespace="A").answer == first.answer
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
