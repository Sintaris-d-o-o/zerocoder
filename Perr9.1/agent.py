"""Мультимодельный агент для Норм-ассистента SINTARIS (задание 9.1).

Роли:
  intent  — «дежурный»: локальная Gemma 4 (Ollama) читает вопрос и отвечает одним словом
            simple / medium / hard;
  simple  — локальная Gemma 4 отвечает на простые вопросы (цена 0);
  medium  — gpt-5-mini отвечает на средние вопросы;
  hard    — gpt-5 отвечает на сложные вопросы;
  judge   — gpt-5 вслепую оценивает качество любого ответа по шкале 1–5.

Все ключи — только из .env (корень репозитория). Тарифы — из pricing.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env")   # zerocoder/.env — общий для всех заданий
load_dotenv(HERE / ".env")          # локальная копия рядом со скриптом, если есть

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:12b")

PRICING_FILE = HERE / "pricing.json"
PRICING = json.loads(PRICING_FILE.read_text(encoding="utf-8")) if PRICING_FILE.exists() else {"models": {}}

TIERS = ("simple", "medium", "hard")
FALLBACK_TIER = "hard"   # если дежурный ответил непонятно — отправляем в самую сильную модель

CLASSIFIER_PROMPT = """You are the intake assistant of a regulatory helpdesk for medical devices \
(EU MDR/IVDR, ISO 13485, CE certificates, EUDAMED).
Classify the user's question into exactly one of three tiers:
- simple: a definition or abbreviation, one field of a certificate record (status, date), a single named \
authority, one fixed number or deadline. Answerable in 1-3 sentences.
- medium: a specific requirement or procedure from one regulation or article. Needs regulatory knowledge, \
one source, a structured answer.
- hard: reasoning across several provisions: device classification by Annex VIII rules, transition \
provisions with conditions and evidence, comparison of two regulations, gap analysis of documents, \
audit preparation.
Answer with exactly one lower-case word: simple, medium or hard."""

ANSWER_PROMPT = """You are a regulatory assistant for medical devices (EU MDR 2017/745, IVDR 2017/746, \
ISO 13485/14971, CE certificates, EUDAMED). Answer the question accurately and concisely, citing the \
relevant articles, annexes or rules where applicable. No preamble."""

JUDGE_PROMPT = """You are a strict examiner in EU medical-device regulation. You receive a question and an \
answer written by an assistant. Grade the answer from 1 to 5:
5 = correct and complete, cites the right provisions; 4 = correct with minor omissions; \
3 = partially correct or vague; 2 = mostly wrong or misleading; 1 = wrong, empty or off-topic.
Reply with JSON only: {"score": <1-5>, "reason": "<one short sentence>"}"""

ROLES: dict[str, dict] = {
    "intent": {"provider": "ollama", "model": OLLAMA_MODEL, "system": CLASSIFIER_PROMPT,
               "max_tokens": 5, "reasoning_effort": None},
    "simple": {"provider": "ollama", "model": OLLAMA_MODEL, "system": ANSWER_PROMPT,
               "max_tokens": 400, "reasoning_effort": None},
    "medium": {"provider": "openai", "model": "gpt-5-mini", "system": ANSWER_PROMPT,
               "max_tokens": None, "reasoning_effort": None},
    "hard":   {"provider": "openai", "model": "gpt-5", "system": ANSWER_PROMPT,
               "max_tokens": None, "reasoning_effort": None},
    # лимит 4000: он включает скрытые токены рассуждений; при лимите 800 gpt-5 дважды
    # потратил весь лимит на рассуждения и вернул пустой ответ без оценки
    "judge":  {"provider": "openai", "model": "gpt-5", "system": JUDGE_PROMPT,
               "max_tokens": 4000, "reasoning_effort": "low"},
}


@dataclass
class CallResult:
    """Результат одного обращения к модели — то, что в уроке возвращала call_model."""
    role: str
    provider: str
    model: str
    answer: str
    tokens_in: int
    tokens_out: int
    reasoning_tokens: int
    cached_tokens: int
    cost_usd: float
    latency_s: float
    cached_call: bool = False   # взят из локального кэша повторных запросов (см. Cache)

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------- тарифы

def price_for(model: str) -> dict:
    """Тариф модели в USD за 1 млн токенов. Локальные модели — нули."""
    models = PRICING.get("models", {})
    if model in models:
        return models[model]
    for key, val in models.items():   # 'gemma4:12b' → запись 'gemma4' и т.п.
        if model.startswith(key) or key.startswith(model):
            return val
    raise KeyError(f"Нет тарифа для модели {model!r} в pricing.json")


def compute_cost(model: str, tokens_in: int, tokens_out: int, cached_tokens: int = 0) -> float:
    p = price_for(model)
    fresh_in = max(tokens_in - cached_tokens, 0)
    usd = (fresh_in * p["input"] + cached_tokens * p.get("cached_input", p["input"])
           + tokens_out * p["output"]) / 1_000_000
    return round(usd, 8)


# ----------------------------------------------------------------------------- бэкенды

class Backends:
    """Тонкая обёртка над двумя API: OpenAI (облако) и Ollama (локально).

    В тестах подменяется фальшивыми методами — поэтому вся сетевая часть собрана здесь.
    """

    def __init__(self) -> None:
        self._openai = None

    # -- OpenAI --
    def chat_openai(self, model: str, system: str, user: str,
                    max_tokens: Optional[int], reasoning_effort: Optional[str]) -> tuple[str, dict]:
        if self._openai is None:
            from openai import OpenAI   # импорт здесь, чтобы тесты без ключа не падали
            self._openai = OpenAI()
        kwargs: dict = {}
        if max_tokens:
            kwargs["max_completion_tokens"] = max_tokens
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        resp = self._openai.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )
        u = resp.usage
        details = getattr(u, "completion_tokens_details", None)
        pdetails = getattr(u, "prompt_tokens_details", None)
        usage = {
            "tokens_in": u.prompt_tokens,
            "tokens_out": u.completion_tokens,
            "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0,
            "cached_tokens": getattr(pdetails, "cached_tokens", 0) or 0,
        }
        return (resp.choices[0].message.content or ""), usage

    # -- Ollama --
    def chat_ollama(self, model: str, system: str, user: str,
                    max_tokens: Optional[int], reasoning_effort: Optional[str]) -> tuple[str, dict]:
        body = {
            "model": model,
            "stream": False,
            "think": False,                       # у Gemma 4 есть режим «размышлений» — на CPU он слишком долгий
            "options": {"temperature": 0, "repeat_penalty": 1.1},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if max_tokens:
            body["options"]["num_predict"] = max_tokens
        data = _http_json(f"{OLLAMA_HOST}/api/chat", body, timeout=900)
        usage = {
            "tokens_in": int(data.get("prompt_eval_count", 0)),
            "tokens_out": int(data.get("eval_count", 0)),
            "reasoning_tokens": 0,
            "cached_tokens": 0,
        }
        return data["message"]["content"], usage

    def count_tokens_ollama(self, model: str, text: str) -> Optional[int]:
        """Сколько токенов в тексте по токенизатору локальной модели (для шага «токенизация»)."""
        try:
            data = _http_json(f"{OLLAMA_HOST}/api/tokenize", {"model": model, "prompt": text}, timeout=120)
            return len(data.get("tokens", []))
        except urllib.error.HTTPError:
            pass  # старые версии Ollama без /api/tokenize — считаем через generate raw
        data = _http_json(f"{OLLAMA_HOST}/api/generate",
                          {"model": model, "prompt": text, "raw": True, "stream": False,
                           "options": {"num_predict": 1}}, timeout=300)
        n = data.get("prompt_eval_count")
        return int(n) if n is not None else None


def _http_json(url: str, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ----------------------------------------------------------------------------- кэш

class Cache:
    """Кэш повторных запросов — та самая стратегия «completion cache» из урока (FrugalGPT).

    Одинаковый запрос к одной и той же модели второй раз не отправляется, а берётся с диска.
    Используем его, чтобы прерванный прогон можно было продолжить без повторной оплаты.
    """

    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        self._data: dict[str, dict] = {}
        if path and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._data[rec["key"]] = rec["result"]

    @staticmethod
    def key(namespace: str, role: str, model: str, system: str, user: str,
            max_tokens: Optional[int], effort: Optional[str]) -> str:
        raw = json.dumps([namespace, role, model, system, user, max_tokens, effort], ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[CallResult]:
        rec = self._data.get(key)
        if rec is None:
            return None
        res = CallResult(**rec)
        res.cached_call = True
        return res

    def put(self, key: str, result: CallResult) -> None:
        self._data[key] = result.to_dict()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "result": result.to_dict()}, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------- агент

class Agent:
    def __init__(self, backends: Optional[Backends] = None, cache: Optional[Cache] = None,
                 roles: Optional[dict] = None) -> None:
        self.backends = backends or Backends()
        self.cache = cache or Cache(None)
        self.roles = roles or ROLES

    # аналог call_model(prompt, model_type) из урока
    def call_model(self, prompt: str, role: str, namespace: str = "") -> CallResult:
        cfg = self.roles[role]
        key = Cache.key(namespace, role, cfg["model"], cfg["system"], prompt,
                        cfg["max_tokens"], cfg["reasoning_effort"])
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        fn = self.backends.chat_openai if cfg["provider"] == "openai" else self.backends.chat_ollama
        t0 = time.perf_counter()
        text, usage = fn(cfg["model"], cfg["system"], prompt, cfg["max_tokens"], cfg["reasoning_effort"])
        latency = time.perf_counter() - t0
        res = CallResult(
            role=role, provider=cfg["provider"], model=cfg["model"], answer=text.strip(),
            tokens_in=usage["tokens_in"], tokens_out=usage["tokens_out"],
            reasoning_tokens=usage.get("reasoning_tokens", 0), cached_tokens=usage.get("cached_tokens", 0),
            cost_usd=compute_cost(cfg["model"], usage["tokens_in"], usage["tokens_out"],
                                  usage.get("cached_tokens", 0)),
            latency_s=round(latency, 3),
        )
        self.cache.put(key, res)
        return res

    def classify(self, question: str, namespace: str = "") -> tuple[str, CallResult]:
        res = self.call_model(question, "intent", namespace)
        return parse_tier(res.answer), res

    def route(self, question: str, namespace: str = "") -> dict:
        """Мультимодельный режим: дежурный выбирает уровень, затем отвечает модель этого уровня."""
        tier, intent = self.classify(question, namespace)
        answer = self.call_model(question, tier, namespace)
        return {
            "tier": tier,
            "intent": intent,
            "answer": answer,
            "cost_usd": round(intent.cost_usd + answer.cost_usd, 8),
            "latency_s": round(intent.latency_s + answer.latency_s, 3),
        }

    def judge(self, question: str, answer: str, namespace: str = "") -> tuple[Optional[int], str, CallResult]:
        prompt = f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
        res = self.call_model(prompt, "judge", namespace)
        score, reason = parse_judge(res.answer)
        return score, reason, res


# ----------------------------------------------------------------------------- разбор ответов

_TIER_RE = re.compile(r"\b(simple|medium|hard)\b", re.IGNORECASE)


def parse_tier(text: str) -> str:
    """Достаём класс из ответа дежурного. Непонятный ответ → FALLBACK_TIER (самая сильная модель)."""
    m = _TIER_RE.search(text or "")
    return m.group(1).lower() if m else FALLBACK_TIER


def parse_judge(text: str) -> tuple[Optional[int], str]:
    text = text or ""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            score = int(obj.get("score"))
            if 1 <= score <= 5:
                return score, str(obj.get("reason", "")).strip()
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    m = re.search(r"\b([1-5])\b", text)
    return (int(m.group(1)) if m else None), text.strip()[:200]
