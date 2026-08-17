"""bot_images.py — image generation for taris, provider-dispatched like bot_llm.py.

One entry point, ``generate_image(prompt, chat_id)``, and one rule: an image is a FILE
on this host, never a remote URL handed to the browser. A DALL·E URL expires in about an
hour, so a post saved on Monday would show a broken frame on Tuesday; every provider
therefore has its bytes written under ``CONTENT_MEDIA_DIR`` and only the relative path
is stored in the database.

Providers
    openai   POST {OPENAI_BASE_URL}/images/generations
             gpt-image-1 always answers with b64_json; dall-e-* answers with a url
             unless response_format=b64_json is asked for. Both shapes are handled.
    off      the feature is disabled — is_configured() is False and every UI control
             that would produce an image hides itself.

The API key is resolved through ``llm_providers.openai_p.get_openai_api_key()``, the same
credential store the LLM uses, so an admin who rotates the key in the panel rotates it
here too instead of leaving a second copy behind.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import re
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from core.bot_config import (
    CONTENT_IMAGE_MODEL,
    CONTENT_IMAGE_PROVIDER,
    CONTENT_IMAGE_QUALITY,
    CONTENT_IMAGE_SIZE,
    CONTENT_IMAGE_TIMEOUT,
    CONTENT_MEDIA_DIR,
    OPENAI_BASE_URL,
)

log = logging.getLogger("taris.images")

# A generated file is a PNG; the extension is fixed so path handling never has to guess.
_EXT = ".png"
_MAX_BYTES = 12 * 1024 * 1024
_SAFE_REL = re.compile(r"^-?\d+/[0-9a-f]{8,32}\.png$")


# ─── Configuration ───────────────────────────────────────────────────────────

def provider() -> str:
    return (CONTENT_IMAGE_PROVIDER or "off").strip().lower()


def is_configured() -> bool:
    """True when an image can actually be produced — provider on AND a key present."""
    if provider() in ("", "off", "none", "0"):
        return False
    if provider() == "openai":
        return bool(_openai_key())
    return False


def _openai_key() -> str:
    """Active OpenAI key from the shared credential store (admin-set file → env)."""
    try:
        from core.llm_providers.openai_p import get_openai_api_key
        return get_openai_api_key() or ""
    except Exception as exc:                              # pragma: no cover - defensive
        log.debug("[Images] key lookup failed: %s", exc)
        return ""


def status() -> dict:
    """What the admin panel / agent settings screen prints about image generation."""
    return {
        "provider":  provider(),
        "model":     CONTENT_IMAGE_MODEL,
        "size":      CONTENT_IMAGE_SIZE,
        "configured": is_configured(),
        "media_dir": CONTENT_MEDIA_DIR,
    }


# ─── Storage ─────────────────────────────────────────────────────────────────

def media_root() -> Path:
    return Path(CONTENT_MEDIA_DIR)


def image_path(rel: str) -> Path | None:
    """Absolute path of a stored image, or None when `rel` is not a well-formed
    `<chat_id>/<hex>.png` — the guard that keeps a crafted value from escaping the
    media directory into the filesystem."""
    rel = (rel or "").replace("\\", "/").strip()
    if not _SAFE_REL.match(rel):
        return None
    path = (media_root() / rel).resolve()
    try:
        path.relative_to(media_root().resolve())
    except ValueError:
        return None
    return path


def image_exists(rel: str) -> bool:
    path = image_path(rel)
    return bool(path and path.is_file())


def delete_image(rel: str) -> bool:
    path = image_path(rel)
    if not path or not path.is_file():
        return False
    try:
        path.unlink()
        return True
    except OSError as exc:
        log.warning("[Images] delete failed: %s", exc)
        return False


def _store(chat_id: int, data: bytes) -> str:
    """Write bytes under CONTENT_MEDIA_DIR/<chat_id>/ and return the relative path."""
    folder = media_root() / str(int(chat_id))
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex[:16]}{_EXT}"
    (folder / name).write_bytes(data)
    return f"{int(chat_id)}/{name}"


# ─── Generation ──────────────────────────────────────────────────────────────

def generate_image(prompt: str, chat_id: int, *, model: str = "", size: str = "",
                   timeout: int = 0) -> dict:
    """Generate one image and store it locally.

    Returns ``{"ok": bool, "path": relative, "provider": str, "model": str,
    "error": str}``. Never raises — the caller is a chat handler or a web route, and a
    provider outage must degrade to a message, not a traceback.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return _err("empty prompt")
    prov = provider()
    if prov in ("", "off", "none", "0"):
        return _err("image generation disabled")
    if prov != "openai":
        return _err(f"unknown image provider '{prov}'")

    key = _openai_key()
    if not key:
        return _err("OPENAI_API_KEY not set")

    model = model or CONTENT_IMAGE_MODEL
    size = size or CONTENT_IMAGE_SIZE
    timeout = timeout or CONTENT_IMAGE_TIMEOUT

    body: dict = {
        "model": model,
        # The models cap the prompt (1000 chars for dall-e-3); trimming here turns a
        # hard 400 into a slightly shorter brief.
        "prompt": prompt[:900],
        "size": size,
        "n": 1,
    }
    if model.startswith("dall-e"):
        # dall-e-* default to a signed URL that expires; ask for the bytes directly.
        body["response_format"] = "b64_json"
        body["quality"] = CONTENT_IMAGE_QUALITY
    url = f"{OPENAI_BASE_URL.rstrip('/')}/images/generations"

    try:
        result = _post_json(url, {"Content-Type": "application/json",
                                  "Authorization": f"Bearer {key}"}, body, timeout)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read(400).decode("utf-8", errors="replace")
        except Exception:
            pass
        log.error("[Images] HTTP %s model=%r: %s", exc.code, model, detail[:300])
        return _err(f"HTTP {exc.code}: {_api_message(detail) or exc.reason}")
    except Exception as exc:
        log.error("[Images] request failed: %s: %s", type(exc).__name__, exc)
        return _err(f"{type(exc).__name__}: {exc}")

    data = (result.get("data") or [{}])[0]
    raw = b""
    if data.get("b64_json"):
        try:
            raw = base64.b64decode(data["b64_json"], validate=False)
        except (binascii.Error, ValueError) as exc:
            return _err(f"bad base64 payload: {exc}")
    elif data.get("url"):
        try:
            raw = _download(data["url"], timeout)
        except Exception as exc:
            return _err(f"download failed: {exc}")
    if not raw:
        return _err("provider returned no image")
    if len(raw) > _MAX_BYTES:
        return _err("image too large")

    try:
        rel = _store(chat_id, raw)
    except OSError as exc:
        log.error("[Images] store failed: %s", exc)
        return _err(f"cannot store image: {exc}")

    log.info("[Images] generated %s (%d KB) model=%s", rel, len(raw) // 1024, model)
    return {"ok": True, "path": rel, "provider": "openai", "model": model,
            "bytes": len(raw), "error": ""}


def _err(message: str) -> dict:
    return {"ok": False, "path": "", "provider": provider(),
            "model": CONTENT_IMAGE_MODEL, "bytes": 0, "error": message}


def _api_message(detail: str) -> str:
    """The provider's own message out of an error body, when it is JSON."""
    try:
        return str(json.loads(detail).get("error", {}).get("message", ""))[:200]
    except Exception:
        return ""


def _post_json(url: str, headers: dict, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download(url: str, timeout: int) -> bytes:
    if not url.lower().startswith("https://"):
        raise ValueError("refusing non-HTTPS image URL")
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read(_MAX_BYTES + 1)


# ─── Prompt authoring ────────────────────────────────────────────────────────

_VISUAL_BRIEF = {
    "ru": "Опиши одним абзацем на английском языке фотореалистичную иллюстрацию для этого поста.",
    "de": "Beschreibe in einem Absatz auf Englisch eine fotorealistische Illustration für diesen Beitrag.",
    "sl": "V enem odstavku v angleščini opiši fotorealistično ilustracijo za to objavo.",
    "en": "Describe, in one paragraph in English, a photorealistic illustration for this post.",
}


def describe_image_for_post(post_text: str, topic: str = "", lang: str = "en",
                            timeout: int = 60) -> str:
    """Turn a finished post into a visual brief for the image model.

    Mirrors the reference implementation's two-step flow (text → image description →
    image): asking the image model to illustrate a whole social post directly produces
    a picture full of unreadable text, while a described *scene* produces an image.
    The brief is requested in English whatever the post's language, because that is what
    the image models are trained on.

    The subject is then **appended to the brief**, not left to it. A small local model
    (TS1 answers from a 2B Ollama model) will happily return a beautiful scene that has
    nothing to do with the post — a Rome itinerary came back as "a solitary oak tree on a
    misty hill" — and an illustration unrelated to the text is worse than a plain one.
    The anchor also survives the language gap: the brief is English, the post usually is
    not, so no word-overlap check could catch the drift.
    """
    instruction = _VISUAL_BRIEF.get(lang, _VISUAL_BRIEF["en"])
    prompt = (
        f"Post:\n{(post_text or '')[:1200]}\n\n"
        f"Topic: {topic[:300]}\n\n"
        f"{instruction}\n"
        "The scene MUST show the subject of the post above — not an unrelated landscape. "
        "No text, no letters, no logos and no watermarks in the image. "
        "Name the subject, the setting, the light, the mood and the camera angle. "
        "Answer with the description only — no preamble, no quotes."
    )
    try:
        from core.bot_llm import ask_llm
        text = (ask_llm(prompt, timeout=timeout, use_case="content") or "").strip()
    except Exception as exc:
        log.warning("[Images] visual brief failed: %s", exc)
        text = ""
    subject = (topic or _first_line(post_text))[:200]
    if not text:
        # Never block image generation on the describer: the subject alone is a usable brief.
        return (f"A photorealistic social-media illustration about {subject}. "
                "No text, no letters, no logos.")
    brief = _strip_thinking(text)[:700]
    return f"{brief} Subject: {subject}. No text or letters in the image."


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        clean = line.strip().lstrip("#*_ ").strip()
        if clean:
            return clean
    return "a social media post"


def _strip_thinking(text: str) -> str:
    """Drop <think>…</think> blocks — local reasoning models emit them and they would
    be sent to the image API verbatim."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def ensure_media_dir() -> None:
    try:
        media_root().mkdir(parents=True, exist_ok=True)
    except OSError as exc:                                # pragma: no cover - defensive
        log.warning("[Images] cannot create media dir %s: %s", CONTENT_MEDIA_DIR, exc)


__all__ = [
    "generate_image", "describe_image_for_post", "is_configured", "provider",
    "status", "image_path", "image_exists", "delete_image", "media_root",
    "ensure_media_dir",
]
