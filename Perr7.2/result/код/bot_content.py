"""
bot_content.py — Content Strategy Agent v2: AI-assisted content plan & post generation.

Plan mode flow:
  1. Q1: niche + audience + goal
  2. Q2: platform (Telegram / Instagram / Facebook / VK / Website)
  3. Q3: use Taris Knowledge Base?
  4. N8N generates 7-post content plan (mode=plan)
  5. Plan preview: [Correct | ✅ Accept & Save | Download | New]
  6. On Accept: plan saved as note (slug cp_YYYYMMDDHHMMSS_HEX)
     → if 2 plans already exist: cleanup menu first (summarise old plan → long-term memory → delete)
  7. Stored plan displayed + numbered buttons [Post #1 … #7]
  8. User picks post #N → N8N generates full post (mode=post, post_index=N, plan_content)
  9. Post preview: [Correct | Save draft | Download | Publish | ◀ Back to plan | New]
     → Save: stored as note (slug post_YYYYMMDDHHMMSS_HEX)
     → if 10 posts already exist: cleanup menu first

Quick-post mode (standalone, no plan):
  Same Q1→Q2→Q3 flow but mode=post, no plan context.
  Post preview: [Correct | 🖼 Illustrate | Save draft | Download | Publish | New]

Illustration (v2026.8.82):
  A finished post can be illustrated: the LLM writes a visual brief, core/bot_images.py
  generates the picture and stores it under CONTENT_MEDIA_DIR. The image travels with
  the post — preview sends a photo, publishing sends a photo with a caption.

Publication statistics (v2026.8.82):
  Saved and published posts are tracked in core/store_content.py (content_posts) with a
  metric series per post (content_metrics). The 📊 screen prints posts, likes, views,
  followers, average engagement and the most frequent topics. Numbers come from the user
  (Telegram exposes none for channel posts) or, when the user publishes to a VK
  community, from the VK API itself via features/bot_content_vk.py.

Storage limits (per user):
  MAX_CONTENT_PLANS  = 2    (notes with slug prefix cp_)
  MAX_CONTENT_POSTS  = 10   (notes with slug prefix post_)
  MAX_TRACKED_POSTS  = 200  (rows in content_posts — the statistics ledger)
  Before any deletion: LLM summarises content → store.save_summary(tier='long')
"""

import io
import logging
import threading
import uuid
from datetime import datetime
from typing import Any, Callable

from features.bot_n8n import call_webhook
from core.bot_config import (
    N8N_CONTENT_GENERATE_WH,
    N8N_CONTENT_PUBLISH_WH,
    CONTENT_TG_CHANNEL_ID,
    N8N_CONTENT_TIMEOUT,
)
from core.bot_llm import ask_llm_or_raise

log = logging.getLogger("taris.content")

MAX_CONTENT_PLANS = 2
MAX_CONTENT_POSTS = 10
# The statistics ledger is not the draft folder: pruning it to ten would throw away the
# history the dashboard is drawn from. It is still bounded — an unbounded per-user table
# on a shared box is a slow leak.
MAX_TRACKED_POSTS = 200
_PLAN_PREFIX = "cp_"
_POST_PREFIX = "post_"

# ─────────────────────────────────────────────────────────────────────────────
# Session state — keyed by chat_id
# Steps: idle → q1 → q2 → q3_kb
#        → generating_plan → plan_preview → correcting_plan
#        → plan_saved (shows plan + post select buttons)
#        → generating_post → post_preview → correcting_post
#        → ask_channel → confirming_publish
#        → cleanup_plans | cleanup_posts (limit enforcement)
#        → config_pub_token → config_pub_channel  (per-user publish settings)
# ─────────────────────────────────────────────────────────────────────────────
_sessions: dict[int, dict] = {}


def is_active(chat_id: int) -> bool:
    return chat_id in _sessions


def get_step(chat_id: int) -> str:
    return _sessions.get(chat_id, {}).get("step", "idle")


def cancel(chat_id: int) -> None:
    _sessions.pop(chat_id, None)


def is_configured() -> bool:
    return True  # always available — N8N is optional; LLM fallback always present


def _generate_content_with_llm(sess: dict, mode: str, post_index: int = 0,
                               correction: str = "") -> str:
    """Generate content plan or post using the local LLM (fallback when N8N absent)."""
    q1   = sess.get("q1", "")
    q2   = sess.get("q2", "Telegram")
    lang = sess.get("lang", "ru")
    kb   = sess.get("_kb_context", "")
    plan = sess.get("plan_content", "")

    lang_instruction = {
        "ru": "Отвечай на русском языке.",
        "de": "Antworte auf Deutsch.",
        "en": "Reply in English.",
    }.get(lang, "Reply in the same language as the request.")

    if mode == "plan":
        kb_block = f"\n\nКонтекст базы знаний:\n{kb}" if kb else ""
        correction_block = f"\n\nПожелания по исправлению: {correction}" if correction else ""
        prompt = (
            f"{lang_instruction}\n\n"
            f"Ты опытный контент-стратег. Создай контент-план из 7 постов для платформы {q2}.\n\n"
            f"Тема / аудитория / цель:\n{q1}"
            f"{kb_block}"
            f"{correction_block}\n\n"
            "Для каждого поста укажи: номер, заголовок, краткое описание, хэштеги, \n"
            "формат (текст/видео/карточка/опрос) и рекомендуемое время публикации.\n"
            "Оформи как нумерованный список."
        )
    else:  # mode == "post"
        plan_block = f"\n\nКонтент-план:\n{plan[:1500]}" if plan else ""
        kb_block = f"\n\nКонтекст базы знаний:\n{kb}" if kb else ""
        correction_block = f"\n\nПожелания по исправлению: {correction}" if correction else ""
        post_ref = f" #{post_index}" if post_index else ""
        prompt = (
            f"{lang_instruction}\n\n"
            f"Ты опытный копирайтер. Напиши полный пост{post_ref} для платформы {q2}.\n\n"
            f"Тема / аудитория / цель:\n{q1}"
            f"{plan_block}"
            f"{kb_block}"
            f"{correction_block}\n\n"
            "Пост должен быть готов к публикации: заголовок, основной текст, призыв к действию, хэштеги."
        )

    return ask_llm_or_raise(prompt, timeout=120, use_case="chat")


def _session(chat_id: int) -> dict:
    return _sessions.setdefault(chat_id, {})


# ─────────────────────────────────────────────────────────────────────────────
# Per-user publish configuration (stored in user_prefs DB)
# Keys: content_pub_token, content_pub_channel
# ─────────────────────────────────────────────────────────────────────────────

_PUB_TOKEN_KEY    = "content_pub_token"
_PUB_CHANNEL_KEY  = "content_pub_channel"
_PUB_VK_TOKEN_KEY = "content_pub_vk_token"
_PUB_VK_GROUP_KEY = "content_pub_vk_group"


def _get_pub_config(chat_id: int) -> dict:
    """Return the per-user publish settings from user_prefs. Empty strings when not set.

    Two independent targets: a Telegram channel (bot token + channel) and a VK community
    (access token + group id). Either may be configured alone.
    """
    try:
        from core.bot_db import db_get_user_pref
        return {
            "token":    db_get_user_pref(chat_id, _PUB_TOKEN_KEY,    ""),
            "channel":  db_get_user_pref(chat_id, _PUB_CHANNEL_KEY,  ""),
            "vk_token": db_get_user_pref(chat_id, _PUB_VK_TOKEN_KEY, ""),
            "vk_group": db_get_user_pref(chat_id, _PUB_VK_GROUP_KEY, ""),
        }
    except Exception as exc:
        log.warning("[content] _get_pub_config failed: %s", exc)
        return {"token": "", "channel": "", "vk_token": "", "vk_group": ""}


def is_vk_configured(chat_id: int) -> bool:
    """True when this user can publish to a VK community (and read its numbers back)."""
    try:
        from features.bot_content_vk import is_configured as _vk_ok
        cfg = _get_pub_config(chat_id)
        return bool(_vk_ok(cfg.get("vk_token", ""), cfg.get("vk_group", "")))
    except Exception:
        return False


def _set_pub_config_value(chat_id: int, key: str, value: str) -> None:
    from core.bot_db import db_set_user_pref
    db_set_user_pref(chat_id, key, value.strip())


def is_publish_configured(chat_id: int) -> bool:
    """True when at least one publish target is fully configured."""
    cfg = _get_pub_config(chat_id)
    telegram_ok = bool(cfg["token"]) and bool(cfg["channel"])
    return telegram_ok or is_vk_configured(chat_id)


# ─────────────────────────────────────────────────────────────────────────────
# Storage helpers
# ─────────────────────────────────────────────────────────────────────────────

def _list_content_plans(chat_id: int) -> list[dict]:
    """Return saved content plans (notes with cp_ prefix), newest first."""
    try:
        from telegram.bot_users import _list_notes_for
        return [n for n in _list_notes_for(chat_id) if n["slug"].startswith(_PLAN_PREFIX)]
    except Exception as exc:
        log.warning("[content] list plans failed: %s", exc)
        return []


def _list_content_posts(chat_id: int) -> list[dict]:
    """Return saved posts (notes with post_ prefix), newest first."""
    try:
        from telegram.bot_users import _list_notes_for
        return [n for n in _list_notes_for(chat_id) if n["slug"].startswith(_POST_PREFIX)]
    except Exception as exc:
        log.warning("[content] list posts failed: %s", exc)
        return []


def _save_content_note(chat_id: int, slug: str, content: str) -> None:
    """Save content to taris notes DB."""
    from telegram.bot_users import _save_note_file
    _save_note_file(chat_id, slug, content)


def _load_content_note(chat_id: int, slug: str) -> str:
    """Load note content. Returns empty string if not found."""
    try:
        from telegram.bot_users import _load_note_text
        return _load_note_text(chat_id, slug) or ""
    except Exception:
        return ""


def _delete_content_note(chat_id: int, slug: str) -> None:
    """Delete a content note from DB and file."""
    try:
        from telegram.bot_users import _delete_note_file
        _delete_note_file(chat_id, slug)
    except Exception as exc:
        log.warning("[content] delete note failed: %s", exc)


def _summarize_to_long_term_memory(chat_id: int, content: str, label: str) -> None:
    """Summarise content via LLM and store as long-term memory."""
    try:
        from core.bot_llm import ask_llm
        from core.store import store
        prompt = (
            f"Summarise the following content plan or post in 3–5 sentences for "
            f"long-term memory. Capture the main topic, target audience, goals, and "
            f"key ideas. Label: {label}\n\n{content[:3000]}"
        )
        summary = ask_llm(prompt, timeout=30)
        if summary:
            store.save_summary(chat_id, f"[ContentAgent] {label}: {summary}", tier="long")
            log.info("[content] long-term memory saved for chat_id=%d label=%r", chat_id, label)
    except Exception as exc:
        log.warning("[content] summarise to memory failed: %s", exc)


def _new_slug(prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{prefix}{ts}_{uuid.uuid4().hex[:6]}"


# ─────────────────────────────────────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────────────────────────────────────

def show_menu(chat_id: int, bot: Any, t: Callable) -> None:
    """Show Content Strategist mode selection menu."""
    from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
    cfg = _get_pub_config(chat_id)
    pub_ok = bool(cfg["token"]) and bool(cfg["channel"])
    if pub_ok:
        cfg_label = t(chat_id, "content_btn_pub_settings_ok").format(channel=cfg["channel"])
    else:
        cfg_label = t(chat_id, "content_btn_pub_settings_unset")
    kb = InlineKeyboardMarkup(row_width=1)
    kb.add(
        InlineKeyboardButton(t(chat_id, "content_btn_plan"), callback_data="content_mode:plan"),
        InlineKeyboardButton(t(chat_id, "content_btn_post"), callback_data="content_mode:post"),
        InlineKeyboardButton(t(chat_id, "content_btn_stats"), callback_data="content_stats"),
        InlineKeyboardButton(cfg_label,                      callback_data="content_pub_config"),
        InlineKeyboardButton(t(chat_id, "content_btn_back"), callback_data="agents_menu"),
    )
    bot.send_message(chat_id, t(chat_id, "content_menu_title"),
                     parse_mode="Markdown", reply_markup=kb)


def start_mode(chat_id: int, mode: str, bot: Any, t: Callable) -> None:
    """Begin a content generation session for the given mode (plan|post)."""
    sess = _session(chat_id)
    sess.clear()
    sess.update({"step": "q1", "mode": mode, "session_id": str(uuid.uuid4())})
    key = "content_q1_plan" if mode == "plan" else "content_q1_post"
    bot.send_message(chat_id, t(chat_id, key), parse_mode="Markdown")


# ─────────────────────────────────────────────────────────────────────────────
# Question flow — shared by both modes
# ─────────────────────────────────────────────────────────────────────────────

def _ask_platform(chat_id: int, bot: Any, t: Callable) -> None:
    from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
    kb = InlineKeyboardMarkup(row_width=2)
    for key, val in [("content_q2_tg", "Telegram"), ("content_q2_ig", "Instagram"),
                     ("content_q2_fb", "Facebook"), ("content_q2_vk", "VK"),
                     ("content_q2_web", "Website")]:
        kb.add(InlineKeyboardButton(t(chat_id, key), callback_data=f"content_platform:{val}"))
    bot.send_message(chat_id, t(chat_id, "content_q2"), parse_mode="Markdown", reply_markup=kb)


def _ask_kb(chat_id: int, bot: Any, t: Callable) -> None:
    from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
    kb = InlineKeyboardMarkup(row_width=2)
    kb.add(
        InlineKeyboardButton(t(chat_id, "content_q3_kb_yes"), callback_data="content_kb:yes"),
        InlineKeyboardButton(t(chat_id, "content_q3_kb_no"),  callback_data="content_kb:no"),
    )
    bot.send_message(chat_id, t(chat_id, "content_q3_kb"), parse_mode="Markdown", reply_markup=kb)


def on_platform_selected(chat_id: int, platform: str, bot: Any, t: Callable) -> None:
    sess = _session(chat_id)
    if sess.get("step") != "q2":
        return
    sess["q2"] = platform
    sess["step"] = "q3_kb"
    _ask_kb(chat_id, bot, t)


def on_kb_selected(chat_id: int, use_kb: bool, bot: Any, t: Callable) -> None:
    sess = _session(chat_id)
    if sess.get("step") != "q3_kb":
        return
    sess["use_kb"] = use_kb
    if sess.get("mode") == "plan":
        _do_generate_plan(chat_id, bot, t)
    else:
        _do_generate_post(chat_id, bot, t, post_index=0)


# ─────────────────────────────────────────────────────────────────────────────
# Plan generation
# ─────────────────────────────────────────────────────────────────────────────

def _do_generate_plan(chat_id: int, bot: Any, t: Callable, correction: str = "") -> None:
    """Generate content plan via N8N (background thread)."""
    sess = _session(chat_id)
    sess["step"] = "generating_plan"
    bot.send_message(chat_id, t(chat_id, "content_generating"), parse_mode="Markdown")

    def _run():
        try:
            if N8N_CONTENT_GENERATE_WH:
                payload: dict[str, Any] = {
                    "chat_id":    chat_id,
                    "mode":       "plan",
                    "q1":         sess.get("q1", ""),
                    "q2":         sess.get("q2", ""),
                    "kb_context": _fetch_kb(chat_id, sess) if sess.get("use_kb") else "",
                    "correction": correction,
                    "lang":       sess.get("lang", "ru"),
                    "session_id": sess.get("session_id", str(uuid.uuid4())),
                }
                result = call_webhook(N8N_CONTENT_GENERATE_WH, payload, timeout=N8N_CONTENT_TIMEOUT)
                if result.get("error"):
                    bot.send_message(chat_id,
                        t(chat_id, "content_generate_error").format(error=result["error"]),
                        parse_mode="Markdown")
                    _sessions.pop(chat_id, None)
                    return
                content = result.get("content") or result.get("result") or ""
                if not content:
                    bot.send_message(chat_id,
                        t(chat_id, "content_generate_error").format(error="Empty response"),
                        parse_mode="Markdown")
                    _sessions.pop(chat_id, None)
                    return
            else:
                # Fallback: generate via local LLM
                if sess.get("use_kb"):
                    sess["_kb_context"] = _fetch_kb(chat_id, sess)
                content = _generate_content_with_llm(sess, mode="plan",
                                                      correction=correction)
                if not content:
                    bot.send_message(chat_id,
                        t(chat_id, "content_generate_error").format(error="Empty response"),
                        parse_mode="Markdown")
                    _sessions.pop(chat_id, None)
                    return
            sess["plan_content"] = content
            sess["step"] = "plan_preview"
            _show_plan_preview(chat_id, bot, t)
        except Exception as exc:
            log.exception("[content] plan generate error: %s", exc)
            bot.send_message(chat_id,
                t(chat_id, "content_generate_error").format(error=str(exc)),
                parse_mode="Markdown")
            _sessions.pop(chat_id, None)

    threading.Thread(target=_run, daemon=True).start()


def _show_plan_preview(chat_id: int, bot: Any, t: Callable) -> None:
    """Show plan with: Correct | Accept & Save | Download | New."""
    from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
    sess = _sessions.get(chat_id, {})
    content = sess.get("plan_content", "")
    display = content[:3800] + ("…" if len(content) > 3800 else "")
    try:
        bot.send_message(chat_id, t(chat_id, "content_preview_plan").format(content=display),
                         parse_mode="Markdown")
    except Exception:
        bot.send_message(chat_id, display)
    kb = InlineKeyboardMarkup(row_width=2)
    kb.add(
        InlineKeyboardButton(t(chat_id, "content_btn_correct"),     callback_data="content_plan_action:correct"),
        InlineKeyboardButton(t(chat_id, "content_btn_accept_plan"), callback_data="content_plan_action:accept"),
    )
    kb.add(
        InlineKeyboardButton(t(chat_id, "content_btn_download"), callback_data="content_plan_action:download"),
        InlineKeyboardButton(t(chat_id, "content_btn_new"),      callback_data="content_plan_action:new"),
    )
    bot.send_message(chat_id, "━━━", reply_markup=kb)


def on_plan_action(chat_id: int, action: str, bot: Any, t: Callable) -> None:
    """Handle plan preview / saved plan button press."""
    sess = _sessions.get(chat_id, {})
    if not sess:
        return
    step = sess.get("step")

    if action == "correct":
        if step != "plan_preview":
            return
        sess["step"] = "correcting_plan"
        bot.send_message(chat_id, t(chat_id, "content_correct_prompt"), parse_mode="Markdown")

    elif action == "accept":
        if step != "plan_preview":
            return
        _accept_and_save_plan(chat_id, bot, t)

    elif action == "download":
        # Download works from both plan_preview and plan_saved
        _do_download(chat_id, bot, t, content_key="plan_content", label="plan")
        if step == "plan_saved":
            _show_plan_with_post_buttons(chat_id, bot, t)
        # plan_preview: buttons still visible, user can act further

    elif action == "new":
        _sessions.pop(chat_id, None)
        show_menu(chat_id, bot, t)


def _accept_and_save_plan(chat_id: int, bot: Any, t: Callable) -> None:
    """Check limit; show cleanup menu if needed, otherwise save."""
    sess = _sessions.get(chat_id, {})
    plans = _list_content_plans(chat_id)
    if len(plans) >= MAX_CONTENT_PLANS:
        sess["step"] = "cleanup_plans"
        sess["pending_save"] = "plan"
        _show_cleanup_menu(chat_id, "plan", plans, bot, t)
        return
    _do_save_plan(chat_id, bot, t)


def _do_save_plan(chat_id: int, bot: Any, t: Callable) -> None:
    """Save plan to notes and transition to plan_saved."""
    sess = _sessions.get(chat_id, {})
    content = sess.get("plan_content", "")
    slug = _new_slug(_PLAN_PREFIX)
    try:
        _save_content_note(chat_id, slug, content)
        sess["plan_slug"] = slug
        sess["step"] = "plan_saved"
        bot.send_message(chat_id, t(chat_id, "content_plan_saved"), parse_mode="Markdown")
        _show_plan_with_post_buttons(chat_id, bot, t)
    except Exception as exc:
        log.warning("[content] save plan error: %s", exc)
        bot.send_message(chat_id, t(chat_id, "content_save_error"), parse_mode="Markdown")
        sess["step"] = "plan_preview"


def _show_plan_with_post_buttons(chat_id: int, bot: Any, t: Callable) -> None:
    """Display stored plan + numbered [Post #1…#7] generation buttons."""
    from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
    sess = _sessions.get(chat_id, {})
    content = sess.get("plan_content", "")
    display = content[:3800] + ("…" if len(content) > 3800 else "")
    try:
        bot.send_message(chat_id, t(chat_id, "content_plan_header").format(content=display),
                         parse_mode="Markdown")
    except Exception:
        bot.send_message(chat_id, display)
    kb = InlineKeyboardMarkup(row_width=2)
    post_btns = [
        InlineKeyboardButton(t(chat_id, "content_btn_gen_post").format(n=n),
                             callback_data=f"content_genpost:{n}")
        for n in range(1, 8)
    ]
    kb.add(*post_btns)
    kb.row(
        InlineKeyboardButton(t(chat_id, "content_btn_download"), callback_data="content_plan_action:download"),
        InlineKeyboardButton(t(chat_id, "content_btn_new"),      callback_data="content_plan_action:new"),
    )
    bot.send_message(chat_id, t(chat_id, "content_select_post_prompt"), reply_markup=kb)


# ─────────────────────────────────────────────────────────────────────────────
# Post generation (from plan or standalone)
# ─────────────────────────────────────────────────────────────────────────────

def on_genpost_selected(chat_id: int, post_index: int, bot: Any, t: Callable) -> None:
    """User selected post #N to expand from the saved plan."""
    sess = _sessions.get(chat_id, {})
    if not sess or sess.get("step") not in ("plan_saved", "post_preview"):
        return
    sess["post_index"] = post_index
    _do_generate_post(chat_id, bot, t, post_index=post_index)


def _do_generate_post(chat_id: int, bot: Any, t: Callable,
                      post_index: int = 0, correction: str = "") -> None:
    """Generate post content via N8N (background thread)."""
    sess = _session(chat_id)
    # A regenerated post is a different post: its old illustration no longer belongs to
    # it, and publishing would send a picture of the previous draft.
    _drop_session_image(sess)
    sess.pop("post_id", None)
    sess["step"] = "generating_post"
    if post_index:
        bot.send_message(chat_id,
            t(chat_id, "content_generating_post").format(n=post_index),
            parse_mode="Markdown")
    else:
        bot.send_message(chat_id, t(chat_id, "content_generating"), parse_mode="Markdown")

    def _run():
        try:
            if N8N_CONTENT_GENERATE_WH:
                payload: dict[str, Any] = {
                    "chat_id":      chat_id,
                    "mode":         "post",
                    "q1":           sess.get("q1", ""),
                    "q2":           sess.get("q2", ""),
                    "kb_context":   _fetch_kb(chat_id, sess) if sess.get("use_kb") else "",
                    "correction":   correction,
                    "lang":         sess.get("lang", "ru"),
                    "session_id":   sess.get("session_id", str(uuid.uuid4())),
                    "post_index":   post_index,
                    "plan_content": sess.get("plan_content", ""),
                }
                result = call_webhook(N8N_CONTENT_GENERATE_WH, payload, timeout=N8N_CONTENT_TIMEOUT)
                if result.get("error"):
                    bot.send_message(chat_id,
                        t(chat_id, "content_generate_error").format(error=result["error"]),
                        parse_mode="Markdown")
                    _recover_after_post_error(chat_id, bot, t, sess)
                    return
                content = result.get("content") or result.get("result") or ""
                if not content:
                    bot.send_message(chat_id,
                        t(chat_id, "content_generate_error").format(error="Empty response"),
                        parse_mode="Markdown")
                    _recover_after_post_error(chat_id, bot, t, sess)
                    return
            else:
                # Fallback: generate via local LLM
                if sess.get("use_kb") and not sess.get("_kb_context"):
                    sess["_kb_context"] = _fetch_kb(chat_id, sess)
                content = _generate_content_with_llm(sess, mode="post",
                                                      post_index=post_index,
                                                      correction=correction)
                if not content:
                    bot.send_message(chat_id,
                        t(chat_id, "content_generate_error").format(error="Empty response"),
                        parse_mode="Markdown")
                    _recover_after_post_error(chat_id, bot, t, sess)
                    return
            sess["post_content"] = content
            sess["step"] = "post_preview"
            _show_post_preview(chat_id, bot, t)
        except Exception as exc:
            log.exception("[content] post generate error: %s", exc)
            bot.send_message(chat_id,
                t(chat_id, "content_generate_error").format(error=str(exc)),
                parse_mode="Markdown")
            _recover_after_post_error(chat_id, bot, t, sess)

    threading.Thread(target=_run, daemon=True).start()


def _recover_after_post_error(chat_id: int, bot: Any, t: Callable, sess: dict) -> None:
    if sess.get("plan_slug"):
        sess["step"] = "plan_saved"
        _show_plan_with_post_buttons(chat_id, bot, t)
    else:
        _sessions.pop(chat_id, None)


def _show_post_preview(chat_id: int, bot: Any, t: Callable) -> None:
    """Show post with action buttons."""
    from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
    sess = _sessions.get(chat_id, {})
    content = sess.get("post_content", "")
    post_index = sess.get("post_index", 0)
    display = content[:3800] + ("…" if len(content) > 3800 else "")
    preview_key = "content_preview_post_from_plan" if post_index else "content_preview_post"
    _send_post_image(chat_id, bot, sess.get("image_path", ""))
    try:
        bot.send_message(chat_id,
            t(chat_id, preview_key).format(content=display, n=post_index),
            parse_mode="Markdown")
    except Exception:
        bot.send_message(chat_id, display)
    kb = InlineKeyboardMarkup(row_width=2)
    kb.add(
        InlineKeyboardButton(t(chat_id, "content_btn_correct"),  callback_data="content_post_action:correct"),
        InlineKeyboardButton(t(chat_id, "content_btn_save"),     callback_data="content_post_action:save"),
    )
    if _images_available():
        if sess.get("image_path"):
            kb.add(
                InlineKeyboardButton(t(chat_id, "content_btn_image_new"), callback_data="content_post_action:image"),
                InlineKeyboardButton(t(chat_id, "content_btn_image_del"), callback_data="content_post_action:image_del"),
            )
        else:
            kb.add(InlineKeyboardButton(t(chat_id, "content_btn_image"),
                                        callback_data="content_post_action:image"))
    kb.add(
        InlineKeyboardButton(t(chat_id, "content_btn_download"), callback_data="content_post_action:download"),
        InlineKeyboardButton(t(chat_id, "content_btn_publish"),  callback_data="content_post_action:publish"),
    )
    if sess.get("plan_slug"):
        kb.add(
            InlineKeyboardButton(t(chat_id, "content_btn_back_to_plan"), callback_data="content_post_action:back_plan"),
            InlineKeyboardButton(t(chat_id, "content_btn_new"),          callback_data="content_post_action:new"),
        )
    else:
        kb.add(InlineKeyboardButton(t(chat_id, "content_btn_new"), callback_data="content_post_action:new"))
    bot.send_message(chat_id, "━━━", reply_markup=kb)


# ─────────────────────────────────────────────────────────────────────────────
# Illustration — visual brief (LLM) → image (core/bot_images.py) → local file
# ─────────────────────────────────────────────────────────────────────────────

def _images_available() -> bool:
    """True when an image can actually be produced. Controls whether the button exists
    at all — offering a control that cannot work is worse than not offering it."""
    try:
        from core import bot_images
        return bot_images.is_configured()
    except Exception:
        return False


def _send_post_image(chat_id: int, bot: Any, rel_path: str) -> bool:
    """Send the post's illustration when there is one, once per version of it.

    The preview is re-rendered on every button press; re-uploading the same picture each
    time would turn one illustration into a column of identical photos in the chat.
    """
    if not rel_path:
        return False
    sess = _sessions.get(chat_id, {})
    if sess.get("_image_sent") == rel_path:
        return False
    try:
        from core import bot_images
        path = bot_images.image_path(rel_path)
        if not path or not path.is_file():
            return False
        with path.open("rb") as fh:
            bot.send_photo(chat_id, fh)
        sess["_image_sent"] = rel_path
        return True
    except Exception as exc:
        log.warning("[content] send image failed: %s", exc)
        return False


def _do_generate_image(chat_id: int, bot: Any, t: Callable) -> None:
    """Generate an illustration for the current post, in the background."""
    sess = _session(chat_id)
    content = sess.get("post_content", "")
    if not content:
        return
    previous = sess.get("image_path", "")
    sess["step"] = "generating_image"
    bot.send_message(chat_id, t(chat_id, "content_image_generating"), parse_mode="Markdown")

    def _run():
        try:
            from core import bot_images
            brief = bot_images.describe_image_for_post(
                content, topic=sess.get("q1", ""), lang=sess.get("lang", "ru"))
            result = bot_images.generate_image(brief, chat_id)
            if not result.get("ok"):
                bot.send_message(chat_id,
                    t(chat_id, "content_image_error").format(error=result.get("error", "?")),
                    parse_mode="Markdown")
            else:
                # Replacing an illustration deletes the old file: an image nothing points
                # at any more is dead weight in the media directory forever.
                if previous and previous != result["path"]:
                    bot_images.delete_image(previous)
                sess["image_path"]   = result["path"]
                sess["image_prompt"] = brief
                bot.send_message(chat_id, t(chat_id, "content_image_done"),
                                 parse_mode="Markdown")
        except Exception as exc:
            log.exception("[content] image generation failed: %s", exc)
            bot.send_message(chat_id,
                t(chat_id, "content_image_error").format(error=str(exc)),
                parse_mode="Markdown")
        finally:
            sess["step"] = "post_preview"
            _show_post_preview(chat_id, bot, t)

    threading.Thread(target=_run, daemon=True).start()


def _drop_session_image(sess: dict) -> None:
    """Forget the session's illustration and delete its file."""
    rel = sess.pop("image_path", "")
    sess.pop("image_prompt", None)
    sess.pop("_image_sent", None)
    if not rel:
        return
    try:
        from core import bot_images
        bot_images.delete_image(rel)
    except Exception as exc:
        log.warning("[content] image delete failed: %s", exc)


def _remove_image(chat_id: int, bot: Any, t: Callable) -> None:
    sess = _session(chat_id)
    _drop_session_image(sess)
    bot.send_message(chat_id, t(chat_id, "content_image_removed"), parse_mode="Markdown")
    _show_post_preview(chat_id, bot, t)


def on_post_action(chat_id: int, action: str, bot: Any, t: Callable) -> None:
    """Handle post preview button press."""
    sess = _sessions.get(chat_id, {})
    if not sess or sess.get("step") != "post_preview":
        return

    if action == "correct":
        sess["step"] = "correcting_post"
        bot.send_message(chat_id, t(chat_id, "content_correct_prompt"), parse_mode="Markdown")

    elif action == "save":
        _accept_and_save_post(chat_id, bot, t)

    elif action == "image":
        if not _images_available():
            bot.send_message(chat_id, t(chat_id, "content_image_unavailable"),
                             parse_mode="Markdown")
            return
        _do_generate_image(chat_id, bot, t)

    elif action == "image_del":
        _remove_image(chat_id, bot, t)

    elif action == "download":
        _do_download(chat_id, bot, t, content_key="post_content", label="post")
        _show_post_preview(chat_id, bot, t)

    elif action == "publish":
        if not is_publish_configured(chat_id):
            _show_publish_not_configured(chat_id, bot, t)
            return
        sess["step"] = "confirming_publish"
        _ask_publish_confirm(chat_id, bot, t)

    elif action == "back_plan":
        sess["step"] = "plan_saved"
        _show_plan_with_post_buttons(chat_id, bot, t)

    elif action == "new":
        _sessions.pop(chat_id, None)
        show_menu(chat_id, bot, t)


def _accept_and_save_post(chat_id: int, bot: Any, t: Callable) -> None:
    """Check limit; show cleanup menu if needed, otherwise save."""
    sess = _sessions.get(chat_id, {})
    posts = _list_content_posts(chat_id)
    if len(posts) >= MAX_CONTENT_POSTS:
        sess["step"] = "cleanup_posts"
        sess["pending_save"] = "post"
        _show_cleanup_menu(chat_id, "post", posts, bot, t)
        return
    _do_save_post(chat_id, bot, t)


def _do_save_post(chat_id: int, bot: Any, t: Callable) -> None:
    """Save post draft to notes, track it for statistics, and return to post_preview."""
    sess = _sessions.get(chat_id, {})
    content = sess.get("post_content", "")
    slug = _new_slug(_POST_PREFIX)
    try:
        _save_content_note(chat_id, slug, content)
        _track_post(chat_id, sess)
        bot.send_message(chat_id,
            t(chat_id, "content_save_done").format(slug=slug),
            parse_mode="Markdown")
        sess["step"] = "post_preview"
        _show_post_preview(chat_id, bot, t)
    except Exception as exc:
        log.warning("[content] save post error: %s", exc)
        bot.send_message(chat_id, t(chat_id, "content_save_error"), parse_mode="Markdown")
        sess["step"] = "post_preview"


# ─────────────────────────────────────────────────────────────────────────────
# Statistics ledger — content_posts / content_metrics
# ─────────────────────────────────────────────────────────────────────────────

def _track_post(chat_id: int, sess: dict) -> str:
    """Record (or update) this session's post in the statistics ledger; return its id.

    Called from BOTH save and publish, and idempotent per session: a user who saves a
    draft and then publishes it must end up with one post in the statistics, not two.
    """
    try:
        from core import store_content as SC
        post_id = sess.get("post_id", "")
        body    = sess.get("post_content", "")
        if post_id and SC.get_post(post_id, chat_id):
            SC.update_post(post_id, body=body,
                           image_path=sess.get("image_path", ""),
                           image_prompt=sess.get("image_prompt", ""))
            return post_id
        post_id = SC.save_post(
            chat_id, body,
            topic=sess.get("q1", ""),
            platform=(sess.get("q2", "") or "").lower(),
            image_path=sess.get("image_path", ""),
            image_prompt=sess.get("image_prompt", ""),
            plan_slug=sess.get("plan_slug", ""),
            post_index=int(sess.get("post_index", 0) or 0),
            lang=sess.get("lang", ""),
        )
        sess["post_id"] = post_id
        _prune_tracked_posts(chat_id)
        return post_id
    except Exception as exc:
        log.warning("[content] track post failed: %s", exc)
        return ""


def _prune_tracked_posts(chat_id: int) -> None:
    """Keep the ledger bounded — drop the oldest rows (and their images) past the cap."""
    try:
        from core import store_content as SC
        posts = SC.list_posts(chat_id, limit=MAX_TRACKED_POSTS + 50)
        for old in posts[MAX_TRACKED_POSTS:]:
            _delete_tracked_post(chat_id, old["id"])
    except Exception as exc:
        log.debug("[content] prune failed: %s", exc)


def _delete_tracked_post(chat_id: int, post_id: str) -> bool:
    """Delete a tracked post, its metric history and its generated image file."""
    try:
        from core import store_content as SC
        post = SC.get_post(post_id, chat_id)
        if not post:
            return False
        if post.get("image_path"):
            try:
                from core import bot_images
                bot_images.delete_image(post["image_path"])
            except Exception as exc:
                log.debug("[content] image cleanup failed: %s", exc)
        return SC.delete_post(chat_id, post_id)
    except Exception as exc:
        log.warning("[content] delete tracked post failed: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup — storage limit enforcement
# ─────────────────────────────────────────────────────────────────────────────

def _show_cleanup_menu(chat_id: int, item_type: str, items: list[dict],
                       bot: Any, t: Callable) -> None:
    """Show list of existing items to delete to make room for a new one."""
    from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
    count = len(items)
    max_val = MAX_CONTENT_PLANS if item_type == "plan" else MAX_CONTENT_POSTS
    limit_key = "content_limit_plans" if item_type == "plan" else "content_limit_posts"
    bot.send_message(chat_id,
        t(chat_id, limit_key).format(count=count, max=max_val),
        parse_mode="Markdown")
    kb = InlineKeyboardMarkup(row_width=1)
    for item in items[:8]:
        title = (item.get("title") or item.get("slug", "?"))[:40]
        slug = item["slug"]
        kb.add(InlineKeyboardButton(
            f"🗑️ {title}",
            callback_data=f"content_del:{item_type}:{slug}",
        ))
    kb.add(InlineKeyboardButton(t(chat_id, "content_btn_new"), callback_data="content_plan_action:new"))
    bot.send_message(chat_id, t(chat_id, "content_cleanup_prompt"), reply_markup=kb)


def on_delete_request(chat_id: int, item_type: str, slug: str, bot: Any, t: Callable) -> None:
    """User tapped a delete button — show confirmation."""
    from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
    sess = _sessions.get(chat_id, {})
    if not sess:
        return
    sess["del_type"] = item_type
    sess["del_slug"] = slug
    # Resolve title from cached notes list
    try:
        from telegram.bot_users import _list_notes_for
        notes = _list_notes_for(chat_id)
        note = next((n for n in notes if n["slug"] == slug), None)
        title = note.get("title", slug) if note else slug
    except Exception:
        title = slug
    sess["del_title"] = title
    kb = InlineKeyboardMarkup(row_width=2)
    kb.add(
        InlineKeyboardButton(t(chat_id, "content_btn_delete_yes"), callback_data="content_del_confirm"),
        InlineKeyboardButton(t(chat_id, "content_btn_delete_no"),  callback_data="content_del_cancel"),
    )
    bot.send_message(chat_id,
        t(chat_id, "content_delete_confirm").format(title=title),
        parse_mode="Markdown",
        reply_markup=kb)


def on_delete_confirmed(chat_id: int, bot: Any, t: Callable) -> None:
    """User confirmed deletion — summarise to long-term memory, delete, continue."""
    sess = _sessions.get(chat_id, {})
    slug  = sess.pop("del_slug", "")
    itype = sess.pop("del_type", "")
    title = sess.pop("del_title", slug)
    if not slug:
        return
    # Load content before deletion for summarisation
    content = _load_content_note(chat_id, slug)
    if content:
        threading.Thread(target=_summarize_to_long_term_memory,
                         args=(chat_id, content, title), daemon=True).start()
    _delete_content_note(chat_id, slug)
    bot.send_message(chat_id, t(chat_id, "content_delete_done"), parse_mode="Markdown")
    # Re-check limit and continue with the pending save
    pending = sess.get("pending_save", "")
    if pending == "plan":
        sess.pop("pending_save", None)
        if len(_list_content_plans(chat_id)) < MAX_CONTENT_PLANS:
            sess["step"] = "plan_preview"
            _do_save_plan(chat_id, bot, t)
        else:
            _show_cleanup_menu(chat_id, "plan", _list_content_plans(chat_id), bot, t)
    elif pending == "post":
        sess.pop("pending_save", None)
        if len(_list_content_posts(chat_id)) < MAX_CONTENT_POSTS:
            sess["step"] = "post_preview"
            _do_save_post(chat_id, bot, t)
        else:
            _show_cleanup_menu(chat_id, "post", _list_content_posts(chat_id), bot, t)


def on_delete_cancelled(chat_id: int, bot: Any, t: Callable) -> None:
    """User cancelled deletion — return to cleanup menu or main menu."""
    sess = _sessions.get(chat_id, {})
    sess.pop("del_slug", None)
    sess.pop("del_type", None)
    sess.pop("del_title", None)
    pending = sess.get("pending_save", "")
    if pending == "plan":
        _show_cleanup_menu(chat_id, "plan", _list_content_plans(chat_id), bot, t)
    elif pending == "post":
        _show_cleanup_menu(chat_id, "post", _list_content_posts(chat_id), bot, t)
    else:
        _sessions.pop(chat_id, None)
        show_menu(chat_id, bot, t)


# ─────────────────────────────────────────────────────────────────────────────
# Publish flow (uses per-user bot token + channel stored in user_prefs)
# ─────────────────────────────────────────────────────────────────────────────

def _show_publish_not_configured(chat_id: int, bot: Any, t: Callable) -> None:
    from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
    kb = InlineKeyboardMarkup(row_width=1)
    back_cb = ("content_post_action:back_plan"
               if _sessions.get(chat_id, {}).get("plan_slug")
               else "content_post_action:new")
    kb.add(
        InlineKeyboardButton(t(chat_id, "content_btn_pub_configure"),  callback_data="content_pub_config"),
        InlineKeyboardButton(t(chat_id, "content_btn_publish_cancel"), callback_data=back_cb),
    )
    bot.send_message(chat_id, t(chat_id, "content_publish_setup_required"),
                     parse_mode="Markdown", reply_markup=kb)


def _ask_publish_confirm(chat_id: int, bot: Any, t: Callable) -> None:
    """Offer one button per configured target — Telegram channel and/or VK community."""
    from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
    cfg = _get_pub_config(chat_id)
    telegram_ok = bool(cfg["token"]) and bool(cfg["channel"])
    vk_ok = is_vk_configured(chat_id)
    targets = []
    kb = InlineKeyboardMarkup(row_width=2)
    if telegram_ok:
        targets.append(cfg["channel"])
        kb.add(InlineKeyboardButton(
            t(chat_id, "content_btn_publish_tg").format(channel=cfg["channel"]),
            callback_data="content_publish:tg"))
    if vk_ok:
        targets.append(f"VK {cfg['vk_group']}")
        kb.add(InlineKeyboardButton(
            t(chat_id, "content_btn_publish_vk").format(group=cfg["vk_group"]),
            callback_data="content_publish:vk"))
    kb.add(InlineKeyboardButton(t(chat_id, "content_btn_publish_cancel"),
                                callback_data="content_publish:cancel"))
    bot.send_message(chat_id,
        t(chat_id, "content_publish_confirm").format(channel=" / ".join(targets)),
        parse_mode="Markdown", reply_markup=kb)


def on_publish_decision(chat_id: int, decision: str, bot: Any, t: Callable) -> None:
    sess = _sessions.get(chat_id, {})
    if sess.get("step") != "confirming_publish":
        return
    if decision == "cancel":
        sess["step"] = "post_preview"
        _show_post_preview(chat_id, bot, t)
        return
    if decision == "vk":
        _publish_to_vk(chat_id, bot, t)
        return
    # "confirm" is the pre-v2026.8.82 callback for the Telegram target — still honoured
    # so a keyboard left open in an old chat keeps working after the upgrade.
    cfg = _get_pub_config(chat_id)
    if not cfg["token"] or not cfg["channel"]:
        sess["step"] = "post_preview"
        _show_publish_not_configured(chat_id, bot, t)
        return
    channel = cfg["channel"]
    token   = cfg["token"]
    content = sess.get("post_content", "")
    image   = sess.get("image_path", "")
    bot.send_message(chat_id,
        t(chat_id, "content_publishing").format(channel=channel),
        parse_mode="Markdown")

    def _run():
        try:
            if image:
                resp_data = _tg_send_photo(token, channel, content, image)
            else:
                import urllib.request, urllib.parse, json as _json
                url  = f"https://api.telegram.org/bot{token}/sendMessage"
                data_bytes = urllib.parse.urlencode({
                    "chat_id":    channel,
                    "text":       content,
                    "parse_mode": "Markdown",
                }).encode()
                req = urllib.request.Request(url, data=data_bytes, method="POST")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    resp_data = _json.loads(resp.read().decode())
            if resp_data.get("ok"):
                _record_publication(chat_id, sess, channel="telegram", target=channel,
                                    external_id=str((resp_data.get("result") or {})
                                                    .get("message_id", "")),
                                    external_url=_tg_post_url(channel, resp_data))
                bot.send_message(chat_id,
                    t(chat_id, "content_published").format(channel=channel),
                    parse_mode="Markdown")
            else:
                err = resp_data.get("description", "Unknown error")
                bot.send_message(chat_id,
                    t(chat_id, "content_publish_error").format(error=err),
                    parse_mode="Markdown")
        except Exception as exc:
            log.exception("[content] publish error: %s", exc)
            bot.send_message(chat_id,
                t(chat_id, "content_publish_error").format(error=str(exc)),
                parse_mode="Markdown")
        finally:
            if sess.get("plan_slug"):
                sess["step"] = "plan_saved"
                _show_plan_with_post_buttons(chat_id, bot, t)
            else:
                _sessions.pop(chat_id, None)

    threading.Thread(target=_run, daemon=True).start()


# Telegram caps a photo caption at 1024 characters — a longer post is sent as picture
# first, text second, rather than being silently truncated by the API.
_TG_CAPTION_LIMIT = 1024


def _tg_send_photo(token: str, channel: str, content: str, image_rel: str) -> dict:
    """Publish picture + text to a channel. Returns the Telegram API response dict."""
    import requests
    from core import bot_images
    path = bot_images.image_path(image_rel)
    if not path or not path.is_file():
        raise RuntimeError("illustration file is missing")
    caption = content if len(content) <= _TG_CAPTION_LIMIT else ""
    with path.open("rb") as fh:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendPhoto",
            data={"chat_id": channel, "caption": caption, "parse_mode": "Markdown"},
            files={"photo": (path.name, fh, "image/png")},
            timeout=60,
        )
    data = resp.json()
    if data.get("ok") and not caption:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": channel, "text": content, "parse_mode": "Markdown"},
                      timeout=30)
    return data


def _tg_post_url(channel: str, resp_data: dict) -> str:
    """https://t.me/<name>/<id> for a public @channel; '' for a numeric -100… id."""
    name = (channel or "").lstrip("@")
    msg_id = (resp_data.get("result") or {}).get("message_id", "")
    if not name or name.startswith("-") or not msg_id:
        return ""
    return f"https://t.me/{name}/{msg_id}"


def _publish_to_vk(chat_id: int, bot: Any, t: Callable) -> None:
    """Publish the current post to the user's VK community, picture included."""
    sess = _session(chat_id)
    cfg = _get_pub_config(chat_id)
    content = sess.get("post_content", "")
    image   = sess.get("image_path", "")
    bot.send_message(chat_id,
        t(chat_id, "content_publishing").format(channel=f"VK {cfg['vk_group']}"),
        parse_mode="Markdown")

    def _run():
        try:
            from features import bot_content_vk as vk
            image_file = None
            if image:
                from core import bot_images
                image_file = bot_images.image_path(image)
            result = vk.publish(cfg["vk_token"], cfg["vk_group"], content, image_file)
            if result.get("ok"):
                _record_publication(chat_id, sess, channel="vk",
                                    target=str(cfg["vk_group"]),
                                    external_id=result.get("post_id", ""),
                                    external_url=result.get("url", ""),
                                    followers=vk.fetch_followers(cfg["vk_token"],
                                                                 cfg["vk_group"]))
                bot.send_message(chat_id,
                    t(chat_id, "content_published_vk").format(url=result.get("url", "vk.com")),
                    parse_mode="Markdown")
            else:
                bot.send_message(chat_id,
                    t(chat_id, "content_publish_error").format(error=result.get("error", "?")),
                    parse_mode="Markdown")
        except Exception as exc:
            log.exception("[content] VK publish error: %s", exc)
            bot.send_message(chat_id,
                t(chat_id, "content_publish_error").format(error=str(exc)),
                parse_mode="Markdown")
        finally:
            if sess.get("plan_slug"):
                sess["step"] = "plan_saved"
                _show_plan_with_post_buttons(chat_id, bot, t)
            else:
                _sessions.pop(chat_id, None)

    threading.Thread(target=_run, daemon=True).start()


def _record_publication(chat_id: int, sess: dict, *, channel: str, target: str,
                        external_id: str = "", external_url: str = "",
                        followers: int = 0) -> None:
    """Move the post into the statistics ledger as published and open its metric series.

    The zero row matters: a published post with no observation at all would be invisible
    in every "measured" count, and the user would have nothing to update.
    """
    try:
        from core import store_content as SC
        post_id = _track_post(chat_id, sess)
        if not post_id:
            return
        SC.mark_published(post_id, channel=channel, external_id=external_id,
                          external_url=external_url)
        SC.update_post(post_id, channel=channel)
        SC.record_metrics(chat_id, post_id,
                          source="vk" if channel == "vk" else "telegram",
                          followers=followers)
        log.info("[content] published post %s to %s (%s)", post_id, channel, target)
    except Exception as exc:
        log.warning("[content] recording publication failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Publish configuration flow (per-user token + channel)
# ─────────────────────────────────────────────────────────────────────────────

def show_pub_config(chat_id: int, bot: Any, t: Callable) -> None:
    """Show current publish settings with edit buttons."""
    from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
    cfg = _get_pub_config(chat_id)
    token_display   = ("\u2022" * 10 + cfg["token"][-4:]) if len(cfg["token"]) > 4 else (cfg["token"] or "—")
    channel_display = cfg["channel"] or "—"
    text = t(chat_id, "content_pub_config_status").format(
        token=token_display, channel=channel_display,
        vk_token=_mask(cfg["vk_token"]), vk_group=cfg["vk_group"] or "—")
    kb = InlineKeyboardMarkup(row_width=1)
    kb.add(
        InlineKeyboardButton(t(chat_id, "content_btn_set_token"),   callback_data="content_pub_set:token"),
        InlineKeyboardButton(t(chat_id, "content_btn_set_channel"), callback_data="content_pub_set:channel"),
    )
    try:
        from features.bot_content_vk import is_enabled as _vk_on
        vk_enabled = _vk_on()
    except Exception:
        vk_enabled = False
    if vk_enabled:
        kb.add(
            InlineKeyboardButton(t(chat_id, "content_btn_set_vk_token"), callback_data="content_pub_set:vk_token"),
            InlineKeyboardButton(t(chat_id, "content_btn_set_vk_group"), callback_data="content_pub_set:vk_group"),
        )
    kb.add(InlineKeyboardButton(t(chat_id, "content_btn_back"), callback_data="content_start"))
    bot.send_message(chat_id, text, parse_mode="Markdown", reply_markup=kb)


def _mask(secret: str) -> str:
    """A credential SHAPE for the screen — never the credential."""
    return ("•" * 10 + secret[-4:]) if len(secret or "") > 4 else (secret or "—")


# Which prompt to show and where to store it, per configurable field. A table rather than
# four near-identical branches: adding a target must not mean copying the same code again.
_PUB_FIELDS = {
    "token":    (_PUB_TOKEN_KEY,    "config_pub_token",    "content_ask_pub_token",   "content_pub_token_saved"),
    "channel":  (_PUB_CHANNEL_KEY,  "config_pub_channel",  "content_ask_pub_channel", "content_pub_channel_saved"),
    "vk_token": (_PUB_VK_TOKEN_KEY, "config_pub_vk_token", "content_ask_vk_token",    "content_vk_token_saved"),
    "vk_group": (_PUB_VK_GROUP_KEY, "config_pub_vk_group", "content_ask_vk_group",    "content_vk_group_saved"),
}
_PUB_STEPS = {step: (pref_key, ask_key, ok_key)
              for pref_key, step, ask_key, ok_key in _PUB_FIELDS.values()}


def on_pub_set(chat_id: int, field: str, bot: Any, t: Callable) -> None:
    """Begin interactive input for a publish credential."""
    entry = _PUB_FIELDS.get(field)
    if not entry:
        return
    _pref_key, step, ask_key, _ok_key = entry
    _session(chat_id)["step"] = step
    bot.send_message(chat_id, t(chat_id, ask_key), parse_mode="Markdown")


def on_pub_config_input(chat_id: int, text: str, bot: Any, t: Callable) -> bool:
    """Handle credential text input during the config flow. Returns True if consumed."""
    sess = _sessions.get(chat_id)
    if not sess:
        return False
    entry = _PUB_STEPS.get(sess.get("step", ""))
    if not entry:
        return False
    pref_key, ask_key, ok_key = entry
    value = text.strip()
    if not value:
        bot.send_message(chat_id, t(chat_id, ask_key), parse_mode="Markdown")
        return True
    _set_pub_config_value(chat_id, pref_key, value)
    sess.pop("step", None)
    bot.send_message(chat_id, t(chat_id, ok_key), parse_mode="Markdown")
    show_pub_config(chat_id, bot, t)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Publication statistics — the numbers screen
# ─────────────────────────────────────────────────────────────────────────────

def show_stats(chat_id: int, bot: Any, t: Callable) -> None:
    """Print the statistics card + recent posts, each with an 'update numbers' button."""
    from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
    sess = _session(chat_id)
    sess["step"] = "stats"
    try:
        from core import store_content as SC
        summary = SC.stats_summary(chat_id)
        posts = SC.list_posts(chat_id, limit=8)
    except Exception as exc:
        log.warning("[content] stats failed: %s", exc)
        bot.send_message(chat_id, t(chat_id, "content_stats_error"), parse_mode="Markdown")
        return

    if not summary["posts_total"]:
        kb = InlineKeyboardMarkup(row_width=1)
        kb.add(InlineKeyboardButton(t(chat_id, "content_btn_back"), callback_data="content_start"))
        bot.send_message(chat_id, t(chat_id, "content_stats_empty"),
                         parse_mode="Markdown", reply_markup=kb)
        return

    topics = ", ".join(f"{r['topic']} ({r['posts']})" for r in summary["top_topics"]) or "—"
    text = t(chat_id, "content_stats_card").format(
        posts=summary["posts_total"], published=summary["published"],
        drafts=summary["drafts"], likes=summary["likes"], views=summary["views"],
        followers=summary["followers"], engagement=summary["engagement_avg"],
        images=summary["with_image"], topics=topics)
    if summary["monthly"]:
        # A bar per month, drawn in text: the Telegram screen has no canvas, and a
        # column of numbers hides the trend the web dashboard shows as a chart.
        text += "\n\n" + t(chat_id, "content_stats_by_month") + "\n" + "\n".join(
            f"`{row['month']}`  {'#' * min(20, row['posts'])} {row['posts']}"
            for row in summary["monthly"])

    kb = InlineKeyboardMarkup(row_width=1)
    for post in posts:
        m = post["metrics"]
        marker = "[+]" if post["status"] == "published" else "[.]"
        label = (f"{marker} {(post['title'] or '—')[:28]} · "
                 f"{m.get('likes', 0)}/{m.get('views', 0)}")
        kb.add(InlineKeyboardButton(label[:64],
                                    callback_data=f"content_metrics:{post['id']}"))
    if is_vk_configured(chat_id):
        kb.add(InlineKeyboardButton(t(chat_id, "content_btn_stats_sync"),
                                    callback_data="content_stats_sync"))
    kb.add(InlineKeyboardButton(t(chat_id, "content_btn_back"), callback_data="content_start"))
    try:
        bot.send_message(chat_id, text, parse_mode="Markdown", reply_markup=kb)
    except Exception:
        bot.send_message(chat_id, text, reply_markup=kb)


def on_metrics_selected(chat_id: int, post_id: str, bot: Any, t: Callable) -> None:
    """User picked a post — show what is known and ask for its current numbers."""
    from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
    try:
        from core import store_content as SC
        post = SC.get_post(post_id, chat_id)
    except Exception as exc:
        log.warning("[content] metrics lookup failed: %s", exc)
        return
    if not post:
        bot.send_message(chat_id, t(chat_id, "content_stats_gone"), parse_mode="Markdown")
        return
    sess = _session(chat_id)
    sess["step"] = "metrics_input"
    sess["metrics_post_id"] = post_id
    m = post["metrics"]
    kb = InlineKeyboardMarkup(row_width=2)
    kb.add(
        InlineKeyboardButton(t(chat_id, "content_btn_post_delete"),
                             callback_data=f"content_post_del:{post_id}"),
        InlineKeyboardButton(t(chat_id, "content_btn_back"), callback_data="content_stats"),
    )
    bot.send_message(chat_id,
        t(chat_id, "content_metrics_prompt").format(
            title=(post["title"] or "—")[:60], likes=m.get("likes", 0),
            views=m.get("views", 0), followers=m.get("followers", 0),
            engagement=post["engagement"]),
        parse_mode="Markdown", reply_markup=kb)


def on_metrics_input(chat_id: int, text: str, bot: Any, t: Callable) -> bool:
    """Parse 'likes views followers' and append the observation. True when consumed."""
    sess = _sessions.get(chat_id, {})
    post_id = sess.get("metrics_post_id", "")
    if not post_id:
        return False
    numbers = _parse_metrics(text)
    if not numbers:
        bot.send_message(chat_id, t(chat_id, "content_metrics_bad_input"), parse_mode="Markdown")
        return True
    likes, views, followers = numbers
    try:
        from core import store_content as SC
        SC.record_metrics(chat_id, post_id, likes=likes, views=views,
                          followers=followers, source="manual")
    except Exception as exc:
        log.warning("[content] record metrics failed: %s", exc)
        bot.send_message(chat_id, t(chat_id, "content_stats_error"), parse_mode="Markdown")
        return True
    sess.pop("metrics_post_id", None)
    bot.send_message(chat_id, t(chat_id, "content_metrics_saved"), parse_mode="Markdown")
    show_stats(chat_id, bot, t)
    return True


def _parse_metrics(text: str) -> tuple[int, int, int] | None:
    """'245 3200 18000' → (245, 3200, 18000); missing trailing values become 0.

    Only the first three integers are read, in the order the prompt asks for them: a
    person typing '245, 3200 просмотров, 18000' must not have the parser guess which
    number is which.
    """
    import re
    found = re.findall(r"\d+", text or "")
    if not found:
        return None
    values = [int(v) for v in found[:3]]
    while len(values) < 3:
        values.append(0)
    return values[0], values[1], values[2]


def on_stats_sync(chat_id: int, bot: Any, t: Callable) -> None:
    """Pull the real numbers for every VK-published post."""
    cfg = _get_pub_config(chat_id)
    if not is_vk_configured(chat_id):
        bot.send_message(chat_id, t(chat_id, "content_vk_not_configured"), parse_mode="Markdown")
        return
    bot.send_message(chat_id, t(chat_id, "content_stats_syncing"), parse_mode="Markdown")

    def _run():
        try:
            from core import store_content as SC
            from features import bot_content_vk as vk
            posts = SC.list_posts(chat_id, limit=200, status="published")
            result = vk.sync_metrics(cfg["vk_token"], cfg["vk_group"], posts)
            bot.send_message(chat_id,
                t(chat_id, "content_stats_synced").format(
                    updated=result["updated"], errors=result["errors"],
                    followers=result["followers"]),
                parse_mode="Markdown")
        except Exception as exc:
            log.warning("[content] VK sync failed: %s", exc)
            bot.send_message(chat_id, t(chat_id, "content_stats_error"), parse_mode="Markdown")
        finally:
            show_stats(chat_id, bot, t)

    threading.Thread(target=_run, daemon=True).start()


def on_tracked_post_delete(chat_id: int, post_id: str, bot: Any, t: Callable) -> None:
    """Remove one post from the statistics ledger (with its image and metric history)."""
    ok = _delete_tracked_post(chat_id, post_id)
    sess = _sessions.get(chat_id, {})
    sess.pop("metrics_post_id", None)
    bot.send_message(chat_id,
                     t(chat_id, "content_delete_done" if ok else "content_stats_gone"),
                     parse_mode="Markdown")
    show_stats(chat_id, bot, t)


# ─────────────────────────────────────────────────────────────────────────────
# Common helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_kb(chat_id: int, sess: dict) -> str:
    try:
        from telegram.bot_access import _docs_rag_context
        return _docs_rag_context(chat_id, sess.get("q1", "")) or ""
    except Exception as exc:
        log.warning("[content] KB fetch failed: %s", exc)
        return ""


def _do_download(chat_id: int, bot: Any, t: Callable,
                 content_key: str = "post_content", label: str = "content") -> None:
    sess = _sessions.get(chat_id, {})
    content = sess.get(content_key, "")
    try:
        filename = f"{label}_{uuid.uuid4().hex[:8]}.txt"
        bot.send_document(chat_id, (filename, io.BytesIO(content.encode("utf-8"))))
    except Exception as exc:
        log.warning("[content] download error: %s", exc)
        bot.send_message(chat_id,
            t(chat_id, "content_generate_error").format(error=str(exc)),
            parse_mode="Markdown")


# ─────────────────────────────────────────────────────────────────────────────
# Text message handler
# ─────────────────────────────────────────────────────────────────────────────

def handle_message(chat_id: int, text: str, bot: Any, t: Callable) -> bool:
    """Process incoming text. Returns True if consumed."""
    sess = _sessions.get(chat_id)
    if not sess:
        return False
    step = sess.get("step")

    if step == "q1":
        sess["q1"] = text.strip()
        sess["step"] = "q2"
        _ask_platform(chat_id, bot, t)
        return True

    if step == "correcting_plan":
        _do_generate_plan(chat_id, bot, t, correction=text.strip())
        return True

    if step == "correcting_post":
        _do_generate_post(chat_id, bot, t,
                          post_index=sess.get("post_index", 0),
                          correction=text.strip())
        return True

    # metric entry for a tracked post ("likes views followers")
    if step == "metrics_input" and on_metrics_input(chat_id, text, bot, t):
        return True

    # publish config input steps
    if on_pub_config_input(chat_id, text, bot, t):
        return True

    return False

