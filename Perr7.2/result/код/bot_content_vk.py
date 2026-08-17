"""bot_content_vk.py — optional VK publishing and statistics for the Content Strategist.

Telegram's Bot API can deliver a post to a channel but will never tell us how it did:
there is no endpoint for a channel post's views, likes or the channel's subscriber count.
So the metrics the dashboard shows are entered by hand — unless the user publishes to a
VK community, which does expose all three. This module is that second, optional route:

    publish(...)        wall.post (with the generated image uploaded as an attachment)
    fetch_post_stats()  wall.getById   → likes / views / comments / reposts
    fetch_followers()   groups.getById → members_count

Credentials are **per user** (stored in user_prefs by bot_content, like the Telegram
publish token), never in the deployment config: one VK community belongs to one user, and
a shared token in bot.env would let every user of the box post to it.

Everything here degrades to ``{"ok": False, "error": …}``; a social network being down or
a token being revoked is an ordinary Tuesday, not a traceback.
"""

from __future__ import annotations

import logging
from pathlib import Path

from core.bot_config import (
    CONTENT_VK_API_VERSION,
    CONTENT_VK_ENABLED,
    CONTENT_VK_TIMEOUT,
)

log = logging.getLogger("taris.content_vk")

_API = "https://api.vk.com/method"


def is_enabled() -> bool:
    """Deployment-level switch. Per-user credentials are checked separately."""
    return bool(CONTENT_VK_ENABLED)


def is_configured(token: str, group_id: str) -> bool:
    return bool(is_enabled() and (token or "").strip() and str(group_id or "").strip())


def _group_id(group_id: str) -> str:
    """VK group ids are positive in parameters and negative as an owner_id. Accept both
    spellings from the user ('226084011' or '-226084011') and normalise."""
    return str(group_id or "").strip().lstrip("-")


def _call(method: str, token: str, params: dict, *, http_method: str = "get") -> dict:
    """One VK API call. Returns the 'response' payload or raises RuntimeError."""
    import requests
    args = dict(params)
    args["access_token"] = token
    args["v"] = CONTENT_VK_API_VERSION
    try:
        if http_method == "post":
            resp = requests.post(f"{_API}/{method}", data=args, timeout=CONTENT_VK_TIMEOUT)
        else:
            resp = requests.get(f"{_API}/{method}", params=args, timeout=CONTENT_VK_TIMEOUT)
        payload = resp.json()
    except Exception as exc:
        raise RuntimeError(f"{method}: {type(exc).__name__}: {exc}") from exc
    if "error" in payload:
        msg = payload["error"].get("error_msg", "unknown error")
        code = payload["error"].get("error_code", "?")
        raise RuntimeError(f"{method}: [{code}] {msg}")
    return payload.get("response", {})


# ─── Publishing ──────────────────────────────────────────────────────────────

def upload_photo(token: str, group_id: str, image_file: str | Path) -> str:
    """Upload a local image to the community's wall album; return its attachment id."""
    import requests
    gid = _group_id(group_id)
    server = _call("photos.getWallUploadServer", token, {"group_id": gid})
    upload_url = server.get("upload_url")
    if not upload_url:
        raise RuntimeError("photos.getWallUploadServer: no upload_url")
    path = Path(image_file)
    if not path.is_file():
        raise RuntimeError(f"image not found: {path.name}")
    with path.open("rb") as fh:
        up = requests.post(upload_url, files={"photo": (path.name, fh, "image/png")},
                           timeout=CONTENT_VK_TIMEOUT).json()
    if "error" in up:
        raise RuntimeError(f"upload: {up['error']}")
    saved = _call("photos.saveWallPhoto", token, {
        "group_id": gid,
        "photo":  up.get("photo", ""),
        "server": up.get("server", ""),
        "hash":   up.get("hash", ""),
    }, http_method="post")
    if not saved:
        raise RuntimeError("photos.saveWallPhoto returned nothing")
    first = saved[0] if isinstance(saved, list) else saved
    return f"photo{first['owner_id']}_{first['id']}"


def publish(token: str, group_id: str, text: str,
            image_file: str | Path | None = None) -> dict:
    """Post to the community wall. Returns {ok, post_id, url, error}.

    A failing image upload does not cancel the post: the text is what the user wrote,
    and losing it because a photo endpoint hiccuped would be the worse outcome. The
    caller is told through `image_error` so the UI can say the picture is missing.
    """
    if not is_configured(token, group_id):
        return {"ok": False, "post_id": "", "url": "", "error": "VK is not configured"}
    gid = _group_id(group_id)
    params: dict = {
        "owner_id": f"-{gid}",
        "from_group": 1,
        "message": (text or "")[:16000],
    }
    image_error = ""
    if image_file:
        try:
            params["attachments"] = upload_photo(token, gid, image_file)
        except Exception as exc:
            image_error = str(exc)
            log.warning("[VK] photo upload failed, posting text only: %s", exc)
    try:
        response = _call("wall.post", token, params, http_method="post")
    except Exception as exc:
        return {"ok": False, "post_id": "", "url": "", "error": str(exc),
                "image_error": image_error}
    post_id = str(response.get("post_id", ""))
    return {
        "ok": True,
        "post_id": post_id,
        "url": f"https://vk.com/wall-{gid}_{post_id}" if post_id else "",
        "error": "",
        "image_error": image_error,
    }


# ─── Statistics ──────────────────────────────────────────────────────────────

def fetch_post_stats(token: str, group_id: str, post_id: str) -> dict:
    """likes / views / comments / reposts of one wall post. {} on any failure."""
    if not is_configured(token, group_id) or not str(post_id).strip():
        return {}
    gid = _group_id(group_id)
    try:
        response = _call("wall.getById", token, {"posts": f"-{gid}_{post_id}"})
    except Exception as exc:
        log.warning("[VK] wall.getById failed: %s", exc)
        return {}
    # v5.236 answers {"items": [...]}; older versions answer a bare list.
    items = response.get("items") if isinstance(response, dict) else response
    if not items:
        return {}
    item = items[0]
    return {
        "likes":    int((item.get("likes") or {}).get("count", 0)),
        "views":    int((item.get("views") or {}).get("count", 0)),
        "comments": int((item.get("comments") or {}).get("count", 0)),
        "shares":   int((item.get("reposts") or {}).get("count", 0)),
    }


def fetch_followers(token: str, group_id: str) -> int:
    """Community members count, 0 when it cannot be read."""
    if not is_configured(token, group_id):
        return 0
    try:
        response = _call("groups.getById", token, {
            "group_id": _group_id(group_id), "fields": "members_count"})
    except Exception as exc:
        log.warning("[VK] groups.getById failed: %s", exc)
        return 0
    groups = response.get("groups") if isinstance(response, dict) else response
    if not groups:
        # Pre-5.130 shape: the response IS the list.
        return int((response or [{}])[0].get("members_count", 0)) if isinstance(response, list) else 0
    return int(groups[0].get("members_count", 0) or 0)


def sync_metrics(token: str, group_id: str, posts: list[dict]) -> dict:
    """Refresh the metrics of every VK-published post. Returns {"updated": n, "errors": n}.

    `posts` are content_posts rows; only those with channel='vk' and an external_id are
    touched. The followers count is read once, not once per post — it is a property of
    the community, and 40 identical calls is how a token gets rate-limited.
    """
    from core import store_content as SC
    updated = errors = 0
    followers = fetch_followers(token, group_id)
    for post in posts:
        if (post.get("channel") or "") != "vk" or not post.get("external_id"):
            continue
        stats = fetch_post_stats(token, group_id, post["external_id"])
        if not stats:
            errors += 1
            continue
        SC.record_metrics(int(post["chat_id"]), post["id"], source="vk",
                          likes=stats.get("likes", 0), views=stats.get("views", 0),
                          shares=stats.get("shares", 0), comments=stats.get("comments", 0),
                          followers=followers)
        updated += 1
    return {"updated": updated, "errors": errors, "followers": followers}
