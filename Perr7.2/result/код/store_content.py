"""store_content.py — data access for the Content Strategist agent's posts and metrics.

The agent used to keep everything it produced in the notes table, which is fine for a
text draft and useless for a publication: a note has no platform, no image, no publish
state and above all no numbers, so nothing could ever be counted. These two tables are
what makes a *published post* a first-class object:

``content_posts``    one row per generated post — topic, platform, body, image, publish state
``content_metrics``  a time series of observed numbers for a post (likes/views/followers)

Metrics are a series, never a column on the post: the same post is measured again a day
later, and overwriting would destroy the history the monthly chart is drawn from. The
"current" value of a post is simply its newest row.

Engagement is **computed**, never stored — (likes + shares + comments) / views — so the
dashboard cannot disagree with the raw numbers it prints next to it.

Dual backend (PostgreSQL + SQLite) for the same reason as ``store_observer.py``: a schema
that lands in one adapter only is a latent 500 on every deployment running the other.

Dependency chain: bot_config → bot_db → store_content   (no feature imports)
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

import core.bot_config  # noqa: F401  (import for its env-loading side effect)

log = logging.getLogger("taris.content_store")

POST_STATUSES = ("draft", "published")
METRIC_SOURCES = ("manual", "telegram", "vk")


def _is_postgres() -> bool:
    return os.environ.get("STORE_BACKEND", "sqlite").lower() == "postgres"


def _pg_dsn() -> str:
    return os.environ.get("STORE_PG_DSN") or os.environ.get("DATABASE_URL") or ""


_pg_pool = None


def _get_pg_pool():
    global _pg_pool
    if _pg_pool is None:
        dsn = _pg_dsn()
        if not dsn:
            raise RuntimeError("STORE_PG_DSN not configured")
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool
        _pg_pool = ConnectionPool(dsn, min_size=1, max_size=2,
                                  kwargs={"row_factory": dict_row})
        log.info("[ContentStore] PostgreSQL pool created")
    return _pg_pool


@contextmanager
def _conn():
    """Yield (connection, is_pg). Postgres uses a small pool; SQLite the thread-local conn."""
    if _is_postgres():
        with _get_pg_pool().connection() as c:
            yield c, True
    else:
        from core.bot_db import get_db
        yield get_db(), False


def _q(sql: str, is_pg: bool) -> str:
    return sql.replace("?", "%s") if is_pg else sql


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rows(cur) -> list[dict]:
    """Normalise both adapters to plain dicts — sqlite3.Row and psycopg dict_row differ."""
    out = []
    for r in cur.fetchall() or []:
        out.append(dict(r))
    return out


def _one(cur) -> dict:
    r = cur.fetchone()
    return dict(r) if r else {}


# ─── Schema ──────────────────────────────────────────────────────────────────

_SCHEMA_PG = [
    """CREATE TABLE IF NOT EXISTS content_posts (
        id           TEXT PRIMARY KEY,
        chat_id      BIGINT NOT NULL,
        created_at   TIMESTAMPTZ DEFAULT NOW(),
        updated_at   TIMESTAMPTZ DEFAULT NOW(),
        topic        TEXT NOT NULL DEFAULT '',
        platform     TEXT NOT NULL DEFAULT '',
        title        TEXT NOT NULL DEFAULT '',
        body         TEXT NOT NULL DEFAULT '',
        image_path   TEXT NOT NULL DEFAULT '',
        image_prompt TEXT NOT NULL DEFAULT '',
        plan_slug    TEXT NOT NULL DEFAULT '',
        post_index   INTEGER DEFAULT 0,
        status       TEXT NOT NULL DEFAULT 'draft',
        published_at TIMESTAMPTZ,
        channel      TEXT NOT NULL DEFAULT '',
        external_id  TEXT NOT NULL DEFAULT '',
        external_url TEXT NOT NULL DEFAULT '',
        lang         TEXT NOT NULL DEFAULT ''
    )""",
    """CREATE TABLE IF NOT EXISTS content_metrics (
        id           TEXT PRIMARY KEY,
        post_id      TEXT NOT NULL,
        chat_id      BIGINT NOT NULL,
        collected_at TIMESTAMPTZ DEFAULT NOW(),
        source       TEXT NOT NULL DEFAULT 'manual',
        likes        INTEGER DEFAULT 0,
        views        INTEGER DEFAULT 0,
        shares       INTEGER DEFAULT 0,
        comments     INTEGER DEFAULT 0,
        followers    INTEGER DEFAULT 0
    )""",
    "CREATE INDEX IF NOT EXISTS idx_content_posts_chat ON content_posts(chat_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_content_metrics_post ON content_metrics(post_id, collected_at DESC)",
]

_SCHEMA_SQLITE = [
    s.replace("TIMESTAMPTZ DEFAULT NOW()", "TEXT")
     .replace("TIMESTAMPTZ", "TEXT")
     .replace("BIGINT", "INTEGER")
    for s in _SCHEMA_PG
]

_schema_ready = False


def init_schema(force: bool = False) -> None:
    """Create the content tables. Idempotent; runs once per process."""
    global _schema_ready
    if _schema_ready and not force:
        return
    with _conn() as (c, is_pg):
        for sql in (_SCHEMA_PG if is_pg else _SCHEMA_SQLITE):
            try:
                c.execute(sql)
                # Commit per statement: on PostgreSQL a failed DDL aborts the whole
                # transaction, so one bad statement would silently drop the rest.
                c.commit()
            except Exception as exc:                      # pragma: no cover - defensive
                log.warning("[ContentStore] DDL failed: %s: %s (%.60s)",
                            type(exc).__name__, exc, sql.strip())
                try:
                    c.rollback()
                except Exception:
                    pass
    _schema_ready = True
    log.info("[ContentStore] schema ready (%s)", "postgres" if _is_postgres() else "sqlite")


def schema_present() -> bool:
    """True when the content tables can be queried — used by the UI to stay silent about
    a feature whose migration has not run yet, instead of 500-ing."""
    try:
        with _conn() as (c, is_pg):
            c.execute(_q("SELECT 1 FROM content_posts LIMIT 1", is_pg))
            return True
    except Exception:
        return False


# ─── Posts ───────────────────────────────────────────────────────────────────

def save_post(chat_id: int, body: str, *, topic: str = "", platform: str = "",
              title: str = "", image_path: str = "", image_prompt: str = "",
              plan_slug: str = "", post_index: int = 0, lang: str = "",
              status: str = "draft") -> str:
    """Insert a post and return its id."""
    init_schema()
    post_id = uuid.uuid4().hex[:16]
    if status not in POST_STATUSES:
        status = "draft"
    if not title:
        title = _derive_title(body)
    with _conn() as (c, is_pg):
        c.execute(_q(
            "INSERT INTO content_posts (id, chat_id, created_at, updated_at, topic, platform,"
            " title, body, image_path, image_prompt, plan_slug, post_index, status, lang)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", is_pg),
            (post_id, int(chat_id), _now_iso(), _now_iso(), topic[:500], platform[:40],
             title[:200], body, image_path[:300], image_prompt[:2000], plan_slug[:80],
             int(post_index or 0), status, lang[:8]))
        c.commit()
    return post_id


def _derive_title(body: str) -> str:
    """First non-empty line, stripped of Markdown heading marks and emphasis."""
    for line in (body or "").splitlines():
        clean = line.strip().lstrip("#*_ ").strip("*_ ").strip()
        if clean:
            return clean[:200]
    return "—"


def update_post(post_id: str, **fields: Any) -> None:
    """Update named columns of a post. Unknown columns are ignored, never guessed."""
    allowed = {"topic", "platform", "title", "body", "image_path", "image_prompt",
               "plan_slug", "post_index", "status", "published_at", "channel",
               "external_id", "external_url", "lang"}
    sets = {k: v for k, v in fields.items() if k in allowed}
    if not sets:
        return
    init_schema()
    cols = ", ".join(f"{k}=?" for k in sets)
    with _conn() as (c, is_pg):
        c.execute(_q(f"UPDATE content_posts SET {cols}, updated_at=? WHERE id=?", is_pg),
                  (*sets.values(), _now_iso(), post_id))
        c.commit()


def get_post(post_id: str, chat_id: int | None = None) -> dict:
    """Return one post (with its latest metrics) or {}. Pass chat_id to enforce ownership."""
    init_schema()
    with _conn() as (c, is_pg):
        if chat_id is None:
            cur = c.execute(_q("SELECT * FROM content_posts WHERE id=?", is_pg), (post_id,))
        else:
            cur = c.execute(_q("SELECT * FROM content_posts WHERE id=? AND chat_id=?", is_pg),
                            (post_id, int(chat_id)))
        row = _one(cur)
    if not row:
        return {}
    row["metrics"] = latest_metrics([post_id]).get(post_id, _zero_metrics())
    row["engagement"] = engagement(row["metrics"])
    return row


def list_posts(chat_id: int, *, limit: int = 50, status: str = "") -> list[dict]:
    """Posts of one user, newest first, each carrying its latest metrics + engagement."""
    init_schema()
    with _conn() as (c, is_pg):
        if status:
            cur = c.execute(_q(
                "SELECT * FROM content_posts WHERE chat_id=? AND status=?"
                " ORDER BY created_at DESC LIMIT ?", is_pg),
                (int(chat_id), status, int(limit)))
        else:
            cur = c.execute(_q(
                "SELECT * FROM content_posts WHERE chat_id=?"
                " ORDER BY created_at DESC LIMIT ?", is_pg),
                (int(chat_id), int(limit)))
        rows = _rows(cur)
    metrics = latest_metrics([r["id"] for r in rows])
    for r in rows:
        r["metrics"] = metrics.get(r["id"], _zero_metrics())
        r["engagement"] = engagement(r["metrics"])
    return rows


def delete_post(chat_id: int, post_id: str) -> bool:
    """Delete a post and its metric history. Returns False when it is not the user's."""
    init_schema()
    with _conn() as (c, is_pg):
        cur = c.execute(_q("SELECT id FROM content_posts WHERE id=? AND chat_id=?", is_pg),
                        (post_id, int(chat_id)))
        if not _one(cur):
            return False
        c.execute(_q("DELETE FROM content_metrics WHERE post_id=?", is_pg), (post_id,))
        c.execute(_q("DELETE FROM content_posts WHERE id=?", is_pg), (post_id,))
        c.commit()
    return True


def mark_published(post_id: str, *, channel: str = "", external_id: str = "",
                   external_url: str = "") -> None:
    update_post(post_id, status="published", published_at=_now_iso(),
                channel=channel[:120], external_id=str(external_id)[:80],
                external_url=external_url[:300])


# ─── Metrics ─────────────────────────────────────────────────────────────────

def _zero_metrics() -> dict:
    return {"likes": 0, "views": 0, "shares": 0, "comments": 0, "followers": 0,
            "source": "", "collected_at": ""}


def engagement(m: dict) -> float:
    """(likes + shares + comments) / views, in percent, one decimal. 0.0 without views."""
    views = int(m.get("views") or 0)
    if views <= 0:
        return 0.0
    reactions = int(m.get("likes") or 0) + int(m.get("shares") or 0) + int(m.get("comments") or 0)
    return round(reactions * 100.0 / views, 1)


def record_metrics(chat_id: int, post_id: str, *, likes: int = 0, views: int = 0,
                   shares: int = 0, comments: int = 0, followers: int = 0,
                   source: str = "manual") -> str:
    """Append one observation for a post. Never overwrites — the history is the chart."""
    init_schema()
    mid = uuid.uuid4().hex[:16]
    if source not in METRIC_SOURCES:
        source = "manual"
    with _conn() as (c, is_pg):
        c.execute(_q(
            "INSERT INTO content_metrics (id, post_id, chat_id, collected_at, source,"
            " likes, views, shares, comments, followers) VALUES (?,?,?,?,?,?,?,?,?,?)", is_pg),
            (mid, post_id, int(chat_id), _now_iso(), source,
             max(0, int(likes or 0)), max(0, int(views or 0)), max(0, int(shares or 0)),
             max(0, int(comments or 0)), max(0, int(followers or 0))))
        c.commit()
    return mid


def latest_metrics(post_ids: list[str]) -> dict[str, dict]:
    """Newest observation per post id. One query, ordered oldest→newest so the last write wins."""
    if not post_ids:
        return {}
    init_schema()
    placeholders = ",".join("?" for _ in post_ids)
    with _conn() as (c, is_pg):
        cur = c.execute(_q(
            f"SELECT * FROM content_metrics WHERE post_id IN ({placeholders})"
            " ORDER BY collected_at ASC", is_pg), tuple(post_ids))
        rows = _rows(cur)
    out: dict[str, dict] = {}
    for r in rows:
        out[r["post_id"]] = r
    return out


def metric_history(chat_id: int, post_id: str, limit: int = 30) -> list[dict]:
    init_schema()
    with _conn() as (c, is_pg):
        cur = c.execute(_q(
            "SELECT * FROM content_metrics WHERE post_id=? AND chat_id=?"
            " ORDER BY collected_at DESC LIMIT ?", is_pg),
            (post_id, int(chat_id), int(limit)))
        return _rows(cur)


# ─── Statistics ──────────────────────────────────────────────────────────────

def _month_of(value: Any) -> str:
    """'YYYY-MM' from a datetime or an ISO string. '' when unparseable."""
    if not value:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m")
    text = str(value)
    return text[:7] if len(text) >= 7 else ""


def stats_summary(chat_id: int, *, months: int = 6, top_n: int = 5) -> dict:
    """Everything the statistics dashboard prints, computed in one pass over the posts.

    Returns totals, averages, the per-month series, the topic ranking and the platform
    split. Followers is the **maximum** observed value, not a sum: the same audience
    counted once per post would multiply an audience of 18 000 by the number of posts.
    """
    posts = list_posts(chat_id, limit=1000)
    total = len(posts)
    published = sum(1 for p in posts if p.get("status") == "published")
    with_image = sum(1 for p in posts if p.get("image_path"))
    likes = views = shares = comments = 0
    followers = 0
    measured = 0
    for p in posts:
        m = p.get("metrics") or {}
        likes += int(m.get("likes") or 0)
        views += int(m.get("views") or 0)
        shares += int(m.get("shares") or 0)
        comments += int(m.get("comments") or 0)
        followers = max(followers, int(m.get("followers") or 0))
        if int(m.get("views") or 0) > 0:
            measured += 1

    eng_values = [p["engagement"] for p in posts if p.get("engagement")]
    engagement_avg = round(sum(eng_values) / len(eng_values), 1) if eng_values else 0.0

    # Topic ranking — the topic field is the user's own "theme / audience / goal" line,
    # so it is trimmed to its first clause to keep the ranking readable.
    topic_counts: dict[str, dict] = {}
    for p in posts:
        key = _topic_key(p.get("topic", ""))
        if not key:
            continue
        slot = topic_counts.setdefault(key, {"topic": key, "posts": 0, "likes": 0, "views": 0})
        slot["posts"] += 1
        slot["likes"] += int((p.get("metrics") or {}).get("likes") or 0)
        slot["views"] += int((p.get("metrics") or {}).get("views") or 0)
    top_topics = sorted(topic_counts.values(),
                        key=lambda r: (r["posts"], r["views"]), reverse=True)[:top_n]

    platform_counts: dict[str, int] = {}
    for p in posts:
        key = (p.get("platform") or "—").lower()
        platform_counts[key] = platform_counts.get(key, 0) + 1
    platforms = sorted(({"platform": k, "posts": v} for k, v in platform_counts.items()),
                       key=lambda r: r["posts"], reverse=True)

    # Per-month series, oldest→newest, gaps included so the chart has no invisible jumps.
    by_month: dict[str, dict] = {}
    for p in posts:
        mon = _month_of(p.get("created_at"))
        if not mon:
            continue
        slot = by_month.setdefault(mon, {"month": mon, "posts": 0, "likes": 0, "views": 0})
        slot["posts"] += 1
        slot["likes"] += int((p.get("metrics") or {}).get("likes") or 0)
        slot["views"] += int((p.get("metrics") or {}).get("views") or 0)
    monthly = [by_month[k] for k in sorted(by_month)][-months:]

    return {
        "posts_total":    total,
        "published":      published,
        "drafts":         total - published,
        "with_image":     with_image,
        "likes":          likes,
        "views":          views,
        "shares":         shares,
        "comments":       comments,
        "followers":      followers,
        "engagement_avg": engagement_avg,
        "measured":       measured,
        "top_topics":     top_topics,
        "platforms":      platforms,
        "monthly":        monthly,
    }


def _topic_key(topic: str) -> str:
    """First clause of the topic line, ≤60 chars — the ranking label."""
    text = (topic or "").strip()
    if not text:
        return ""
    for sep in (",", ".", ";", "\n", " / "):
        if sep in text:
            text = text.split(sep, 1)[0].strip()
            break
    return text[:60]
