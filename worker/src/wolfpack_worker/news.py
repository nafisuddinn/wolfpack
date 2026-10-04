"""News headlines for The Scout: fetch (Alpaca / Benzinga), store, refresh.

Source: Alpaca's news API via alpaca-py's `NewsClient`, using the existing
(free) Alpaca keys. History from 2015, roughly 200 calls/min, headline only
(the body is never requested or stored). Design: docs/design/scout-design.md.

Point-in-time discipline (Decision Log 2026-10-04):
* `created_at` (vendor publish time) is the ONLY timestamp a backtest uses:
  the feature window for the session closing at close_t is
  (close_{t-1}, close_t] on created_at (see scout/windows.py).
* `first_seen_at` is when WolfPack first stored the article. Rows are
  insert-once (ON CONFLICT DO NOTHING; the migration also rejects UPDATEs),
  so a later fetch can never move first_seen_at or swap in a revised
  headline. Live features additionally require first_seen_at <= the run's
  news cutoff, so a late-arriving article is never used retroactively.
* Backfilled rows get first_seen_at = the backfill time and
  ingest_mode = "backfill"; for them only created_at is meaningful. Vendor
  revisions (updated_at > created_at) and vendor archive backfill cannot be
  undone historically; they are measured (scout/coverage.py) and reported.

Licensing: headline text and URLs are PRIVATE (Supabase `news_articles`,
service_role only, or the gitignored local cache under worker/data/). They
must never reach `signal_payload`, the public feed, rationale text, MLflow
artifacts or any committed file.

Stores:
* SupabaseNewsStore: the production table (migration 20261004_0003).
* LocalNewsStore: the same semantics on an append-only JSONL file (or in
  memory with path=None). Used by tests, and as an offline training cache
  when the production table is not available. Never commit it.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

_WORKER_ROOT = Path(__file__).resolve().parents[2]
# Gitignored (.gitignore: worker/data/). Licensed headline text lives here.
DEFAULT_CACHE_PATH = _WORKER_ROOT / "data" / "news_cache" / "news_articles.jsonl"

TABLE = "news_articles"
INGEST_MODES = ("backfill", "live")
ARTICLE_COLUMNS = (
    "id",
    "created_at",
    "vendor_updated_at",
    "headline",
    "source",
    "url",
    "symbols",
    "first_seen_at",
    "ingest_mode",
)
# The daily refresh re-fetches this many calendar days every run: the Scout's
# features look back 25 sessions (~36 calendar days), so an article that the
# vendor publishes late (created_at backdated into an earlier window) is still
# picked up, stored with its real first_seen_at, and counted as a late arrival.
LIVE_REFETCH_DAYS = 45

_PAGE_SIZE = 1000
_UPSERT_CHUNK = 500


def _aware_utc(ts: Any, label: str) -> datetime:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware, got {ts!r}")
    return t.tz_convert("UTC").to_pydatetime()


@dataclass(frozen=True)
class NewsArticle:
    """One headline. No body, summary, author or images (headline only)."""

    id: int
    created_at: datetime
    vendor_updated_at: Optional[datetime]
    headline: str
    source: str
    url: Optional[str]
    symbols: tuple[str, ...]


def article_from_raw(raw: Mapping[str, Any]) -> NewsArticle:
    """Parse one article from Alpaca's raw news JSON (or alpaca-py News)."""
    get = raw.get if isinstance(raw, Mapping) else (lambda k, d=None: getattr(raw, k, d))
    upd = get("updated_at")
    return NewsArticle(
        id=int(get("id")),
        created_at=_aware_utc(get("created_at"), "created_at"),
        vendor_updated_at=_aware_utc(upd, "updated_at") if upd is not None else None,
        headline=str(get("headline") or ""),
        source=str(get("source") or ""),
        url=get("url") or None,
        symbols=tuple(str(s) for s in (get("symbols") or ())),
    )


class NewsFetcher(Protocol):
    def fetch(self, symbols: Sequence[str], start: datetime, end: datetime) -> list[NewsArticle]:
        """Articles mentioning any of `symbols` that the vendor returns for
        [start, end]. NOTE: Alpaca filters this range on the article's
        UPDATED time, not created_at (verified 2026-10-04: a query from
        2016-01-04 returns articles created in Dec 2015 and revised later).
        Every article created in [start, end] has updated_at >= created_at,
        so tiling a range by update time still collects all of them, and
        callers bucket by created_at themselves."""
        ...


class AlpacaNewsFetcher:
    """alpaca-py NewsClient (market-data API; free with the paper keys).

    Read-only market data: this client cannot place orders. Content is never
    requested (include_content=False); alpaca-py pages internally (50/page).
    The returned headline is the vendor's CURRENT text, i.e. the latest
    revision for an article revised after publication (the residual revision
    leak measured in scout/coverage.py).
    """

    def __init__(self, api_key: str, secret_key: str) -> None:
        from alpaca.data.historical.news import NewsClient

        self._client = NewsClient(api_key=api_key, secret_key=secret_key, raw_data=True)

    @classmethod
    def from_config(cls, config) -> "AlpacaNewsFetcher":
        return cls(config.alpaca_api_key, config.alpaca_secret_key)

    def fetch(self, symbols: Sequence[str], start: datetime, end: datetime) -> list[NewsArticle]:
        from alpaca.data.requests import NewsRequest

        req = NewsRequest(
            symbols=",".join(symbols),
            start=_aware_utc(start, "start"),
            end=_aware_utc(end, "end"),
            sort="asc",
            include_content=False,
            exclude_contentless=False,
        )
        raw = self._client.get_news(req)
        items = raw.get("news", []) if isinstance(raw, Mapping) else []
        return [article_from_raw(a) for a in items]


class NewsStore(Protocol):
    def insert_new(self, articles: Iterable[NewsArticle], *, first_seen_at: datetime, ingest_mode: str) -> int:
        """Insert articles not already stored (by id); never update. Returns
        the number of new rows."""
        ...

    def get_articles(self, symbols: Sequence[str], start: datetime, end: datetime) -> pd.DataFrame:
        """ARTICLE_COLUMNS for articles mentioning any of `symbols` with
        start < created_at <= end, sorted by (created_at, id)."""
        ...

    def earliest_created_at(self) -> Optional[datetime]: ...

    def latest_created_at(self) -> Optional[datetime]: ...


def _row(a: NewsArticle, first_seen_at: datetime, ingest_mode: str) -> dict[str, Any]:
    return {
        "id": int(a.id),
        "created_at": a.created_at.isoformat(),
        "vendor_updated_at": a.vendor_updated_at.isoformat() if a.vendor_updated_at else None,
        "headline": a.headline,
        "source": a.source,
        "url": a.url,
        "symbols": list(a.symbols),
        "first_seen_at": first_seen_at.isoformat(),
        "ingest_mode": ingest_mode,
    }


def _check_insert(first_seen_at: datetime, ingest_mode: str) -> datetime:
    if ingest_mode not in INGEST_MODES:
        raise ValueError(f"ingest_mode must be one of {INGEST_MODES}, got {ingest_mode!r}")
    return _aware_utc(first_seen_at, "first_seen_at")


def articles_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Rows (dicts with ARTICLE_COLUMNS) -> typed frame, sorted (created_at, id)."""
    if not rows:
        df = pd.DataFrame({c: pd.Series(dtype=object) for c in ARTICLE_COLUMNS})
        df["id"] = df["id"].astype("int64")
        for c in ("created_at", "vendor_updated_at", "first_seen_at"):
            df[c] = pd.to_datetime(df[c], utc=True)
        return df
    df = pd.DataFrame([{c: r.get(c) for c in ARTICLE_COLUMNS} for r in rows], columns=list(ARTICLE_COLUMNS))
    df["id"] = df["id"].astype("int64")
    for c in ("created_at", "vendor_updated_at", "first_seen_at"):
        df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601")
    df["symbols"] = df["symbols"].map(lambda s: tuple(s or ()))
    df = df.drop_duplicates("id").sort_values(["created_at", "id"], kind="mergesort").reset_index(drop=True)
    return df


class LocalNewsStore:
    """Append-only JSONL (one row per line), or in-memory with path=None.

    Same insert-once semantics as the production table. Holds licensed
    headline text: the default location is gitignored; never commit it.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else None
        self._rows: dict[int, dict[str, Any]] = {}
        if self.path is not None and self.path.exists():
            for i, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                self._rows.setdefault(int(rec["id"]), rec)  # first line for an id wins
        self._frame: Optional[pd.DataFrame] = None

    def insert_new(self, articles: Iterable[NewsArticle], *, first_seen_at: datetime, ingest_mode: str) -> int:
        first_seen_at = _check_insert(first_seen_at, ingest_mode)
        new = []
        for a in articles:
            if int(a.id) in self._rows:
                continue
            r = _row(a, first_seen_at, ingest_mode)
            self._rows[int(a.id)] = r
            new.append(r)
        if new and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                for r in new:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        if new:
            self._frame = None
        return len(new)

    def _all(self) -> pd.DataFrame:
        if self._frame is None:
            self._frame = articles_frame(list(self._rows.values()))
        return self._frame

    def get_articles(self, symbols: Sequence[str], start: datetime, end: datetime) -> pd.DataFrame:
        start, end = _aware_utc(start, "start"), _aware_utc(end, "end")
        df = self._all()
        want = set(symbols)
        mask = (df["created_at"] > start) & (df["created_at"] <= end)
        mask &= df["symbols"].map(lambda s: bool(want.intersection(s)))
        return df.loc[mask].reset_index(drop=True)

    def earliest_created_at(self) -> Optional[datetime]:
        df = self._all()
        return None if df.empty else df["created_at"].min().to_pydatetime()

    def latest_created_at(self) -> Optional[datetime]:
        df = self._all()
        return None if df.empty else df["created_at"].max().to_pydatetime()


class SupabaseNewsStore:
    """The private `news_articles` table (service-role client only)."""

    def __init__(self, client) -> None:
        self._client = client

    def insert_new(self, articles: Iterable[NewsArticle], *, first_seen_at: datetime, ingest_mode: str) -> int:
        first_seen_at = _check_insert(first_seen_at, ingest_mode)
        rows = [_row(a, first_seen_at, ingest_mode) for a in articles]
        inserted = 0
        for i in range(0, len(rows), _UPSERT_CHUNK):
            resp = (
                self._client.table(TABLE)
                .upsert(rows[i : i + _UPSERT_CHUNK], on_conflict="id", ignore_duplicates=True)
                .execute()
            )
            inserted += len(resp.data or [])
        return inserted

    def get_articles(self, symbols: Sequence[str], start: datetime, end: datetime) -> pd.DataFrame:
        start, end = _aware_utc(start, "start"), _aware_utc(end, "end")
        out: list[dict] = []
        offset = 0
        while True:
            resp = (
                self._client.table(TABLE)
                .select(",".join(ARTICLE_COLUMNS))
                .ov("symbols", list(symbols))
                .gt("created_at", start.isoformat())
                .lte("created_at", end.isoformat())
                .order("created_at", desc=False)
                .order("id", desc=False)
                .range(offset, offset + _PAGE_SIZE - 1)
                .execute()
            )
            page = resp.data or []
            if not page:
                break
            out.extend(page)
            offset += len(page)
        return articles_frame(out)

    def _edge(self, desc: bool) -> Optional[datetime]:
        resp = self._client.table(TABLE).select("created_at").order("created_at", desc=desc).limit(1).execute()
        if not resp.data:
            return None
        return pd.Timestamp(resp.data[0]["created_at"]).tz_convert("UTC").to_pydatetime()

    def earliest_created_at(self) -> Optional[datetime]:
        return self._edge(False)

    def latest_created_at(self) -> Optional[datetime]:
        return self._edge(True)


@dataclass(frozen=True)
class NewsRefreshResult:
    """Outcome of the daily refresh. `cutoff` is the news cutoff for this run
    (live features use only rows with first_seen_at <= cutoff); None if the
    refresh failed, which makes The Scout skip the session."""

    status: str  # "ok" | "failed"
    n_fetched: int = 0
    n_inserted: int = 0
    cutoff: Optional[datetime] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def refresh_news(
    *,
    fetcher: NewsFetcher,
    store: NewsStore,
    universe: Sequence[str],
    now: datetime,
    refetch_days: int = LIVE_REFETCH_DAYS,
) -> NewsRefreshResult:
    """Fetch [now - refetch_days, now] (by vendor update time, see
    NewsFetcher) for the universe and insert new rows as ingest_mode "live"
    with first_seen_at = now (the same clock as the cutoff). Everything
    created in the feature lookback is updated within it, so the lookback is
    always complete. Never raises for a fetch/store failure: returns status
    "failed" so the daily job can keep running the other personas.
    """
    now = _aware_utc(now, "now")
    try:
        arts = fetcher.fetch(list(universe), now - timedelta(days=refetch_days), now)
        n_new = store.insert_new(arts, first_seen_at=now, ingest_mode="live")
    except Exception as exc:  # noqa: BLE001 - any failure -> Scout skips, others trade
        logger.warning("news.refresh_news failed: %s", exc, extra={"reason": "news_refresh_failed"})
        return NewsRefreshResult(status="failed", error=f"{type(exc).__name__}: {exc}")
    logger.info("news.refresh_news: fetched %d, inserted %d new", len(arts), n_new)
    return NewsRefreshResult(status="ok", n_fetched=len(arts), n_inserted=n_new, cutoff=now)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
