-- The Scout (Persona #4): news headlines from Alpaca's news API (Benzinga),
-- fetched with alpaca-py's NewsClient. Design: docs/design/scout-design.md
-- section 5.
--
-- PRIVATE TABLE. Headline text and URLs are stored here for scoring only and
-- must never reach the anon role, the public feed, `signal_payload`, or any
-- rationale text: Benzinga's redistribution terms are unclear and the repo
-- and feed are public (Decision Log 2026-10-04). So, unlike every other
-- table, there is NO anon select policy, and table privileges are revoked
-- from anon/authenticated as well (belt and braces on top of RLS).
--
-- Point-in-time columns:
--   created_at     vendor publish time. The Scout's feature window for the
--                  session that closes at close_t is (close_{t-1}, close_t]
--                  on this column; it is the ONLY timestamp backtests use.
--   first_seen_at  when WolfPack first stored the row. Live features also
--                  require first_seen_at <= the run's news cutoff, so an
--                  article that arrives late is never used retroactively.
--                  The worker passes it explicitly (same clock as the
--                  cutoff); the default is only for manual inserts.
--   vendor_updated_at  vendor's updated_at. updated_at > created_at means the
--                  article was revised after publication; the stored headline
--                  may be the revised one. Measured and reported, not fixable.
--
-- Rows are insert-once: the worker inserts with ON CONFLICT DO NOTHING, and
-- the trigger below rejects every UPDATE, so first_seen_at (and the headline
-- first stored) can never be overwritten by a later fetch.

create table if not exists news_articles (
  id bigint primary key,
  created_at timestamptz not null,
  vendor_updated_at timestamptz,
  headline text not null,
  source text not null,
  url text,
  symbols text[] not null default '{}',
  first_seen_at timestamptz not null default now(),
  ingest_mode text not null check (ingest_mode in ('backfill', 'live'))
);

create index if not exists news_articles_symbols_gin on news_articles using gin (symbols);
create index if not exists news_articles_created_at_idx on news_articles (created_at);

alter table news_articles enable row level security;

create policy "news_articles_all_service_role" on news_articles
  for all
  to service_role
  using (true)
  with check (true);

revoke all on table news_articles from anon, authenticated;

create or replace function news_articles_reject_update()
returns trigger
language plpgsql
set search_path = public
as $$
begin
  raise exception 'news_articles rows are insert-once (first_seen_at must never be overwritten)';
end;
$$;

create trigger news_articles_no_update
  before update on news_articles
  for each row execute function news_articles_reject_update();
