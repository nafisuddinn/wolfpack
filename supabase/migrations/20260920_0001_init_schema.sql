-- WolfPack init schema
-- Tables: personas, prices, trades, trust_weights
-- Views: price_returns, feed_trades
-- RPC: set_trade_rationale (write-once rationale, SECURITY DEFINER)
--
-- All tables: RLS enabled, anon role gets SELECT only, service_role for writes.

-- ---------------------------------------------------------------------------
-- personas
-- ---------------------------------------------------------------------------
create table if not exists personas (
  id uuid primary key default gen_random_uuid(),
  slug text unique not null,
  name text not null,
  entity_kind text not null check (entity_kind in ('base', 'ensemble', 'judge')),
  strategy_type text not null check (
    strategy_type in (
      'trend_following',
      'mean_reversion',
      'ml_classifier',
      'ml_sentiment',
      'ensemble_trust_weighted',
      'ensemble_debate'
    )
  ),
  tagline text not null,
  description text,
  avatar_url text,
  is_active boolean not null default true,
  created_at timestamptz not null default now()
);

alter table personas enable row level security;

create policy "personas_select_anon" on personas
  for select
  to anon
  using (true);

create policy "personas_all_service_role" on personas
  for all
  to service_role
  using (true)
  with check (true);

-- ---------------------------------------------------------------------------
-- prices
-- ---------------------------------------------------------------------------
create table if not exists prices (
  ticker text not null,
  timeframe text not null default '1Day',
  ts timestamptz not null,
  open numeric(18, 6) not null,
  high numeric(18, 6) not null,
  low numeric(18, 6) not null,
  close numeric(18, 6) not null,
  volume bigint,
  source text not null default 'alpaca',
  ingested_at timestamptz not null default now(),
  primary key (ticker, timeframe, ts)
);

alter table prices enable row level security;

create policy "prices_select_anon" on prices
  for select
  to anon
  using (true);

create policy "prices_all_service_role" on prices
  for all
  to service_role
  using (true)
  with check (true);

create view price_returns
  with (security_invoker = on)
  as
  select
    ticker,
    timeframe,
    ts,
    close,
    ln(close / lag(close) over (partition by ticker, timeframe order by ts)) as log_return
  from prices;

-- ---------------------------------------------------------------------------
-- trades
-- ---------------------------------------------------------------------------
create table if not exists trades (
  id uuid primary key default gen_random_uuid(),
  persona_id uuid not null references personas(id),
  ticker text not null,
  side text not null check (side in ('buy', 'sell')),
  qty numeric(18, 6) not null check (qty > 0),
  signal_ts timestamptz not null,
  submitted_at timestamptz not null default now(),
  filled_at timestamptz,
  fill_price numeric(18, 6),
  status text not null default 'pending' check (status in ('pending', 'filled', 'rejected', 'canceled')),
  broker text not null default 'alpaca_paper' check (broker = 'alpaca_paper'),
  broker_order_id text unique,
  run_id text not null,
  signal_payload jsonb not null default '{}',
  rationale text,
  rationale_written_at timestamptz,
  rationale_author text,
  check (num_nonnulls(rationale, rationale_written_at) <> 1)
);

create index if not exists trades_persona_submitted_idx on trades (persona_id, submitted_at desc);
create index if not exists trades_submitted_idx on trades (submitted_at desc);
create index if not exists trades_missing_rationale_idx on trades (submitted_at) where rationale is null;

alter table trades enable row level security;

create policy "trades_select_anon" on trades
  for select
  to anon
  using (true);

create policy "trades_all_service_role" on trades
  for all
  to service_role
  using (true)
  with check (true);

create view feed_trades
  with (security_invoker = on)
  as
  select trades.*, personas.slug as persona_slug, personas.name as persona_name,
         personas.entity_kind as persona_entity_kind, personas.strategy_type as persona_strategy_type
  from trades
  join personas on personas.id = trades.persona_id
  where trades.rationale is not null
    and trades.status = 'filled';

-- ---------------------------------------------------------------------------
-- trust_weights
-- ---------------------------------------------------------------------------
create table if not exists trust_weights (
  id uuid primary key default gen_random_uuid(),
  persona_id uuid not null references personas(id),
  week_start date not null,
  statistical_score numeric,
  social_score numeric,
  weight numeric not null check (weight >= 0 and weight <= 1),
  method_version text not null default 'v0',
  inputs jsonb not null default '{}',
  computed_at timestamptz not null default now(),
  unique (persona_id, week_start, method_version)
);

alter table trust_weights enable row level security;

create policy "trust_weights_select_anon" on trust_weights
  for select
  to anon
  using (true);

create policy "trust_weights_all_service_role" on trust_weights
  for all
  to service_role
  using (true)
  with check (true);

-- ---------------------------------------------------------------------------
-- RPC: set_trade_rationale — write-once rationale setter.
-- This is the ONLY way the anon-keyed Claude step is allowed to write
-- rationale text; RLS denies all other anon writes to trades.
-- ---------------------------------------------------------------------------
create or replace function set_trade_rationale(p_trade_id uuid, p_rationale text)
returns void
language plpgsql
security definer
set search_path = public
as $$
begin
  update trades
  set
    rationale = p_rationale,
    rationale_written_at = now(),
    rationale_author = 'claude_code'
  where id = p_trade_id
    and rationale is null;
end;
$$;

revoke all on function set_trade_rationale(uuid, text) from public;
grant execute on function set_trade_rationale(uuid, text) to anon;
grant execute on function set_trade_rationale(uuid, text) to service_role;
