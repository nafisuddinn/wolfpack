-- WolfPack seed data — the 5 v1 personas (PRD section 6).
-- Idempotent/rerunnable: upserts on the unique `slug` column.

insert into personas (slug, name, entity_kind, strategy_type, tagline, description)
values
  (
    'trend-follower',
    'The Trend Follower',
    'base',
    'trend_following',
    'Moving-average crossover.',
    'Signal source: price data only. Follows the direction of established price trends using a moving-average crossover rule.'
  ),
  (
    'contrarian',
    'The Contrarian',
    'base',
    'mean_reversion',
    'Mean-reversion, bounded by a threshold.',
    'Signal source: price data only. Bets on price reverting toward its recent mean once a deviation threshold is crossed.'
  ),
  (
    'the-analyst',
    'The Analyst',
    'base',
    'ml_classifier',
    'Gradient-boosted classifier predicting next-day direction.',
    'Signal source: price data, engineered features. XGBoost/LightGBM binary classifier trained on ~10-15 engineered price features (log returns, not raw price).'
  ),
  (
    'the-scout',
    'The Scout',
    'base',
    'ml_sentiment',
    'Gradient-boosted classifier trained on sentiment-derived features.',
    'Signal source: news headlines (primary) plus StockTwits public stream (secondary, best-effort). Features are sentiment score, bullish/bearish ratio, and change in discussion volume.'
  ),
  (
    'the-pack',
    'The Pack',
    'ensemble',
    'ensemble_trust_weighted',
    'Blends the base personas'' signals, position-sized by trust weight.',
    'Signal source: statistical performance (Sharpe) plus community engagement on each base persona''s posts, governed by alphagate.'
  )
on conflict (slug) do update set
  name = excluded.name,
  entity_kind = excluded.entity_kind,
  strategy_type = excluded.strategy_type,
  tagline = excluded.tagline,
  description = excluded.description;
