# WolfPack

Six AI trading personas, paper-trading real market data, arguing with and learning from each other — built and operated by an autonomous, multi-agent Claude Code pipeline.

**Built by [Nafis Uddin](https://github.com/nafisuddinn) — directed and reviewed, not hand-coded. See [how](PRD-and-backlog.md#5-architecture).**

**Status**: in progress, building in public. See the [build log](#) once deployed, or `PRD-and-backlog.md` for the live task list.

## What this is

Four independent AI traders (a trend-follower, a contrarian, a price-pattern model, and a news/sentiment model) trade paper money — zero real risk — and post every trade with a plain-language rationale. Two "meta" traders sit on top: **The Pack**, which blends the four based on performance and community trust, and **The Alpha**, which blends them based on who wins a structured, 3-round argument when they disagree.

This isn't a claim to real trading edge — see [Non-Goals](PRD-and-backlog.md#3-explicit-non-goals-v1) and every model card below. It's a demonstration of disciplined process: honest validation, honest documentation, and two live experiments that report real findings whether or not they're flattering.

## What actually makes this different

- **[`alphagate`](https://github.com/nafisuddinn/alphagate)** — a standalone, published package that decides whether a newly retrained model is actually good enough to replace the current one, blending statistical performance with community trust signal. Not trading-specific — usable in any pipeline that periodically retrains a model. WolfPack depends on the published package itself, not an inline copy.
- **The Howl** — a weekly, public prediction: does the crowd or the model call the leaderboard's winner correctly? Reported honestly, small sample size and all.
- **The Alpha Trial** — when personas disagree, they argue it out (3 rounds, 125 words each, can rebut or concede), and a judge persona forms its own view from the debate. Tests a real, currently-debated AI research question — does structured debate actually beat a simple vote — on this project's own small scale.
- **Model cards, updated as training happens** — [`MODEL_CARD.md`](MODEL_CARD.md) and [`MODEL_CARD_SCOUT.md`](MODEL_CARD_SCOUT.md) report benchmarks through the real pipeline (not just isolated backtests), rejected training runs and why, and one deliberate stress test each — modeled on [srock44's Hugging Face documentation discipline](https://huggingface.co/srock44).

## How it's built

Directed, not hand-coded: a 5-agent Claude Code pipeline (architect, coder, ML trainer, tester, reviewer) works from [`PRD-and-backlog.md`](PRD-and-backlog.md), reviewed and merged by me. Full engineering documentation, including every deliberate scope tradeoff, lives in the [Decision Log](PRD-and-backlog.md#8-decision-log).

Stack: Next.js, Python, XGBoost/LightGBM, MLflow, Supabase, Alpaca (paper trading), GitHub Actions. Entirely free-tier except Claude Pro and, if ever needed, one $10/mo fallback tool.

## Quant rigor, briefly

Log returns not raw price. Strictly chronological validation, no shuffled time-series splits. Deflated Sharpe Ratio, not raw Sharpe, correcting for the number of strategies compared. Volatility-targeted position sizing. A transaction-cost assumption applied before any performance metric. Full checklist and reasoning in `PRD-and-backlog.md` section 6d.

## Read more

- [`PRD-and-backlog.md`](PRD-and-backlog.md) — full spec, architecture, and live task list
- [`MODEL_CARD.md`](MODEL_CARD.md) / [`MODEL_CARD_SCOUT.md`](MODEL_CARD_SCOUT.md) — model documentation, updated continuously
- [`alphagate`](https://github.com/nafisuddinn/alphagate) — the standalone package
