# WolfPack — Final PRD & Backlog

*A social feed of AI trading personas — including a self-governing ensemble that weighs both statistical performance and community trust — trading on paper accounts, built by an autonomous AI coding pipeline.*

Owner: Nafis Uddin · Status: In progress

---

## 1. Problem & Why

Quant firms make decisions under uncertainty using structured reasoning, validate models rigorously before trusting them, and explain that reasoning clearly. Most people never see any of that — it's opaque. WolfPack makes it visible: a small set of AI trading personas run distinct, explainable strategies on paper accounts, post every trade with a plain-language rationale, and are ranked by risk-adjusted return — not raw P&L.

On top of that sits the project's real differentiator: a fourth "meta" persona, **The Pack**, whose trust in each of the other personas is governed by an extension to this project's own model-governance package — one that blends statistical performance *and* live community signal, rather than statistics alone. That's the deliberate synthesis of the two things this project is inspired by: Jane Street's model-risk discipline, and Meta's social/engagement-driven ranking — combined at the architecture level, not just the surface level.

This is a personal project to learn a new domain (quantitative trading) from scratch, in public, using an AI-driven autonomous build process, while producing a real PM artifact trail (PRD, decision log, retro) alongside the code.

## 2. Goals (v1, 4 weeks)

- [ ] 4 base trading personas + 1 ensemble persona (The Pack), running automatically on paper accounts against real market data
- [ ] Every trade posts to a public feed with a plain-language rationale
- [ ] A leaderboard ranking personas by Sharpe ratio
- [ ] A standalone, published, project-specific model-governance package (`alphagate`) that WolfPack itself depends on
- [ ] The Howl: a weekly human-vs-machine forecasting comparison, with an honestly-reported running scoreboard
- [ ] (Stretch) The Alpha Trial: structured persona debate + a 5th judged/synthesized tracked entity
- [ ] A build log showing the autonomous dev process over time
- [ ] Deployed, working demo + short case-study write-up

## 3. Explicit Non-Goals (v1)

- No real money, ever. No investment advice framing anywhere in the product.
- No follow graph or comments — reactions/likes only (this is also the input signal for The Pack — see 6c).
- No more than 4 personas at launch.
- No custom market-data pipeline — use a hosted paper-trading API.
- No mobile app — responsive web only.
- No claim of real trading edge, anywhere, for any persona — stated explicitly in the write-up.

## 4. Success Metrics

- System uptime: personas trade on schedule without manual intervention for 7+ consecutive days
- Feed completeness: 100% of executed trades have a logged rationale
- Leaderboard correctness: Sharpe calculation verified against a manual spot-check
- `alphagate` correctness: promotion decisions verified against manual spot-checks on both statistical and blended-trust scenarios
- Demo reliability: live deploy survives a cold demo with no errors

## 5. Architecture

**Cost principle: everything free except Claude Pro (already owned) and OpenCode Go ($10/mo, accepted exception, Phase 2 only).**

- **Trading data & paper execution**: Alpaca Markets (free paper trading, real market data, no real money — no legal/regulatory exposure)
- **Mechanical trade execution**: Python worker, triggered by GitHub Actions cron (free, public repo) — runs each persona's strategy, places paper trades via Alpaca, writes the raw trade to the DB. Pure mechanical logic, no LLM call, no cost.
- **Rationale & sentiment generation**: a Claude Code task, not a live paid API call. Once daily, Claude Code (running headlessly under the Pro subscription) reads the day's raw trades and writes a first-person rationale per trade in that persona's voice, using feature importances (for The Analyst) and trust-weight changes (for The Pack) as its source material. $0 marginal cost — stays inside Pro-plan usage.
- **Database**: Postgres via Supabase (free tier; daily cron + Claude Code runs keep it from auto-pausing)
- **Frontend**: Next.js on Vercel Hobby (free, personal/non-commercial)
- **Autonomous build process**: Claude Code working headlessly from this backlog, one task at a time, opening a PR per task for review before merge
- **Overflow (Phase 2, not v1)**: OpenCode Go as a fallback coding agent if Claude Code's Pro usage limits are hit mid-sprint — the one accepted paid exception besides Claude Pro itself

## 5a. Complexity Principle

Keep individual trading strategies deliberately simple and explainable — legibility, not sophistication, is the point, and it avoids inviting questions about quant depth that would be dishonest to answer. Let complexity live in the *systems and governance* layer instead: unattended multi-day autonomous operation, a reviewed agent task loop, a real promotion-gate package, and — the one genuinely novel piece — a trust model that blends statistical and social signal. If a change adds trading sophistication without adding to that story, cut it.

## 6. The 5 Personas (v1)

| Persona | What it is | Signal source |
|---|---|---|
| The Trend Follower | Moving-average crossover | Price data only |
| The Contrarian | Mean-reversion, bounded by a threshold | Price data only |
| The Analyst | Gradient-boosted classifier (XGBoost/LightGBM) predicting next-day direction from ~10-15 engineered features | Price data, engineered features |
| **The Scout** | Gradient-boosted classifier trained on sentiment-derived features (news sentiment score, StockTwits bullish/bearish ratio, *change* in discussion volume — not raw levels, same stationarity principle as price returns) | News headlines (primary, reliable) + StockTwits public stream (secondary, best-effort — see 6a-ii) |
| **The Pack** | Ensemble that blends the base personas' signals, position-sized by each persona's current *trust weight* — governed by `alphagate` (see 6c) | Statistical performance (Sharpe) + community engagement on each persona's posts |

(The Alpha, from The Alpha Trial in section 6g, is a 6th tracked entity — see that section.)

## 6a-ii. The Scout's data sources — what's actually free right now

Checked directly rather than assumed, since this changes fast:

- **News headlines (primary)**: the same free-tier news API already used elsewhere in the project. Reliable, no approval gate.
- **StockTwits (secondary, best-effort)**: StockTwits' official developer registration is currently closed while they review their API and terms, but their public, unauthenticated stream endpoint (bullish/bearish-tagged messages per ticker) still works without a key as of this writing. Build this as a soft dependency — if the endpoint breaks or disappears, The Scout falls back to news-only rather than failing.
- **Reddit — explicitly not depended on**: Reddit now requires pre-approval for any API access (even personal, non-commercial use), with reported 2-4 week approval queues as of mid-2026. Apply for access in parallel if you want it eventually; don't gate any backlog item on it arriving.
- **Lookahead-bias risk specific to this persona, worth extra care**: a news article published *after* the close, summarizing why a stock moved that day, can leak the outcome into the model disguised as input if timestamp filtering isn't strict. `tester` now checks this explicitly (see the updated `tester.md`) — it's a sharper, easier-to-miss version of the same lookahead-bias discipline already required for price data.

## 6a. The ML Layer ("The Analyst")

- Model: XGBoost/LightGBM binary classifier, CPU-only, trains in seconds, on free Alpaca price data
- Retraining: weekly, via the same free GitHub Actions cron
- Experiment tracking: MLflow in local file-store mode — commits into the repo, $0, no server
- Explainability: feature importances feed Claude Code's daily rationale task
- Explicit non-claim: no meaningful real-world trading edge on daily price data — the point is disciplined process, not alpha

## 6a-i. Model documentation discipline (inspired by Sean Rockwitz's Hugging Face model cards)

The transferable lesson from srock44's model cards isn't the LLM fine-tuning tooling (LoRA/Unsloth don't apply to a gradient-boosted tree model) — it's the evaluation and documentation rigor. Four concrete practices, ported directly:

1. **Report metrics through the real pipeline, not just in isolation.** Show directional accuracy *and* DSR as actually realized through the full pipeline (vol-targeted sizing, transaction costs applied) — these numbers will differ from a clean backtest, and reporting both honestly (not just the flattering one) is the point.
2. **Log rejected challengers, not just the promoted champion.** `alphagate`'s weekly comparison already produces this data — the addition is writing down *why* a challenger lost, not discarding it. Nearly free, since the comparison already happens.
3. **One honest stress-test fixture.** Run The Analyst against one specific historical high-volatility window (a real flash-crash or crisis period) and report the result plainly, whether or not it's flattering.
4. **Name the specific failure mode, not a generic disclaimer.** Once training is underway, replace "no real edge" with the actual observed failure pattern (e.g., degraded accuracy in specific market conditions) — precise and honest beats vague and honest.

Maintained in `MODEL_CARD.md` (template provided), updated as training progresses — not written once at the end.

## 6d. Quant rigor pass — the "not basic finance-guy knowledge" checklist

Not a claim to institutional-level quant research — an honest claim to knowing where the naive version of this project breaks, and fixing those specific spots. Five additions, all standard real quant practice:

1. **Stationary features**: model inputs are log returns, not raw prices. Raw price is non-stationary and breaks generalization the moment price levels shift.
2. **No lookahead bias**: strict chronological walk-forward validation — train only on the past, test strictly forward in time, never randomly shuffled. *Documented limitation*: a fully rigorous version would add purging/embargoing around overlapping label windows (per Bailey & López de Prado, "Advances in Financial Machine Learning") — noted as Phase 2 given the timeline, not silently skipped.
3. **Deflated Sharpe Ratio on the leaderboard** (Bailey & López de Prado), not raw Sharpe. Corrects for selection bias from testing 4 personas at once — the "best" persona's raw Sharpe is partly inflated by the fact that several strategies were tried. This is the single highest-signal addition in the project.
4. **Volatility-targeted position sizing**: position size scales inversely with the traded asset's recent volatility (basic risk parity), not a fixed dollar amount per trade.
5. **Transaction cost assumption**: a flat basis-point cost applied to every trade's realized return before any Sharpe/DSR calculation — avoids the frictionless-market fantasy that undermines an otherwise-careful backtest.

Effort estimate: ~5-7 hours total, mostly in Week 2-3. Funded by treating Week 4 visual polish as the first thing to cut if time runs short — this checklist is higher priority than making the UI prettier.

## 6b. `alphagate` — the standalone artifact

Champion/challenger promotion logic, written from day one as a small, project-specific, dependency-light Python package — not app-embedded — that WolfPack itself depends on (published, not copied inline).

- Core function: given a champion, a challenger, a metric, and holdout data, returns a promote/reject decision and logs it. Prevents a worse model from silently replacing a better one.
- Used to gate The Analyst's weekly retrains.
- This is the "adoptable engineering" answer: a small, honest version of a pattern real quant/ML teams already use before deploying a new model.

## 6c. The genuinely novel piece: trust-weighted governance (The Pack + `alphagate` extension)

This is the actual synthesis of "Jane Street" and "Meta" at the architecture level, not just in the UI:

- Each of the 3 base personas has a **trust weight**, recalculated weekly by `alphagate`, combining two inputs:
  1. **Statistical signal** — trailing Sharpe ratio (the quant/risk-control input)
  2. **Social signal** — engagement on that persona's recent trade posts (reactions, and the ratio of positive to skeptical/flagging reactions — the Meta-style input)
- The Pack persona sizes its own paper positions as a weighted blend of the base personas' signals, using these trust weights.
- Why this matters: a persona's statistics can look fine for a while even as its calls start drawing visible skepticism from anyone reading its rationale — the blended signal can catch a qualitative red flag before it shows up in a lagging statistical metric. That's a real (if small-scale) version of "human-in-the-loop signal alongside automated validation," which is a genuinely defensible idea in ML governance generally, not just in trading.
- `alphagate` is written to support this as a general capability: pluggable trust-signal sources, not just a single statistical metric — this is the specific piece of engineering that isn't "another Alpaca bot," because most social-trading demos treat the social layer as cosmetic. Here it's a functional input to a governance decision.
- Every weight change gets written up by Claude Code in plain language (e.g. "Pack reduced trust in The Analyst this week — Sharpe held steady, but its last three calls drew unusually skeptical reactions") — this is also where the feed and the governance layer visibly connect for a reader.

## 6f. The Howl — human vs. machine forecasting (the standout feature)

Not more ML sophistication — a live, ongoing, honestly-reported comparison of crowd forecasting against The Pack's own model-driven forecasting. This is the piece that isn't a commodity: training a model on price data is done everywhere; running a real, public experiment on whether humans or the model call it better is not.

- **Mechanic**: every Monday, visitors are asked one question — "Which persona leads the leaderboard by DSR this Friday?" One vote per visitor (session-based, no auth needed), four options.
- **The Pack's own forecast is free**: whichever persona currently holds the highest trust weight *is* The Pack's implied prediction for the week — no new model required, just reading data that already exists.
- **Friday**: compare the actual DSR leader against (a) the crowd's majority vote and (b) The Pack's implied pick. Log the result. Running scoreboard, week over week.
- **Honesty requirement, stated plainly in the write-up**: sample size (both weeks and voters) will be small for a solo project — this is illustrative, not statistically rigorous, and that caveat is said outright, not buried. Same documentation discipline as `MODEL_CARD.md`.
- **Why this is the differentiator**: it produces a genuine finding you don't already know the answer to, sits at the intersection of behavioral finance (crowd wisdom vs. model forecasting) and Jane Street's actual business (they are major prediction-market participants), and is the one place the "social" layer does something functional rather than decorative.
- **Scope**: one new table (predictions: session id, week, persona picked, timestamp), a lightweight voting widget, a weekly comparison job, and a results write-up. No new paid service. ~4-6 hours — funded the same way as everything else: visual polish is the first thing to cut if time is short, this is not.

## 6e. Optional artifact: OCaml port of the promotion-gate core logic

Scoped small and separate from the main build — not integrated into the running app, not on the critical path. Jane Street's primary development language is OCaml; a small, standalone, well-commented reimplementation of `alphagate`'s core promote/reject decision logic in OCaml is a deliberate, honest signal of having done real homework about the firm specifically.

- **Scope**: the pure decision logic only (champion, challenger, metric, holdout data → promote/reject), not a full port of the app or the ML training pipeline.
- **Timing**: Week 4 stretch goal or explicitly Phase 2 — never let this compete with core-scope work.
- **Non-negotiable condition**: Claude Code must explain the code as it's written, not just deliver a finished file — the goal is being able to walk through it yourself in an interview, not having a file you can't discuss. If you can't explain it, it's a liability, not an asset — don't ship it in that state.
- **Why OCaml specifically here**: its type system can make an invalid decision state unrepresentable at compile time, which is a genuine, defensible reason to have picked it for this piece, not just a language flex.

## 6g. The Alpha Trial — structured debate + judgment (Week 4 stretch)

Grounded in real research, not just a feature idea: a well-studied line of AI work has multiple language model agents debate their reasoning over several rounds to reach a better answer than any one agent alone — but a 2025 study specifically found that plain majority voting accounts for most of debate's apparent benefit, meaning whether debate itself adds real value beyond simple voting is a genuinely open, unsettled question. This project gets to test that question honestly, at its own small scale, alongside The Howl's crowd-vs-model comparison.

- **Mechanic**: when the base personas disagree on a call, it becomes an Alpha Trial — a public thread where each involved persona posts up to 3 comments, 125 words max each. Each comment can rebut, reinforce, or concede a point — genuine agreement is a valid, informative outcome, not just argument for argument's sake. Hard limits, enforced in generation.
- **The Alpha**: a new entity that reads the full argument and forms its own trade — not simply picking the "winning" persona, but weighing synthesis across sides, and explicitly treating convergence (personas ending up in agreement) as its own signal, not just declaring a winner when none is needed. Tracked as a 6th independent paper-trading entity: its own trades, its own leaderboard position, same DSR/vol-sizing/transaction-cost rigor as every other tracked entity.
- **Deliberate safety boundary (Decision Log entry required)**: base personas' strategy logic never self-modifies based on debate outcomes — that would be a real risk-control problem and would break their legibility. The Alpha is where "learning from the argument" lives; Trend Follower, Contrarian, and The Analyst stay exactly as simple and explainable as before.
- **Not the same mechanism as The Pack**: The Pack blends by statistics + community trust; The Alpha blends by argument quality. Two different ensemble philosophies running side by side is the point, not redundancy — and it's what makes "does structured debate actually beat a simpler blend" a real, honestly-reportable question for this project specifically.
- **Not gated by `alphagate`**: unlike The Analyst's weekly retrain, there's no old-model-vs-new-model to compare each week — The Alpha makes a fresh judgment call every trial. Its resulting trades still get tracked with full rigor; it just doesn't go through the champion/challenger promotion process.
- **Clarity requirement**: the leaderboard must make it easy, at a glance, to see all 6 tracked entities' current standing and understand what each one is (a one-line description per entity — e.g. "Pack — blends by performance + community trust," "Alpha — blends by argument quality") — not just a bare numbers table.
- **Scope**: ~6-10 hours, larger than The Howl. Week 4 stretch goal, not core-path. If time is short, the fallback is 1 debate round instead of 3, not cutting the feature entirely.
- **Extending The Howl**: once The Alpha exists, add it as a 6th prediction option — no schema change needed, just a new row. If it doesn't ship in time, The Howl launches with 5 options and this is a clean Phase 2 addition.

## 7. Risk Register

- **Regulatory/legal**: paper trading only, no real funds, clear "not investment advice / educational" disclosure
- **Data licensing**: free-tier APIs only, within terms of service
- **Rate limits**: worker backs off gracefully on Alpaca/API limits, doesn't crash
- **Single point of failure**: scheduled worker needs a basic health check so a silent failure doesn't go unnoticed
- **Model risk**: no persona has meaningful real-world predictive edge — stated plainly in the write-up. `alphagate`'s gate mitigates silent degradation and now blended-signal drift, not lack of edge.
- **Social-signal risk**: engagement-based trust weighting could be gamed by a burst of reactions with no real information content — worth naming as a known limitation, not solving in v1 (a real product would need bot/abuse detection; out of scope here, and saying so is itself the right move).
- **Third-party data dependency risk (The Scout)**: StockTwits' public stream is unofficial/undocumented and could change or disappear without notice — The Scout is built to degrade gracefully to news-only rather than fail. Reddit access is not depended on given its new pre-approval requirement and multi-week queue.

## 8. Decision Log

*(Add an entry every real tradeoff — timestamp, decision, why, what was traded off. Start in Week 1.)*

| Date | Decision | Why | Traded off |
|---|---|---|---|
| 2026-09-19 | Split each cron workflow (`daily-trades.yml`, `weekly-retrain.yml`) into a mechanical, non-LLM step plus a narrowly-scoped Claude step used only for prose (rationale/write-up), each with least-privilege secrets (Claude step gets only `SUPABASE_URL`/`SUPABASE_ANON_KEY`, never Alpaca or News API creds; Claude billed via `CLAUDE_CODE_OAUTH_TOKEN`/Pro subscription, never `ANTHROPIC_API_KEY`) | PRD section 5 specifies mechanical trade execution must be pure Python at $0 LLM cost; the previous single Claude-does-everything step also silently no-op'd if the `claude` CLI was missing, since `subprocess.run()` had no error checking | Two steps per workflow instead of one — more workflow YAML and a bit of duplicated wiring, in exchange for cost control, credential scoping, and honest failure |
| 2026-09-19 | Changed `.gitignore` from a bare `.claude/` line to `.claude/*` plus `!.claude/settings.json` | The bare `.claude/` ignore (added when scrubbing the private manager/subagent-delegation instructions from git history) accidentally also swept out `settings.json`, which holds the PreToolUse hook that blocks any reference to Alpaca's live trading endpoint — a non-negotiable safety backstop that must stay in the public repo and in CI | A directory-level ignore can no longer be a single line; subagent definitions and other local `.claude/` files stay private as intended, but the ignore rule is now two lines and depends on remembering to negate future files that should be public |
| 2026-09-19 | Created a new public `.github/RUNBOOK.md` containing only the operational procedures the cron jobs need (daily rationale writing, weekly retrain/alphagate write-up, non-negotiables), instead of pointing CI at `CLAUDE.md` | `CLAUDE.md` (with the manager/subagent-delegation philosophy) was deliberately removed from git history and is no longer present on any CI checkout, but both workflows still told Claude to "Read CLAUDE.md" | Some duplication between `RUNBOOK.md` and `CLAUDE.md` (the non-negotiables appear in both); the manager/subagent-delegation philosophy stays private and is not visible to the CI-run Claude step |
| 2026-09-19 | Mechanical steps in both workflows use honest no-op `# TODO`/`echo` placeholders instead of stub scripts or fake success, since the worker and retrain scripts don't exist yet (separate, not-yet-done backlog items) | Matches the project rule to state limitations plainly rather than implying working functionality that hasn't been built | The cron jobs currently do nothing useful end-to-end until the real scripts land, and that is visible in the workflow output rather than hidden |
| 2026-09-19 | Added a deterministic "Ensure commit is pushed" step after each Claude step in both workflows, rather than trusting the Claude step to remember to push | A commit made but not pushed (e.g. Claude's turn budget running out mid-task) would otherwise fail silently and the job would still report green, losing the day's rationale or retrain write-up with no signal | A small amount of duplicated verification logic across both workflow files, in exchange for eliminating a silent-data-loss failure mode |

## 9. Backlog

### Week 1 — Foundations
- [ ] Set up Alpaca paper trading account, verify API access with a test order
- [ ] Scaffold repo: Next.js app, Python worker, Supabase Postgres schema (trades, personas, prices, trust_weights tables)
- [ ] Architect `alphagate`'s core promotion logic as a standalone, generic module from day one — champion, challenger, metric fn, holdout data in; promote/reject + log out
- [ ] Confirm Claude Code headless run works end-to-end on one trivial task
- [ ] Implement Persona #1 (Trend Follower): strategy logic, places paper trade, logs raw trade to DB
- [ ] Add Claude Code's daily rationale-writing task; confirm it runs entirely on Pro-plan usage, no separate API billing
- [ ] Draft one-pager + this PRD (mark v1)

### Week 2 — More personas + feed
- [ ] Implement Persona #2 (Contrarian): strategy logic + raw trade logging
- [ ] Implement Persona #3 (The Analyst): feature engineering using log returns (stationary, not raw price), initial XGBoost model with strict chronological train/test split, MLflow local tracking, wire mechanical execution to model output
- [ ] Create `MODEL_CARD.md` from the template — fill in as training happens, not at the end
- [ ] Implement Persona #4 (The Scout): news-sentiment ingestion (primary), StockTwits public-stream ingestion with graceful fallback (secondary), sentiment/volume-change feature engineering (not raw levels), strict timestamp filtering to prevent news-based lookahead bias, chronological train/test split
- [ ] Create `MODEL_CARD_SCOUT.md` from the same template — fill in as training happens
- [ ] Build feed UI: chronological trade posts with rationale text, reactions (like/skeptical)
- [ ] Build persona profile pages: avatar, name, running P&L, trade history
- [ ] Start Decision Log — backfill Week 1, keep it live going forward

### Week 3 — Governance layer, ensemble, leaderboard, deploy
- [ ] Implement Sharpe ratio calculation per persona, spot-check manually
- [ ] Implement Deflated Sharpe Ratio (Bailey & López de Prado) accounting for the full multi-entity trial count (now larger with The Scout, The Pack, and The Alpha added — more strategies compared means the selection-bias correction matters more, not less); leaderboard ranks by DSR, not raw Sharpe
- [ ] Implement volatility-targeted position sizing (risk-parity style) across all personas
- [ ] Apply a flat basis-point transaction cost assumption to realized returns before any Sharpe/DSR calculation
- [ ] Build leaderboard page, ranked by DSR
- [ ] Finish generalizing `alphagate`: pluggable trust-signal sources (statistical + social), tests, README, publish to PyPI
- [ ] Wire WolfPack's app code to depend on the *published* `alphagate` package (dogfooding, not inline copy)
- [ ] Set up weekly retrain + `alphagate` promotion gate for The Analyst
- [ ] Extend `alphagate` to log rejected challengers with reason, not just the promoted champion
- [ ] Run and document one stress-test fixture: The Analyst's behavior on a specific historical high-volatility window, reported honestly in `MODEL_CARD.md`
- [ ] Implement trust-weight calculation (Sharpe + reaction ratio) and The Pack persona's blended position sizing
- [ ] Wire feature importances (Analyst) and trust-weight changes (Pack) into Claude Code's daily rationale task
- [ ] Build build-log page rendering Claude Code's commit/task history over time
- [ ] Build The Howl: predictions table, weekly voting widget, one vote per session
- [ ] Wire The Pack's current trust weights to auto-generate its implied weekly forecast (no new model — reads existing data)
- [ ] Deploy: frontend to Vercel, worker + cron to GitHub Actions, confirm unattended runs for 48+ hours

### Week 4 — Polish + write-up (front-load this, don't leave it to the last 2 days)
- [ ] (Stretch, only if Weeks 1-3 landed on schedule) Build The Alpha Trial: debate thread UI, constrained comment generation (3 rounds, 125 words), The Alpha's judgment/synthesis logic, 6th tracked entity wired into the leaderboard and DSR pipeline
- [ ] (Stretch) Add The Alpha as a 5th option in The Howl
- [ ] Bug bash: full cold demo run, fix anything that breaks
- [ ] Visual polish pass (treat as nice-to-have if time is short — protect Week 3 governance work first)
- [ ] Write Risk Register final version, including the model-risk and social-signal-risk notes stated plainly
- [ ] Write Retro: what worked, what changed, what's explicitly Phase 2
- [ ] Publish The Howl's running results honestly, including the small-sample-size caveat stated plainly, not buried
- [ ] Write the case-study document (1-2 pages): problem, approach, what was learned about quant trading from scratch, key decisions and why, what's next — explicitly separate "the feed is the demo" from "`alphagate`'s trust-weighted governance is the reusable/novel piece" and "The Howl is the genuine finding," and state the model-risk non-claim plainly
- [ ] Final PRD update marking shipped scope vs. original scope, with reasons for any cuts

## 10. Phase 2 (explicitly out of scope for v1)

- OpenCode failover: automatic handoff from Claude Code to OpenCode when Pro plan limits are hit
- Expand beyond 4 personas
- Real follow graph + comments
- Abuse/gaming detection for the social-signal input to `alphagate`
- "Copy trade" simulation for a visiting user
- Purged/embargoed cross-validation (full rigor beyond simple walk-forward)
- Local GPU-based sequence models (RTX 3060) as a "what more compute would unlock" exploration — not required, GBTs don't need a GPU

## 11. Operations — keeping it running unattended

Two separate concerns, don't conflate them:

- **The production loop (trade execution, weekly retrain) must never depend on a laptop being on.** It runs entirely on GitHub Actions' free cloud runners, on a schedule — this works regardless of whether any local machine is open, asleep, or off. No phone, no Remote Control, no extra setup needed for this to keep running.
- **Claude Code Remote Control** (research preview) is a separate, optional convenience: it lets the Claude mobile app monitor/steer a Claude Code session that is actively running *locally* — execution stays on the machine, the phone is a window into it. This requires the local machine to stay on and awake for the duration. Useful for checking in on an active dev session away from the desk; not a substitute for the cloud-scheduled production loop.

