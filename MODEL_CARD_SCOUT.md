# The Scout — Model Card

*Maintained by Nafis Uddin, as part of [WolfPack](https://github.com/nafisuddinn/wolfpack).*

*Update this as training happens, not once at the end. Modeled on the documentation discipline in srock44's Hugging Face model cards (https://huggingface.co/srock44) — report what actually happened, including the parts that don't flatter the model.*

## Why this exists

The Scout is WolfPack's ML-driven persona: a gradient-boosted classifier predicting next-day price direction from engineered features. It exists to demonstrate a disciplined, honest ML development process — not to claim real trading edge. See the non-claim in `PRD-and-backlog.md` section 6a.

## What's in this repo

- `train_scout.py` — the exact training script
- `features.py` — feature engineering (log returns, volatility, RSI, MA spread, volume change)
- `eval_scout.py` — standalone benchmark harness, reproduces the table below
- MLflow tracking data (local file-store, committed) — full experiment history

Reproduce with:
```bash
python eval_scout.py --model <mlflow-run-id>
```

## Benchmark

*Fill in as each version trains. Report both the isolated backtest number and the number as realized through the full pipeline — they will differ.*

| Version | Directional accuracy | DSR (isolated backtest) | DSR (through full pipeline: vol-sizing + txn costs) | Max drawdown | Promoted by `alphagate`? |
|---|---|---|---|---|---|
| v1 | | | | | |

## Stress test

*One specific historical high-volatility window. Report the result plainly — this is meant to be an honest check, not a flattering one.*

- **Window tested**: [e.g. a specific historical flash-crash or high-volatility date range]
- **Result**: [what actually happened — degraded accuracy, wider drawdown, model behavior under stress]
- **Interpretation**: [what this does and doesn't tell you about the model]

## Training history

*Pass-by-pass, including retrains that made things worse. Log rejected `alphagate` challengers here too, not just the promoted champion.*

| Date | Change | Result | Promoted? | Why |
|---|---|---|---|---|
| | Initial training | | | |

## Known limitations

*Name the specific observed failure mode once you have one — not a generic disclaimer.*

- No meaningful real-world predictive edge on public daily price data — this is a process demonstration, not an alpha claim.
- [Add the specific failure pattern once observed, e.g. "accuracy degrades during low-volume sessions"]

## License / reuse

Training code, eval harness, and MLflow tracking data are all in this repo — nothing depends on anything unreleased.

## Data source notes (specific to The Scout)

- **News headlines**: primary, reliable free-tier source.
- **StockTwits public stream**: secondary, best-effort — unofficial/undocumented endpoint, no key required as of this writing, but not guaranteed stable. If it breaks, The Scout should fall back to news-only rather than fail.
- **Reddit**: not used — Reddit's 2026 Responsible Builder Policy requires pre-approval with multi-week queues; not depended on for this project.
