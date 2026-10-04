"""Independent brute-force oracle for scout_v1 (written by the tester, not the implementer).

Re-derives every feature from the spec with plain loops (window = (close_{i-1}, close_i] found by a linear
scan, no searchsorted/bincount/cumulative sums) on a calendar that spans the March DST switch and two
early-close days, with articles placed at, one second either side of, and far from the closes. Also checks
that rewriting every article published after close_t cannot change any feature row up to t.
"""

from __future__ import annotations

import math
import random
from datetime import timedelta

import numpy as np
import pandas as pd
from scout_helpers import UNIVERSE, articles, make_sessions

from wolfpack_worker.scout import features as F

DAYS = [d.date() for d in pd.bdate_range("2026-02-02", periods=80)]  # crosses 2026-03-08 DST start
EARLY = [DAYS[35], DAYS[55]]
SESS = make_sessions(DAYS, early_close=EARLY)
CLOSES = [s.close for s in SESS]


def _arts(seed: int = 7, n: int = 900):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        base = SESS[rng.randrange(len(SESS))].close
        dt = rng.choice([0, 1, -1, 3600, -3600, -86400, 2 * 86400, rng.randint(-50000, 50000)])
        rows.append((i, base + timedelta(seconds=dt), tuple(rng.sample(UNIVERSE, rng.randint(1, 3))), None,
                     rng.uniform(-1, 1)))
    return articles(rows)


def _window(t):
    for i in range(1, len(CLOSES)):
        if CLOSES[i - 1] < t <= CLOSES[i]:
            return i
    return -1


def test_features_match_a_brute_force_reference_across_dst_and_early_closes():
    a = _arts()
    a = a.assign(w=[_window(t) for t in a["created_at"]])
    got = F.build_scout_features(a.drop(columns="w"), SESS, UNIVERSE)

    def ns(tk, i):
        m = a[(a.w == i) & a.symbols.map(lambda s: tk in s)]
        return len(m), float(m.score.sum())

    def cw(tk, lo, hi):
        n = sum(ns(tk, j)[0] for j in range(lo, hi + 1))
        s = sum(ns(tk, j)[1] for j in range(lo, hi + 1))
        return s / n if n else 0.0

    for tk in UNIVERSE:
        for t in range(25, len(SESS)):
            n0, s0 = ns(tk, t)
            mkt = a[a.w == t]  # every article here carries at least one universe symbol
            base = np.mean([math.log1p(ns(tk, j)[0]) for j in range(t - 20, t)])
            s5 = cw(tk, t - 4, t)
            ref = [s0 / n0 if n0 else 0.0, float(n0 > 0), math.log1p(n0) - base, s5, s5 - cw(tk, t - 24, t - 5),
                   float(mkt.score.mean()) if len(mkt) else 0.0]
            np.testing.assert_allclose(got[tk].iloc[t].to_numpy(), ref, atol=1e-12, err_msg=f"{tk} session {t}")


def test_rewriting_everything_after_close_t_never_changes_rows_up_to_t():
    a = _arts()
    base = F.build_scout_features(a, SESS, UNIVERSE)
    for t in (30, 36, 56):
        late = a["created_at"] > CLOSES[t]
        b = a.copy()
        b.loc[late, "score"] = 0.99
        b.loc[late, "created_at"] = b.loc[late, "created_at"] + timedelta(minutes=7)
        alt = F.build_scout_features(b.sort_values(["created_at", "id"]), SESS, UNIVERSE)
        for tk in UNIVERSE:
            np.testing.assert_array_equal(alt[tk].iloc[: t + 1].to_numpy(), base[tk].iloc[: t + 1].to_numpy())
