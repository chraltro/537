"""Measure the spreads `model/scale.py` freezes, and print what changed.

The rating transform divides a club's distance from average by a fixed spread.
Fixed, because a spread recomputed each build would let one club's summer move
every other club's rating, and a season where the title race tightened would
silently inflate everybody.

Fixed does not mean guessed. Each constant in `scale.py` was measured, once,
by this script, over the corpus as it stood. Run it again when the corpus grows
enough to matter; it prints the current constant beside the fresh measurement so
the size of the drift is visible before anything is edited.

    python -m tools.measure_scale

It writes nothing. Editing `scale.py` is a deliberate act, because every rating
on the site moves when those numbers do.
"""
from __future__ import annotations

import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import europe, rankings, scale  # noqa: E402
from model.parse import TeamRegistry  # noqa: E402

#: Which measure feeds which dimension, and whether the spread is of the value
#: or of its logarithm. Ratios are logged, differences are not, exactly as
#: `scale.dimension` does it.
MEASURES = (
    ("home", "home_edge", False),
    ("big", "big_edge", False),
    ("consistency", "gd_sd", False),
)

SHOT = (
    ("finishing", "conversion", True),
    ("creation", "sot_pm", True),
    ("discipline", "foul_index", False),
)


def spread(values: list[float], log: bool) -> float | None:
    import math
    xs = [v for v in values if v is not None]
    if log:
        xs = [math.log(v) for v in xs if v > 0]
    return statistics.pstdev(xs) if len(xs) > 20 else None


def main() -> None:
    print("Loading the pooled corpus…")
    corpus = europe.Corpus(TeamRegistry()).load(quiet=True)
    payload = rankings.build(corpus, quiet=True)
    prof = payload.get("_profile")
    if prof is None:
        # `build` does not publish the raw measures, so recompute them here from
        # the same matches it used. Cheaper than changing its return shape.
        import datetime as dt
        ref = max(dt.date.today(), dt.date(2026, 8, 1))
        hist = corpus.before(ref)
        pool = sorted({m.home for m in hist} | {m.away for m in hist})
        from model import ratings as R
        fit = R.fit_pooled(hist, pool, ref, group_of=corpus.group_of,
                           club_league=corpus.club_leagues(),
                           default_group=europe.EUROPE)
        recent: dict[str, list] = {}
        for m in hist:
            for t in (m.home, m.away):
                recent.setdefault(t, []).append(m)
        prof = {t: rankings._profile(ms, t, fit) for t, ms in recent.items()}

    print(f"\n{len(prof)} clubs profiled\n")
    print(f"{'dimension':<14}{'measured':>10}{'in scale.py':>14}{'drift':>10}")
    print("-" * 48)
    for name, key, log in MEASURES:
        vals = [r[key] for r in prof.values() if r.get(key) is not None]
        got = spread(vals, log)
        cur = scale.EUROPE_SD.get(name)
        drift = f"{(got - cur) / cur * 100:+.0f}%" if (got and cur) else ""
        print(f"{name:<14}{'-' if got is None else round(got, 4):>10}"
              f"{'-' if cur is None else cur:>14}{drift:>10}"
              f"   ({len(vals)} clubs)")

    print("\nShot-based dimensions come from the five leagues with a shot feed,")
    print("so their population is the big five and their reference is its mean.")
    import json
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "site", "data", "shooting.json")
    try:
        sh = json.load(open(path))
    except OSError:
        print("  (site/data/shooting.json not built; run `python -m model.run` first)")
        return
    clubs = sh.get("clubs") or {}
    for name, key, log in SHOT:
        vals = [c.get(key) for c in clubs.values() if isinstance(c, dict)]
        got = spread(vals, log)
        cur = scale.EUROPE_SD.get(name)
        drift = f"{(got - cur) / cur * 100:+.0f}%" if (got and cur) else ""
        n = len([v for v in vals if v is not None])
        print(f"{name:<14}{'-' if got is None else round(got, 4):>10}"
              f"{'-' if cur is None else cur:>14}{drift:>10}   ({n} clubs)")


if __name__ == "__main__":
    main()
