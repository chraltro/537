"""Measure the spreads `model/scale.py` freezes, and print what changed.

The rating transform divides a club's distance from average by a fixed spread.
Fixed, because a spread recomputed each build would let one club's summer move
every other club's rating, and a season where the title race tightened would
silently inflate everybody.

Fixed does not mean guessed. Each constant in `scale.py` was measured, once, by
this script, over the corpus as it stood. Run it again when the corpus grows
enough to matter; it prints the current constant beside the fresh measurement so
the size of the drift is visible before anything is edited.

It also prints the reliability of each measure: what share of the spread between
clubs is real, as opposed to the luck of the matches we happened to see. That
column is why two dimensions this site used to publish no longer exist. Home
advantage came out at 7% real and big games at 4%, and both had been drawn as
axes of a radar out of 100, which is the most confident way there is to publish
a random number. Nothing below half gets published.

    python -m tools.measure_scale

It writes nothing. Editing `scale.py` is a deliberate act, because every rating
on the site moves when those numbers do.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import europe, rankings, scale  # noqa: E402
from model.parse import TeamRegistry  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Which measure feeds which dimension, and whether the spread is of the value
#: or of its logarithm. Ratios are logged, differences are not, exactly as
#: `scale.dimension` does it.
FROM_RESULTS = (("consistency", "gd_sd", False),)
FROM_SHOTS = (("creation", "sot_pm", True),
              ("finishing", "conversion", True),
              ("discipline", "foul_index", False))


def spread(values: list[float], log: bool) -> float | None:
    xs = [v for v in values if v is not None]
    if log:
        xs = [math.log(v) for v in xs if v > 0]
    return statistics.pstdev(xs) if len(xs) > 20 else None


def line(name: str, got: float | None, reliability: float | None, n: int) -> None:
    cur = scale.EUROPE_SD.get(name)
    drift = f"{(got - cur) / cur * 100:+.0f}%" if (got and cur) else ""
    rel = scale.RELIABILITY.get(name)
    print(f"{name:<14}{'-' if got is None else round(got, 4):>10}"
          f"{'-' if cur is None else cur:>13}{drift:>9}"
          f"{'-' if rel is None else rel:>13}"
          f"{'' if reliability is None else f'{reliability:>9.2f}'}"
          f"   ({n} clubs)")


def main() -> None:
    print("Loading the pooled corpus…")
    corpus = europe.Corpus(TeamRegistry()).load(quiet=True)
    ref = max(dt.date.today(), dt.date(2026, 8, 1))
    hist = corpus.before(ref)
    recent: dict[str, list] = {}
    for m in hist:
        for t in (m.home, m.away):
            recent.setdefault(t, []).append(m)
    prof = {t: rankings._profile(ms, t) for t, ms in recent.items()}
    prof = {t: r for t, r in prof.items() if r}

    print(f"\n{len(prof)} clubs profiled\n")
    print(f"{'dimension':<14}{'measured':>10}{'frozen':>13}{'drift':>9}"
          f"{'reliability':>13}{'fresh':>9}")
    print("-" * 70)

    # The goal differences behind `gd_sd`, recomputed here over the same window
    # `_profile` uses, so the sampling half of the split can be estimated. Kept
    # out of `_profile` itself: production code should not carry a field that
    # exists only for a measurement script.
    gd_of: dict[str, list[int]] = {}
    for t, ms in recent.items():
        if t not in prof:
            continue
        gd = []
        for m in sorted(ms, key=lambda x: x.date)[-rankings.PROFILE_MATCHES:]:
            if m.hg is None or m.ag is None:
                continue
            gd.append(m.hg - m.ag if m.home == t else m.ag - m.hg)
        if len(gd) >= 30:
            gd_of[t] = gd

    for name, key, log in FROM_RESULTS:
        vals = [r[key] for r in prof.values() if r.get(key) is not None]
        # Sampling variance of a standard deviation over n matches is about
        # sd^2 / 2n, which is what makes the split possible at all.
        rel = None
        got = spread(vals, log)
        if got and gd_of:
            obs = got ** 2
            noise = statistics.mean(
                [statistics.pvariance(g) / (2 * len(g)) for g in gd_of.values()])
            rel = max(obs - noise, 0) / obs
        line(name, got, rel, len(vals))

    try:
        sh = json.load(open(os.path.join(HERE, "site", "data", "shooting.json")))
    except OSError:
        print("\n  (site/data/shooting.json not built; run `python -m model.run` first)")
        return
    clubs = [c for c in (sh.get("clubs") or {}).values() if isinstance(c, dict)]
    print()
    for name, key, log in FROM_SHOTS:
        vals = [c.get(key) for c in clubs]
        n = len([v for v in vals if v is not None])
        # Per-club sampling variance, on the same scale the spread is taken on.
        if key == "sot_pm":
            wv = [1.0 / c["sot"] for c in clubs if c.get("sot")]
        elif key == "conversion":
            wv = [(1 - c["conversion"]) / (c["conversion"] * c["sot"])
                  for c in clubs if c.get("conversion") and c.get("sot")]
        else:
            wv = [(c["yellow_pm"] + 9 * c["red_pm"] + c["fouls_pm"] / 36) / c["n"]
                  for c in clubs if c.get("foul_index") is not None]
        got = spread(vals, log)
        rel = None
        if got and wv:
            obs, noise = got ** 2, statistics.mean(wv)
            rel = max(obs - noise, 0) / obs
        line(name, got, rel, n)

    print("\nThe shot-based population is the big five entire: no other feed this "
          "build\ncan reach has a shot or a card in it.")


if __name__ == "__main__":
    main()
