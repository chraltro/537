"""Preseason priors.

Three questions have to be answered before a ball is kicked:

1. How much of last season's rating still applies?  (carryover)
2. How does a Championship rating translate to the Premier League?  (promotion)
3. What does everyone else already know that the results cannot?  (the market)

The first two are measured from history rather than asserted. The third is an
explicit, dated, sourced input whose influence decays to nothing as real results
arrive.
"""
from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np

from . import config, leagues, ratings
from .data import Dataset

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, ".cache")
MARKET_DIR = os.path.join(HERE, "data", "market_priors")

#: Minimum promoted-club pairs before a league's own promotion regression is
#: trusted. Below this the fit is noise dressed up as a measurement.
MIN_PAIRS = 8

#: The Premier League's own fitted constants, the fallback for a league whose
#: second tier is too short to measure. Snapshot of this pipeline's PL
#: calibration over 2013-14..2025-26 (221 continuing and 39 promoted pairs),
#: taken 2026-08-19; it drifts by a point or two as the mirrors revise history,
#: which is exactly why the live number is measured per league and this is only
#: the floor. A promoted club keeps about a third of its measured second-tier
#: edge and starts a third of a goal per game below the division's average.
#: There is no reason to think Spain or Italy differ enough from that for it to
#: be worse than assuming no shrink at all, which is what a failed regression
#: would otherwise imply.
PL_FALLBACK = {
    "continuing": {"slope": 0.9053, "intercept": 0.0250, "n": 0, "source": "premier-league"},
    "promoted": {"slope": 0.3225, "intercept": -0.3441, "n": 0, "source": "premier-league"},
}

#: Bump when the meaning of a cached calibration changes, so a stale file is
#: recomputed instead of being read with a key it never had.
CAL_VERSION = 2


def cal_path(league: leagues.League) -> str:
    return os.path.join(CACHE, f"calibration-{league.slug}.json")


def market_path(league: leagues.League) -> str:
    return os.path.join(MARKET_DIR, league.market_file)


def _season_start(season: str) -> dt.date:
    return dt.date(int(season.split("-")[0]), 7, 1)


def _pl_seasons_present(ds: Dataset) -> list[str]:
    return sorted({m.season for m in ds.top if m.season != ds.season})


def _teams_in(ds: Dataset, season: str) -> set[str]:
    return {m.home for m in ds.top if m.season == season}


def _teams_above(ds: Dataset, season: str) -> set[str]:
    """Clubs that played in the division above in `season`. Empty for a top
    flight, which is what makes every rule below a no-op there."""
    return {m.home for m in ds.above if m.season == season}


def _centred_net(fit: ratings.Fit, teams: list[str]) -> dict[str, float]:
    """Net rating relative to the average of *these* clubs.

    Recentring is what makes a cross-division fit comparable to a single-season
    one: both then mean 'goals better than this league's average side'.
    """
    net = {t: np.log(fit.offence(t)) - np.log(fit.defence(t))
           for t in teams if t in fit.index}
    if not net:
        return {}
    m = float(np.mean(list(net.values())))
    return {t: v - m for t, v in net.items()}


#: A believable range for a carryover slope. Below zero says a club's preseason
#: edge predicts the *opposite* of what happens; above this says the season
#: amplifies preseason differences rather than regressing them, which contradicts
#: every measurement in this repository (the Premier League's promoted slope is
#: 0.31 over 39 cases). Either is a small, noisy sample talking, not a fact about
#: the league, and the Eredivisie and Primeira Liga have only eight seasons each.
SLOPE_BAND = (0.02, 1.25)


def regress(pairs: list[tuple[float, float]], key: str, source: str,
            fallback: str | None = None) -> dict:
    """Fit one league's own correction, or fall back to the Premier League's.

    Below MIN_PAIRS the slope is dominated by whichever two clubs happened to
    overachieve, so the honest choice is a correction measured somewhere with
    enough seasons rather than one measured here with too few. The result says
    which, so the site can too.

    A slope outside `SLOPE_BAND` falls back for the same reason even when there
    were enough pairs: eight seasons of a smaller league can produce a fitted
    slope of 2.0, and applying it would double every promoted club's rating gap
    instead of shrinking it.
    """
    fb = PL_FALLBACK[fallback or key]
    if len(pairs) < MIN_PAIRS:
        return {**fb, "n": len(pairs), "reason": "too few pairs"}
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    slope, intercept = np.polyfit(x, y, 1)
    lo, hi = SLOPE_BAND
    if not lo <= slope <= hi:
        return {**fb, "n": len(pairs),
                "measured_slope": round(float(slope), 4),
                "reason": f"measured slope {slope:.2f} outside {lo}-{hi}"}
    return {"slope": float(slope), "intercept": float(intercept),
            "n": len(pairs), "source": source}


def calibrate(ds: Dataset, shot_conv, *, refresh: bool = False) -> dict:
    """Measure how preseason ratings actually translate into league results.

    For every past season: fit on data available before it started, then fit
    what really happened, and regress one on the other -- separately for clubs
    that stayed, clubs that had just come up, and clubs that had just come down.

    That third group only exists for a division with something above it, and
    getting it wrong is expensive: lumping a relegated club in with the promoted
    ones applies a correction measured on clubs arriving from a weaker league to
    a club arriving from a stronger one, which pushes it the wrong way by about
    a third of a goal a game.
    """
    path = cal_path(ds.league)
    if not refresh and os.path.exists(path):
        try:
            cached = json.load(open(path))
            if cached.get("version") == CAL_VERSION:
                return cached
        except (OSError, ValueError):
            pass

    seasons = _pl_seasons_present(ds)
    pairs_cont: list[tuple[float, float]] = []
    pairs_prom: list[tuple[float, float]] = []
    pairs_releg: list[tuple[float, float]] = []

    for prev, cur in zip(seasons, seasons[1:]):
        if cur < "2013-14":
            continue
        ref = _season_start(cur)
        hist = ds.before(ref)
        if len(hist) < 2000:
            continue
        cur_teams = sorted(_teams_in(ds, cur))
        # Only complete seasons, whatever size the division was that year.
        # Ligue 1 dropped from 20 clubs to 18 in 2023-24 and Serie A rose from
        # 18 to 20 in 2004-05; pinning this to today's team count would throw
        # away most of the history in exactly the leagues that need it.
        n_cur = len(cur_teams)
        if n_cur < 16 or sum(1 for m in ds.top if m.season == cur) != n_cur * (n_cur - 1):
            continue
        pool = sorted({m.home for m in hist} | {m.away for m in hist})
        prior = ratings.fit(hist, pool, ref, shot_conv=shot_conv)

        actual_matches = [m for m in ds.top if m.season == cur]
        actual = ratings.fit(actual_matches, cur_teams, _season_start(cur) + dt.timedelta(days=365),
                             decay=0.0, shot_conv=shot_conv)

        pred = _centred_net(prior, cur_teams)
        real = _centred_net(actual, cur_teams)
        newcomers = set(cur_teams) - _teams_in(ds, prev)
        came_down = newcomers & _teams_above(ds, prev)
        for t in cur_teams:
            if t not in pred or t not in real:
                continue
            if t in came_down:
                pairs_releg.append((pred[t], real[t]))
            elif t in newcomers:
                pairs_prom.append((pred[t], real[t]))
            else:
                pairs_cont.append((pred[t], real[t]))

    out = {
        "version": CAL_VERSION,
        "continuing": regress(pairs_cont, "continuing", ds.league.slug),
        "promoted": regress(pairs_prom, "promoted", ds.league.slug),
        # A club dropping into this division is rated on its record in the one
        # above, which the fit has placed on the same scale via the clubs that
        # played in both. That is a measurement, not a guess, so when there are
        # too few cases to regress it the honest fallback is to treat it like
        # any continuing club rather than to shrink it like a promoted one.
        "relegated": regress(pairs_releg, "relegated", ds.league.slug,
                             fallback="continuing"),
        "league": ds.league.slug,
        "generated": dt.date.today().isoformat(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    return out


def arrivals(ds: Dataset, teams: list[str], prev_season: str) -> dict[str, str]:
    """How each club got here: 'stayed', 'up' or 'down'.

    A top flight only ever has the first two, because `ds.above` is empty and no
    club can arrive from a division that is not loaded.
    """
    returning = _teams_in(ds, prev_season)
    above = _teams_above(ds, prev_season)
    out = {}
    for t in teams:
        if t in returning:
            out[t] = "stayed"
        elif t in above:
            out[t] = "down"
        else:
            out[t] = "up"
    return out


#: Which measured correction each kind of arrival gets.
_CORRECTION = {"stayed": "continuing", "up": "promoted", "down": "relegated"}


def preseason_net(ds: Dataset, fit: ratings.Fit, cal: dict, teams: list[str],
                  prev_season: str) -> dict[str, float]:
    """Apply the measured carryover / promotion / relegation corrections."""
    pred = _centred_net(fit, teams)
    how = arrivals(ds, teams, prev_season)
    out = {}
    for t in teams:
        key = _CORRECTION[how[t]]
        c = cal.get(key) or cal["continuing"]
        out[t] = c["slope"] * pred.get(t, 0.0) + c["intercept"]
    m = float(np.mean(list(out.values())))
    return {t: v - m for t, v in out.items()}


# --------------------------------------------------------------------------
# Market anchor
# --------------------------------------------------------------------------
def devig(odds: dict[str, float]) -> dict[str, float]:
    """Decimal odds -> probabilities with the bookmaker's margin divided out."""
    raw = {k: 1.0 / v for k, v in odds.items() if v and v > 1.0}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()} if total > 0 else {}


def load_market(path: str) -> dict:
    """A missing file is not an error: it means that league has no anchor yet,
    and `market_weight` is multiplied by nothing."""
    if not os.path.exists(path):
        return {}
    return json.load(open(path))


def market_weight(matches_played: int, league: leagues.League | None = None) -> float:
    """Full weight before a ball is kicked, nothing left after MARKET_DECAY_MW.

    Ten matchweeks is roughly where a season's own results carry more
    information than the preseason consensus did.
    """
    lg = league or leagues.DEFAULT
    mw = matches_played / lg.n_teams * 2.0
    frac = max(0.0, 1.0 - mw / config.MARKET_DECAY_MW)
    return config.MARKET_WEIGHT * frac


def fit_market_adjustment(fit, fixtures, teams: list[str], market: dict, *,
                          league: leagues.League | None = None,
                          base_adj: dict[str, float] | None = None,
                          rounds: int = 9, n_sims: int = 8000,
                          verbose: bool = False) -> dict[str, float]:
    """Solve for the per-club rating nudge that reproduces the betting market.

    Bookmakers price in things a results model cannot see -- a new manager, a
    £90m striker, a squad gutted over the summer. Rather than hand-editing
    ratings, this asks a mechanical question: what rating shift would make the
    simulation agree with the market's title and relegation prices?

    Only clubs with an actual quote are moved; the rest are left alone and drift
    only through renormalisation.
    """
    from . import simulate

    lg = league or leagues.DEFAULT
    title = devig(market.get("title", {})) if market.get("title") else {}
    rel_raw = {k: 1.0 / v for k, v in market.get("relegation", {}).items() if v > 1.0}
    # Relegation prices are near-coherent on their own (a known number of clubs
    # go down), so scale gently to the available places rather than
    # force-normalising.
    if rel_raw:
        scale = min(1.0, lg.releg_places / max(sum(rel_raw.values()), 1e-9))
        rel = {k: min(v * scale, 0.97) for k, v in rel_raw.items()}
    else:
        rel = {}

    adj = dict(base_adj or {t: 0.0 for t in teams})
    idx = {t: i for i, t in enumerate(teams)}

    def logit(p):
        p = min(max(p, 1e-4), 1 - 1e-4)
        return np.log(p / (1 - p))

    for r in range(rounds):
        sim = simulate.simulate_season(fit, fixtures, teams, league=lg, adj=adj,
                                       n_sims=n_sims, scenarios=40, seed=config.SEED + r)
        gain = 0.055 * (1.0 - 0.5 * r / max(rounds - 1, 1))    # damp as it converges
        delta = {t: [] for t in teams}
        for t, p in title.items():
            if t in idx:
                delta[t].append(logit(p) - logit(float(sim["title"][idx[t]])))
        for t, p in rel.items():
            if t in idx:
                delta[t].append(logit(float(sim["relegation"][idx[t]])) - logit(p))
        moved = 0.0
        for t, ds in delta.items():
            if not ds:
                continue
            step = float(np.clip(np.mean(ds) * gain, -0.12, 0.12))
            adj[t] += step
            moved = max(moved, abs(step))
        m = float(np.mean(list(adj.values())))
        adj = {t: v - m for t, v in adj.items()}
        if verbose:
            print(f"  market round {r + 1}: largest move {moved:+.4f}")
        if moved < 0.004:
            break
    return adj
