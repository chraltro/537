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

from . import config, ratings
from .data import Dataset

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAL_PATH = os.path.join(HERE, ".cache", "calibration.json")


def _season_start(season: str) -> dt.date:
    return dt.date(int(season.split("-")[0]), 7, 1)


def _pl_seasons_present(ds: Dataset) -> list[str]:
    return sorted({m.season for m in ds.pl if m.season != config.SEASON})


def _teams_in(ds: Dataset, season: str) -> set[str]:
    return {m.home for m in ds.pl if m.season == season}


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


def calibrate(ds: Dataset, shot_conv, *, refresh: bool = False) -> dict:
    """Measure how preseason ratings actually translate into league results.

    For every past season: fit on data available before it started, then fit
    what really happened, and regress one on the other -- separately for clubs
    that stayed up and clubs that had just come up.
    """
    if not refresh and os.path.exists(CAL_PATH):
        return json.load(open(CAL_PATH))

    seasons = _pl_seasons_present(ds)
    pairs_cont: list[tuple[float, float]] = []
    pairs_prom: list[tuple[float, float]] = []

    for prev, cur in zip(seasons, seasons[1:]):
        if cur < "2013-14":
            continue
        ref = _season_start(cur)
        hist = [m for m in ds.pl if m.date < ref] + [m for m in ds.ch if m.date < ref]
        if len(hist) < 2000:
            continue
        cur_teams = sorted(_teams_in(ds, cur))
        if len(cur_teams) != config.N_TEAMS:
            continue
        pool = sorted({m.home for m in hist} | {m.away for m in hist})
        prior = ratings.fit(hist, pool, ref, shot_conv=shot_conv)

        actual_matches = [m for m in ds.pl if m.season == cur]
        actual = ratings.fit(actual_matches, cur_teams, _season_start(cur) + dt.timedelta(days=365),
                             decay=0.0, shot_conv=shot_conv)

        pred = _centred_net(prior, cur_teams)
        real = _centred_net(actual, cur_teams)
        promoted = cur_teams and (set(cur_teams) - _teams_in(ds, prev))
        for t in cur_teams:
            if t not in pred or t not in real:
                continue
            (pairs_prom if t in promoted else pairs_cont).append((pred[t], real[t]))

    def regress(pairs):
        if len(pairs) < 8:
            return 1.0, 0.0, len(pairs)
        x = np.array([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])
        slope, intercept = np.polyfit(x, y, 1)
        return float(slope), float(intercept), len(pairs)

    cs, ci, cn = regress(pairs_cont)
    ps, pi, pn = regress(pairs_prom)
    out = {
        "continuing": {"slope": cs, "intercept": ci, "n": cn},
        "promoted": {"slope": ps, "intercept": pi, "n": pn},
        "generated": dt.date.today().isoformat(),
    }
    os.makedirs(os.path.dirname(CAL_PATH), exist_ok=True)
    json.dump(out, open(CAL_PATH, "w"), indent=1)
    return out


def preseason_net(ds: Dataset, fit: ratings.Fit, cal: dict, teams: list[str],
                  prev_season: str) -> dict[str, float]:
    """Apply the measured carryover / promotion corrections to raw ratings."""
    pred = _centred_net(fit, teams)
    returning = _teams_in(ds, prev_season)
    out = {}
    for t in teams:
        c = cal["continuing"] if t in returning else cal["promoted"]
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
    if not os.path.exists(path):
        return {}
    return json.load(open(path))


def market_weight(matches_played: int) -> float:
    """Full weight before a ball is kicked, nothing left after MARKET_DECAY_MW.

    Ten matchweeks is roughly where a season's own results carry more
    information than the preseason consensus did.
    """
    mw = matches_played / config.N_TEAMS * 2.0
    frac = max(0.0, 1.0 - mw / config.MARKET_DECAY_MW)
    return config.MARKET_WEIGHT * frac


def fit_market_adjustment(fit, fixtures, teams: list[str], market: dict, *,
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

    title = devig(market.get("title", {})) if market.get("title") else {}
    rel_raw = {k: 1.0 / v for k, v in market.get("relegation", {}).items() if v > 1.0}
    # Relegation prices are near-coherent on their own (three teams go down), so
    # scale gently to the three available places rather than force-normalising.
    if rel_raw:
        scale = min(1.0, config.RELEGATION_PLACES / max(sum(rel_raw.values()), 1e-9))
        rel = {k: min(v * scale, 0.97) for k, v in rel_raw.items()}
    else:
        rel = {}

    adj = dict(base_adj or {t: 0.0 for t in teams})
    idx = {t: i for i, t in enumerate(teams)}

    def logit(p):
        p = min(max(p, 1e-4), 1 - 1e-4)
        return np.log(p / (1 - p))

    for r in range(rounds):
        sim = simulate.simulate_season(fit, fixtures, teams, adj=adj,
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
