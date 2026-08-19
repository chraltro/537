"""Match probabilities and Monte Carlo season simulation.

Two levels of precision on purpose. Match pages show exact Dixon-Coles score
distributions, because those numbers get read closely. Season simulation uses
sampling, because only the aggregates matter there -- but it still samples from
the exact per-match distribution, and it redraws team ratings between scenarios
so the final table reflects 'we are not certain how good these teams are' and
not merely 'football is random'.
"""
from __future__ import annotations

import numpy as np

from . import config, leagues
from .ratings import Fit, tau

MAXG = config.MAX_GOALS


def score_matrix(lh: float, la: float, rho: float, maxg: int = MAXG) -> np.ndarray:
    """P(home goals = i, away goals = j) as a (maxg+1) x (maxg+1) grid."""
    k = np.arange(maxg + 1)
    lg = np.array([0.0] + list(np.cumsum(np.log(np.arange(1, maxg + 1)))))
    ph = np.exp(-lh + k * np.log(lh) - lg)
    pa = np.exp(-la + k * np.log(la) - lg)
    m = np.outer(ph, pa)
    i, j = np.meshgrid(k, k, indexing="ij")
    m *= tau(i, j, lh, la, rho)
    return m / m.sum()


def outcome_probs(m: np.ndarray) -> tuple[float, float, float]:
    home = float(np.tril(m, -1).sum())
    draw = float(np.trace(m))
    away = float(np.triu(m, 1).sum())
    return home, draw, away


def match_report(fit: Fit, home: str, away: str, adj: dict[str, float] | None = None) -> dict:
    lh, la = _lambdas(fit, home, away, adj)
    m = score_matrix(lh, la, fit.rho)
    ph, pd, pa = outcome_probs(m)
    flat = m.flatten()
    order = np.argsort(flat)[::-1][:6]
    top = [{"h": int(i // m.shape[1]), "a": int(i % m.shape[1]), "p": float(flat[i])}
           for i in order]
    tot = np.add.outer(np.arange(m.shape[0]), np.arange(m.shape[1]))
    return {
        "grid": m,
        "home_win": ph, "draw": pd, "away_win": pa,
        "xg_home": lh, "xg_away": la,
        "top_scores": top,
        "over25": float(m[tot >= 3].sum()),
        "btts": float(m[1:, 1:].sum()),
        "cs_home": float(m[:, 0].sum()),
        "cs_away": float(m[0, :].sum()),
    }


def _lambdas(fit: Fit, home: str, away: str, adj: dict[str, float] | None):
    """Match expectations, optionally shifted by a net-rating adjustment.

    A net-rating nudge is split evenly between scoring more and conceding less:
    with no reason to attribute the shift to one end of the pitch, splitting it
    is the neutral choice.
    """
    lh, la = fit.lambdas(home, away)
    if adj:
        h = adj.get(home, 0.0) / 2
        a = adj.get(away, 0.0) / 2
        lh *= np.exp(h - a)
        la *= np.exp(a - h)
    return float(lh), float(la)


def build_lambdas(fit: Fit, fixtures, adj: dict[str, float] | None = None):
    lh = np.empty(len(fixtures))
    la = np.empty(len(fixtures))
    for k, f in enumerate(fixtures):
        lh[k], la[k] = _lambdas(fit, f.home, f.away, adj)
    return lh, la


def simulate_season(fit: Fit, fixtures, teams: list[str], *,
                    league: leagues.League | None = None,
                    adj: dict[str, float] | None = None,
                    n_sims: int = config.N_SIMS,
                    rating_sd: float = config.RATING_SD,
                    seed: int = config.SEED,
                    scenarios: int = 200,
                    leverage: bool = False) -> dict:
    """Play the rest of the season `n_sims` times.

    Results already on the board are carried in as fact; only the remaining
    fixtures are simulated. `league` decides where the European and relegation
    lines fall; it defaults to the Premier League for callers with none in hand.
    """
    lg = league or leagues.DEFAULT
    ucl, europa, releg = lg.ucl_places, lg.europa_places, lg.releg_places
    rng = np.random.default_rng(seed)
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    base_pts = np.zeros(n)
    base_gf = np.zeros(n)
    base_ga = np.zeros(n)
    remaining = []
    for f in fixtures:
        i, j = idx[f.home], idx[f.away]
        if f.played:
            base_gf[i] += f.hg; base_ga[i] += f.ag
            base_gf[j] += f.ag; base_ga[j] += f.hg
            base_pts[i] += 3 if f.hg > f.ag else (1 if f.hg == f.ag else 0)
            base_pts[j] += 3 if f.ag > f.hg else (1 if f.hg == f.ag else 0)
        else:
            remaining.append(f)

    hi = np.array([idx[f.home] for f in remaining], dtype=np.int64)
    ai = np.array([idx[f.away] for f in remaining], dtype=np.int64)
    lh0, la0 = build_lambdas(fit, remaining, adj)

    per = max(1, n_sims // max(scenarios, 1))
    scenarios = max(1, n_sims // per)
    pos_counts = np.zeros((n, n), dtype=np.int64)
    pts_team = np.empty((scenarios * per, n))
    # Points by finishing position, kept per simulated season: this is what
    # answers "how many points win the title" and "what keeps you up", which
    # no club-centric average can.
    pos_pts = np.empty((scenarios * per, n), dtype=np.float32)
    gd_sum = np.zeros(n)
    # Conditional tallies for match importance: for every remaining fixture and
    # every possible result, how often each club ends up champion / in the top
    # five / relegated. Counting this inside the existing simulation is nearly
    # free, and it answers the question a fan actually has -- does this game
    # matter? -- without a second set of conditional runs.
    n_ev = 3
    lev_hits = np.zeros((3, len(remaining), n * n_ev), dtype=np.float64) if leverage else None
    lev_n = np.zeros((3, len(remaining)), dtype=np.float64) if leverage else None
    kk = np.arange(MAXG + 1)
    lgam = np.array([0.0] + list(np.cumsum(np.log(np.arange(1, MAXG + 1)))))
    cursor = 0

    for _ in range(scenarios):
        # One draw of "how good is everyone really", held fixed for `per` seasons.
        shock = rng.normal(0.0, rating_sd, n) if rating_sd > 0 else np.zeros(n)
        lh = lh0 * np.exp(shock[hi] / 2 - shock[ai] / 2)
        la = la0 * np.exp(shock[ai] / 2 - shock[hi] / 2)

        ph = np.exp(-lh[:, None] + kk[None, :] * np.log(lh)[:, None] - lgam[None, :])
        pa = np.exp(-la[:, None] + kk[None, :] * np.log(la)[:, None] - lgam[None, :])
        grid = ph[:, :, None] * pa[:, None, :]
        gi, gj = np.meshgrid(kk, kk, indexing="ij")
        grid *= tau(gi, gj, lh[:, None, None], la[:, None, None], fit.rho)
        flat = grid.reshape(len(remaining), -1)
        flat /= flat.sum(axis=1, keepdims=True)
        cdf = np.cumsum(flat, axis=1)

        u = rng.random((per, len(remaining)))
        pick = np.array([np.searchsorted(cdf[m], u[:, m]) for m in range(len(remaining))]).T
        hg = (pick // (MAXG + 1)).astype(np.int16)
        ag = (pick % (MAXG + 1)).astype(np.int16)

        pts = np.tile(base_pts, (per, 1))
        gf = np.tile(base_gf, (per, 1))
        ga = np.tile(base_ga, (per, 1))
        hw = (hg > ag) * 3 + (hg == ag)
        aw = (ag > hg) * 3 + (hg == ag)
        np.add.at(pts, (slice(None), hi), hw)
        np.add.at(pts, (slice(None), ai), aw)
        np.add.at(gf, (slice(None), hi), hg)
        np.add.at(ga, (slice(None), hi), ag)
        np.add.at(gf, (slice(None), ai), ag)
        np.add.at(ga, (slice(None), ai), hg)

        gd = gf - ga
        # Points, then goal difference, then goals scored. Serie A and La Liga
        # actually settle level clubs on head-to-head first; over a full season
        # that changes the odd placing, never the shape of the distribution, and
        # tracking it would cost a per-pair tally in the hot loop.
        # A genuine tie is settled by a play-off, so break it at random here.
        key = (pts * 1e9 + (gd + 200) * 1e4 + gf * 1e0
               + rng.random((per, n)) * 1e-3)
        order = np.argsort(-key, axis=1)          # order[s, r] = team finishing r-th
        # bincount, not fancy-index +=: repeated (team, position) pairs across the
        # seasons in this scenario must accumulate, and `+= 1` would count each once.
        flat_idx = order.ravel() * n + np.tile(np.arange(n), per)
        pos_counts += np.bincount(flat_idx, minlength=n * n).reshape(n, n)
        pts_team[cursor:cursor + per] = pts
        pos_pts[cursor:cursor + per] = np.take_along_axis(pts, order, axis=1)
        gd_sum += gd.sum(axis=0)
        cursor += per

        if leverage and len(remaining):
            rank = np.argsort(order, axis=1)            # rank[s, t] = finish (0-based)
            ev = np.empty((per, n, n_ev), dtype=np.float32)
            ev[:, :, 0] = rank == 0
            ev[:, :, 1] = rank < ucl
            ev[:, :, 2] = rank >= n - releg
            ev_flat = ev.reshape(per, n * n_ev)
            res = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))     # (per, n_rem)
            for k in range(3):
                mk = (res == k).T.astype(np.float32)                 # (n_rem, per)
                lev_hits[k] += mk @ ev_flat
                lev_n[k] += mk.sum(axis=1)

    total = cursor
    p = pos_counts / total
    return {
        "teams": teams,
        "position": p,
        "points_mean": pts_team[:total].mean(axis=0),
        "points_p10": np.percentile(pts_team[:total], 10, axis=0),
        "points_p90": np.percentile(pts_team[:total], 90, axis=0),
        "points_max": pts_team[:total].max(axis=0),
        "points_min": pts_team[:total].min(axis=0),
        "gd_mean": gd_sum / total,
        "title": p[:, 0],
        "ucl": p[:, :ucl].sum(axis=1),
        "europa": p[:, ucl:ucl + europa].sum(axis=1),
        "relegation": p[:, -releg:].sum(axis=1),
        "n_sims": total,
        "lines": _lines(pos_pts[:total], lg),
        "leverage": _leverage(lev_hits, lev_n, remaining, teams) if leverage else None,
    }


def _lines(pos_pts: np.ndarray, league: leagues.League | None = None) -> dict:
    """The season's thresholds, read off the simulated tables.

    'How many points win the league' is a different question from 'how many
    points will the winner average', and only the first is useful to a fan
    doing arithmetic in April. Each line reports the points of the side that
    finished in the boundary position, plus the total that was enough in 90%
    of seasons."""
    lg = league or leagues.DEFAULT
    out = {}
    # 'top5' keeps its name across leagues even where the line is 3rd or 4th:
    # the site reads the key and labels it from the manifest's ucl_places.
    for key, pos in (("title", 0), ("top5", lg.ucl_places - 1),
                     ("safety", lg.n_teams - lg.releg_places - 1)):
        col = pos_pts[:, pos]
        out[key] = {
            "p10": int(np.percentile(col, 10)),
            "p50": int(np.percentile(col, 50)),
            "p90": int(np.percentile(col, 90)),
            # for the title you must MEET the winner's total; for top-five and
            # safety, matching the boundary side's points is (roughly) enough
            "enough90": int(np.percentile(col, 90)) + (1 if key != "title" else 0),
        }
    return out


#: Leverage events. Domestic leagues swing title/UCL/relegation; a cup's
#: league phase swings direct qualification / the play-off cut / elimination.
#: The keys are part of the site contract -- the front end maps them to words.
EVENTS = ("title", "ucl", "releg")
CUP_EVENTS = ("top8", "qualify", "out")


def _leverage(hits, counts, remaining, teams) -> list[dict]:
    """Turn conditional tallies into a per-match importance score.

    A match matters to the degree that its result moves someone's season. The
    score is the largest swing in any club's title, top-five or relegation
    chance between a home win and an away win -- so a mid-table dead rubber
    scores near zero even if both clubs are good, and a relegation six-pointer
    scores highly even between two poor sides.
    """
    if hits is None or not len(remaining):
        return []
    n = len(teams)
    out = []
    with np.errstate(invalid="ignore", divide="ignore"):
        probs = hits / np.maximum(counts, 1)[:, :, None]          # (3, n_rem, n*3)
    probs = probs.reshape(3, len(remaining), n, len(EVENTS))
    for m in range(len(remaining)):
        swing = probs[0, m] - probs[2, m]                          # home win - away win
        seen = np.abs(swing)
        # Only rank a club-event pair when the outcome is genuinely in play;
        # a swing between 0.1% and 0.2% is noise, not drama.
        live = (np.maximum(probs[0, m], probs[2, m]) > 0.02) & (seen > 0.005)
        if not live.any():
            out.append({"score": 0.0, "swings": []})
            continue
        flat = np.where(live, seen, 0.0).ravel()
        order_idx = np.argsort(flat)[::-1][:3]
        swings = []
        for f in order_idx:
            if flat[f] <= 0:
                break
            t_i, e_i = divmod(int(f), len(EVENTS))
            swings.append({"team": teams[t_i], "event": EVENTS[e_i],
                           "home": round(float(probs[0, m, t_i, e_i]), 4),
                           "away": round(float(probs[2, m, t_i, e_i]), 4),
                           "swing": round(float(swing[t_i, e_i]), 4)})
        out.append({"score": round(float(flat.max()), 4), "swings": swings})
    return out
