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

from . import config
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
                    adj: dict[str, float] | None = None,
                    n_sims: int = config.N_SIMS,
                    rating_sd: float = config.RATING_SD,
                    seed: int = config.SEED,
                    scenarios: int = 200) -> dict:
    """Play the rest of the season `n_sims` times.

    Results already on the board are carried in as fact; only the remaining
    fixtures are simulated.
    """
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
    pts_all = np.empty(scenarios * per)
    pts_team = np.empty((scenarios * per, n))
    gd_sum = np.zeros(n)
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
        # Premier League order: points, then goal difference, then goals scored.
        # A genuine tie is settled by a play-off, so break it at random here.
        key = (pts * 1e9 + (gd + 200) * 1e4 + gf * 1e0
               + rng.random((per, n)) * 1e-3)
        order = np.argsort(-key, axis=1)          # order[s, r] = team finishing r-th
        # bincount, not fancy-index +=: repeated (team, position) pairs across the
        # seasons in this scenario must accumulate, and `+= 1` would count each once.
        flat_idx = order.ravel() * n + np.tile(np.arange(n), per)
        pos_counts += np.bincount(flat_idx, minlength=n * n).reshape(n, n)
        pts_team[cursor:cursor + per] = pts
        gd_sum += gd.sum(axis=0)
        cursor += per

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
        "ucl": p[:, :config.UCL_PLACES].sum(axis=1),
        "europa": p[:, config.UCL_PLACES:config.UCL_PLACES + config.EUROPA_PLACES].sum(axis=1),
        "relegation": p[:, -config.RELEGATION_PLACES:].sum(axis=1),
        "n_sims": total,
    }
