"""Time-weighted Dixon-Coles ratings.

Each club gets an attack and a defence parameter, which convert directly into
the two numbers SPI was built on: goals expected scored, and goals expected
conceded, against an average team on neutral ground.

The one idea worth stating plainly: the model is fitted twice, once on goals
actually scored and once on goals the shot profile implies, then the two sets of
ratings are blended. Shots are the more repeatable signal over a handful of
matches, which is why a rating model beats reading the league table.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from . import config
from .parse import Match


# --------------------------------------------------------------------------
# Shots -> expected goals
# --------------------------------------------------------------------------
def fit_shot_conversion(matches: list[Match]) -> tuple[float, float]:
    """League-wide Poisson conversion: goals ~ a*on_target + b*off_target.

    No shot locations are available in a free feed, so this is a coarse xG
    stand-in. It is still enough to separate a team that created chances from
    one that got a lucky bounce, which is the whole point.
    """
    rows = [(m.hst, m.hs, m.hg) for m in matches if m.hst is not None and m.hs is not None] + \
           [(m.ast, m.as_, m.ag) for m in matches if m.ast is not None and m.as_ is not None]
    rows = [(st, s - st, g) for st, s, g in rows if st is not None and s is not None
            and g is not None and s >= st >= 0]
    if len(rows) < 500:
        return 0.30, 0.03
    st = np.array([r[0] for r in rows], float)
    off = np.array([r[1] for r in rows], float)
    g = np.array([r[2] for r in rows], float)

    def nll(p):
        a, b = np.exp(p)
        lam = np.maximum(a * st + b * off, 1e-9)
        return float(np.sum(lam - g * np.log(lam)))

    res = minimize(nll, np.log([0.30, 0.03]), method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
    a, b = np.exp(res.x)
    return float(a), float(b)


def expected_goals(m: Match, a: float, b: float) -> tuple[float, float] | None:
    if None in (m.hs, m.as_, m.hst, m.ast):
        return None
    hx = a * m.hst + b * max(m.hs - m.hst, 0)
    ax = a * m.ast + b * max(m.as_ - m.ast, 0)
    return hx, ax


# --------------------------------------------------------------------------
# Dixon-Coles
# --------------------------------------------------------------------------
class Fit:
    """One fitted rating set."""

    def __init__(self, teams: list[str], att: np.ndarray, dfn: np.ndarray,
                 mu: float, home: float, rho: float) -> None:
        self.teams = teams
        self.index = {t: i for i, t in enumerate(teams)}
        self.att = att
        self.dfn = dfn
        self.mu = mu
        self.home = home
        self.rho = rho

    def lambdas(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        i, j = self.index[home], self.index[away]
        adv = 0.0 if neutral else self.home
        return (float(np.exp(self.mu + self.att[i] - self.dfn[j] + adv)),
                float(np.exp(self.mu + self.att[j] - self.dfn[i])))

    def offence(self, team: str) -> float:
        """Goals this club would be expected to score against an average team."""
        i = self.index[team]
        return float(np.exp(self.mu + self.att[i]))

    def defence(self, team: str) -> float:
        """Goals it would be expected to concede against an average team."""
        i = self.index[team]
        return float(np.exp(self.mu - self.dfn[i]))


def _pack(teams, matches, ref_date, decay):
    idx = {t: i for i, t in enumerate(teams)}
    hi, ai, hg, ag, w = [], [], [], [], []
    for m in matches:
        if m.home not in idx or m.away not in idx:
            continue
        hi.append(idx[m.home])
        ai.append(idx[m.away])
        hg.append(m.hg)
        ag.append(m.ag)
        w.append(np.exp(-decay * (ref_date - m.date).days))
    return (np.array(hi), np.array(ai), np.asarray(hg, float),
            np.asarray(ag, float), np.asarray(w, float))


def _fit_core(n, hi, ai, y_h, y_a, w, ridge, x0=None):
    """MLE for attack/defence/home/intercept with an analytic gradient.

    The low-score correlation term is profiled out afterwards: it barely moves
    the attack and defence estimates but makes the objective far messier, and
    fitting it separately keeps the backtest's hundreds of refits fast.
    """
    def obj(x):
        att, dfn = x[:n], x[n:2 * n]
        mu, home = x[2 * n], x[2 * n + 1]
        lh = np.exp(mu + att[hi] - dfn[ai] + home)
        la = np.exp(mu + att[ai] - dfn[hi])
        nll = float(np.sum(w * (lh - y_h * np.log(lh) + la - y_a * np.log(la))))
        nll += ridge * float(np.sum(att ** 2) + np.sum(dfn ** 2))
        rh = w * (y_h - lh)
        ra = w * (y_a - la)
        g_att = np.zeros(n)
        g_dfn = np.zeros(n)
        np.add.at(g_att, hi, -rh)
        np.add.at(g_att, ai, -ra)
        np.add.at(g_dfn, ai, rh)
        np.add.at(g_dfn, hi, ra)
        g_att += 2 * ridge * att
        g_dfn += 2 * ridge * dfn
        grad = np.concatenate([g_att, g_dfn, [-np.sum(rh) - np.sum(ra)], [-np.sum(rh)]])
        return nll, grad

    if x0 is None:
        x0 = np.zeros(2 * n + 2)
        x0[2 * n] = np.log(max(y_h.mean(), 0.1))
        x0[2 * n + 1] = 0.25
    res = minimize(obj, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-10})
    x = res.x
    att, dfn = x[:n], x[n:2 * n]
    # Identifiability: attack and defence are only defined up to a shift.
    att = att - att.mean()
    dfn = dfn - dfn.mean()
    return att, dfn, float(x[2 * n]), float(x[2 * n + 1])


def tau(hg, ag, lh, la, rho):
    """Dixon-Coles correction for the dependence between low scores.

    Without it the model systematically under-predicts 0-0 and 1-1, which are
    the two most common scorelines in the league.
    """
    t = np.ones_like(np.asarray(hg, float))
    h, a = np.asarray(hg), np.asarray(ag)
    t = np.where((h == 0) & (a == 0), 1 - lh * la * rho, t)
    t = np.where((h == 0) & (a == 1), 1 + lh * rho, t)
    t = np.where((h == 1) & (a == 0), 1 + la * rho, t)
    t = np.where((h == 1) & (a == 1), 1 - rho, t)
    return np.maximum(t, 1e-9)


def _fit_rho(hi, ai, y_h, y_a, w, att, dfn, mu, home):
    lh = np.exp(mu + att[hi] - dfn[ai] + home)
    la = np.exp(mu + att[ai] - dfn[hi])
    ih, ia = np.rint(y_h), np.rint(y_a)

    def nll(rho):
        return -float(np.sum(w * np.log(tau(ih, ia, lh, la, rho))))

    res = minimize_scalar(nll, bounds=(-0.25, 0.25), method="bounded")
    return float(res.x)


def fit(matches: list[Match], teams: list[str], ref_date, *,
        decay: float = config.TIME_DECAY, ridge: float = config.RIDGE,
        goals_weight: float = config.GOALS_WEIGHT,
        shot_conv: tuple[float, float] | None = None) -> Fit:
    """Fit ratings on goals and on shot-implied goals, then blend."""
    n = len(teams)
    hi, ai, y_h, y_a, w = _pack(teams, matches, ref_date, decay)
    if len(hi) == 0:
        return Fit(teams, np.zeros(n), np.zeros(n), np.log(1.35), 0.25, 0.0)

    att_g, dfn_g, mu_g, home_g = _fit_core(n, hi, ai, y_h, y_a, w, ridge)
    rho = _fit_rho(hi, ai, y_h, y_a, w, att_g, dfn_g, mu_g, home_g)

    if goals_weight >= 0.999 or shot_conv is None:
        return Fit(teams, att_g, dfn_g, mu_g, home_g, rho)

    a, b = shot_conv
    xg = [(m, expected_goals(m, a, b)) for m in matches]
    sub = [(m, xv) for m, xv in xg if xv is not None]
    if len(sub) < 200:
        return Fit(teams, att_g, dfn_g, mu_g, home_g, rho)

    idx = {t: i for i, t in enumerate(teams)}
    keep = [(idx[m.home], idx[m.away], xv[0], xv[1],
             np.exp(-decay * (ref_date - m.date).days))
            for m, xv in sub if m.home in idx and m.away in idx]
    shi = np.array([k[0] for k in keep])
    sai = np.array([k[1] for k in keep])
    sxh = np.array([k[2] for k in keep], float)
    sxa = np.array([k[3] for k in keep], float)
    sw = np.array([k[4] for k in keep], float)
    att_x, dfn_x, mu_x, home_x = _fit_core(n, shi, sai, sxh, sxa, sw, ridge)

    # Only blend for clubs that actually have shot data. Championship matches
    # come from a goals-only feed, so blending a promoted club toward a rating
    # fitted on no evidence would quietly drag it to league average.
    seen = np.zeros(n)
    np.add.at(seen, shi, sw)
    np.add.at(seen, sai, sw)
    coverage = np.clip(seen / max(np.median(seen[seen > 0]) * 0.25, 1e-9), 0, 1)
    g = goals_weight + (1 - goals_weight) * (1 - coverage)

    return Fit(teams,
               g * att_g + (1 - g) * att_x,
               g * dfn_g + (1 - g) * dfn_x,
               float(goals_weight * mu_g + (1 - goals_weight) * mu_x),
               float(goals_weight * home_g + (1 - goals_weight) * home_x),
               rho)
