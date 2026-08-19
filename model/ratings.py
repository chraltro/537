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
    """One fitted rating set.

    `warm` carries the raw parameters keyed by club, so a later fit over a
    slightly different match window can start from here instead of from zero.
    In the walk-forward backtest consecutive fits differ by about ten matches,
    which turns hundreds of cold optimisations into a few iterations each.
    """

    def __init__(self, teams: list[str], att: np.ndarray, dfn: np.ndarray,
                 mu: float, home: float, rho: float,
                 warm: dict | None = None,
                 homes: dict[str, float] | None = None,
                 default_group: str | None = None,
                 club_league: dict[str, str] | None = None) -> None:
        self.teams = teams
        self.index = {t: i for i, t in enumerate(teams)}
        self.att = att
        self.dfn = dfn
        self.mu = mu
        self.home = home
        self.rho = rho
        self.warm = warm or {}
        #: Home advantage per competition group. A single-league fit has one
        #: entry and `home` is it; a pooled fit has one per domestic league plus
        #: one for Europe, because European home advantage is measurably about
        #: half again the Premier League's (plan 3.1) and averaging the two
        #: makes both wrong.
        self.homes: dict[str, float] = dict(homes or {})
        self.default_group = default_group
        #: Which league each club's rating is shrunk toward, kept for callers
        #: that want to report or re-centre it.
        self.club_league: dict[str, str] = dict(club_league or {})

    def home_advantage(self, group: str | None = None) -> float:
        """Log home advantage for one competition group."""
        if group is None:
            group = self.default_group
        if group is not None and group in self.homes:
            return self.homes[group]
        return self.home

    def lambdas(self, home: str, away: str, neutral: bool = False,
                group: str | None = None) -> tuple[float, float]:
        i, j = self.index[home], self.index[away]
        adv = 0.0 if neutral else self.home_advantage(group)
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


def _warm_start(n, teams, warm, key):
    """Rebuild a starting vector for `teams` from a previous fit's parameters.

    Clubs the earlier fit never saw simply start at zero, which is the right
    prior for a side entering the window.
    """
    if not warm or key not in warm:
        return None
    att_map, dfn_map, mu, home = warm[key]
    x0 = np.zeros(2 * n + 2)
    for i, t in enumerate(teams):
        x0[i] = att_map.get(t, 0.0)
        x0[n + i] = dfn_map.get(t, 0.0)
    x0[2 * n], x0[2 * n + 1] = mu, home
    return x0


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


# --------------------------------------------------------------------------
# Pooled fit: one model over several competitions at once
# --------------------------------------------------------------------------
def _pack_pooled(teams, matches, ref_date, decay, group_of, group_ids):
    """As `_pack`, plus the competition group each match belongs to."""
    idx = {t: i for i, t in enumerate(teams)}
    gidx = {g: i for i, g in enumerate(group_ids)}
    hi, ai, hg, ag, w, gi = [], [], [], [], [], []
    for m in matches:
        if m.home not in idx or m.away not in idx:
            continue
        hi.append(idx[m.home])
        ai.append(idx[m.away])
        hg.append(m.hg)
        ag.append(m.ag)
        w.append(np.exp(-decay * (ref_date - m.date).days))
        gi.append(gidx[group_of(m)])
    return (np.array(hi), np.array(ai), np.asarray(hg, float),
            np.asarray(ag, float), np.asarray(w, float),
            np.asarray(gi, np.int64))


def _fit_pooled_core(n, hi, ai, y_h, y_a, w, gi, n_groups, li, n_leagues,
                     ridge, x0=None):
    """MLE over a pooled corpus, with two changes from `_fit_core`.

    *A home coefficient per competition group.* `home` becomes a vector indexed
    by the match's group instead of a scalar shared by every league on the
    planet.

    *A ridge toward the league mean rather than toward zero.* Each club's attack
    is `league_mean[L(club)] + dev`, and only `dev` is penalised. This is the
    change that makes a pooled fit safe: the old zero-centred penalty, applied
    across a corpus that contains both the Premier League and the Gibraltar
    league, would drag Real Madrid down and Lincoln Red Imps up in equal measure
    and call the result a rating. The league means are free parameters, and what
    identifies them is the European matches -- the only edges in the graph that
    join one league to another.

    Parameter layout: [dev_att(n) | dev_dfn(n) | mu | home(n_groups) |
                       lmean_att(n_leagues) | lmean_dfn(n_leagues)].
    """
    n_par = 2 * n + 1 + n_groups + 2 * n_leagues
    o_mu = 2 * n
    o_home = o_mu + 1
    o_la = o_home + n_groups
    o_ld = o_la + n_leagues

    def obj(x):
        d_att, d_dfn = x[:n], x[n:2 * n]
        mu = x[o_mu]
        home = x[o_home:o_home + n_groups]
        la_m, ld_m = x[o_la:o_la + n_leagues], x[o_ld:o_ld + n_leagues]
        att = la_m[li] + d_att
        dfn = ld_m[li] + d_dfn
        lh = np.exp(mu + att[hi] - dfn[ai] + home[gi])
        la = np.exp(mu + att[ai] - dfn[hi])
        nll = float(np.sum(w * (lh - y_h * np.log(lh) + la - y_a * np.log(la))))
        nll += ridge * float(np.sum(d_att ** 2) + np.sum(d_dfn ** 2))

        rh = w * (y_h - lh)
        ra = w * (y_a - la)
        g_att = np.zeros(n)
        g_dfn = np.zeros(n)
        np.add.at(g_att, hi, -rh)
        np.add.at(g_att, ai, -ra)
        np.add.at(g_dfn, ai, rh)
        np.add.at(g_dfn, hi, ra)
        # The league mean moves every club in that league together, so its
        # gradient is the sum of theirs -- which is also why a league with no
        # European match is free to drift: nothing outside it constrains the sum.
        g_la = np.bincount(li, weights=g_att, minlength=n_leagues)
        g_ld = np.bincount(li, weights=g_dfn, minlength=n_leagues)
        g_home = -np.bincount(gi, weights=rh, minlength=n_groups)
        grad = np.concatenate([
            g_att + 2 * ridge * d_att,
            g_dfn + 2 * ridge * d_dfn,
            [-np.sum(rh) - np.sum(ra)],
            g_home, g_la, g_ld,
        ])
        return nll, grad

    if x0 is None:
        x0 = np.zeros(n_par)
        x0[o_mu] = np.log(max(y_h.mean(), 0.1))
        x0[o_home:o_home + n_groups] = 0.25
    res = minimize(obj, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 800, "ftol": 1e-10})
    x = res.x
    li_att = x[o_la:o_la + n_leagues][li] + x[:n]
    li_dfn = x[o_ld:o_ld + n_leagues][li] + x[n:2 * n]
    # Attack and defence are only defined up to a shift; unlike the single-league
    # fit, here the shift is genuinely non-zero (league means are free), so the
    # intercept has to absorb it or the whole corpus changes scoring level.
    a_bar, d_bar = float(li_att.mean()), float(li_dfn.mean())
    mu = float(x[o_mu]) + a_bar - d_bar
    return (li_att - a_bar, li_dfn - d_bar, mu,
            np.asarray(x[o_home:o_home + n_groups], float))


def fit_pooled(matches: list[Match], teams: list[str], ref_date, *,
               group_of, club_league: dict[str, str],
               default_group: str | None = None,
               decay: float = config.TIME_DECAY,
               ridge: float = config.RIDGE) -> Fit:
    """One Dixon-Coles fit over several competitions at once.

    Goals only: the pooled corpus has no shot columns outside the big five, and
    blending a shot-fitted rating for one fifth of the clubs would put those
    clubs on a different scale from the rest -- exactly the disease this fit is
    meant to cure.
    """
    n = len(teams)
    groups = sorted({group_of(m) for m in matches})
    leagues_ = sorted({club_league.get(t, "other") for t in teams})
    lidx = {g: i for i, g in enumerate(leagues_)}
    li = np.array([lidx[club_league.get(t, "other")] for t in teams], np.int64)

    hi, ai, y_h, y_a, w, gi = _pack_pooled(teams, matches, ref_date, decay,
                                           group_of, groups)
    if len(hi) == 0:
        return Fit(teams, np.zeros(n), np.zeros(n), np.log(1.35), 0.25, 0.0)

    att, dfn, mu, home = _fit_pooled_core(
        n, hi, ai, y_h, y_a, w, gi, len(groups), li, len(leagues_), ridge)
    homes = dict(zip(groups, (float(h) for h in home)))
    dflt = default_group if default_group in homes else groups[0]
    rho = _fit_rho(hi, ai, y_h, y_a, w, att, dfn, mu, homes[dflt])
    return Fit(teams, att, dfn, mu, homes[dflt], rho,
               homes=homes, default_group=dflt, club_league=club_league)


def staleness_sd(teams: list[str], last_seen: dict, ref_date, *,
                 base: float = config.RATING_SD, cap: float = 2.0) -> np.ndarray:
    """Per-club rating uncertainty, widened where the evidence is old.

    Twenty-nine of the participating associations have no domestic result after
    May 2025 (plan 1.5). A club rated on a squad that has since been rebuilt is
    not wrongly rated, it is *uncertainly* rated, and the scenario resampling in
    `simulate_season` is the honest place to say so: `base x (1 + months/12)`,
    capped at twice base, per plan 3.3.
    """
    out = np.full(len(teams), float(base))
    for i, t in enumerate(teams):
        seen = last_seen.get(t)
        if seen is None:
            out[i] = base * cap
            continue
        months = max((ref_date - seen).days, 0) / 30.44
        out[i] = base * min(1.0 + months / 12.0, cap)
    return out


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
        shot_conv: tuple[float, float] | None = None,
        warm: dict | None = None) -> Fit:
    """Fit ratings on goals and on shot-implied goals, then blend."""
    n = len(teams)
    hi, ai, y_h, y_a, w = _pack(teams, matches, ref_date, decay)
    if len(hi) == 0:
        return Fit(teams, np.zeros(n), np.zeros(n), np.log(1.35), 0.25, 0.0)

    att_g, dfn_g, mu_g, home_g = _fit_core(
        n, hi, ai, y_h, y_a, w, ridge, _warm_start(n, teams, warm, "goals"))
    rho = _fit_rho(hi, ai, y_h, y_a, w, att_g, dfn_g, mu_g, home_g)
    out_warm = {"goals": (dict(zip(teams, att_g)), dict(zip(teams, dfn_g)), mu_g, home_g)}

    if goals_weight >= 0.999 or shot_conv is None:
        return Fit(teams, att_g, dfn_g, mu_g, home_g, rho, out_warm)

    a, b = shot_conv
    xg = [(m, expected_goals(m, a, b)) for m in matches]
    sub = [(m, xv) for m, xv in xg if xv is not None]
    if len(sub) < 200:
        return Fit(teams, att_g, dfn_g, mu_g, home_g, rho, out_warm)

    idx = {t: i for i, t in enumerate(teams)}
    keep = [(idx[m.home], idx[m.away], xv[0], xv[1],
             np.exp(-decay * (ref_date - m.date).days))
            for m, xv in sub if m.home in idx and m.away in idx]
    shi = np.array([k[0] for k in keep])
    sai = np.array([k[1] for k in keep])
    sxh = np.array([k[2] for k in keep], float)
    sxa = np.array([k[3] for k in keep], float)
    sw = np.array([k[4] for k in keep], float)
    att_x, dfn_x, mu_x, home_x = _fit_core(
        n, shi, sai, sxh, sxa, sw, ridge, _warm_start(n, teams, warm, "shots"))
    out_warm["shots"] = (dict(zip(teams, att_x)), dict(zip(teams, dfn_x)), mu_x, home_x)

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
               rho, out_warm)
