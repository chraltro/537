"""Walk-forward evaluation.

The only honest way to claim a forecast is any good: at every point in the past,
fit on what was knowable then, predict what happened next, and score it. Nothing
here ever sees a result before predicting it.

Scored against three baselines, because "beats a coin flip" is not a claim worth
making. If the model does not beat all three out of sample, the model is wrong.
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict

import numpy as np

from . import config, europe, ratings
from .data import Dataset
from .parse import Match
from .simulate import outcome_probs, score_matrix

EPS = 1e-12


def _result_index(m: Match) -> int:
    return 0 if m.hg > m.ag else (1 if m.hg == m.ag else 2)


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, 1))))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    o = np.zeros_like(p)
    o[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - o) ** 2, axis=1)))


def rps(p: np.ndarray, y: np.ndarray) -> float:
    """Ranked probability score: penalises being wrong in the right direction
    less than being wrong in the wrong direction, which is the right treatment
    for an ordered home/draw/away outcome."""
    o = np.zeros_like(p)
    o[np.arange(len(y)), y] = 1.0
    cp, co = np.cumsum(p, axis=1), np.cumsum(o, axis=1)
    return float(np.mean(np.sum((cp[:, :2] - co[:, :2]) ** 2, axis=1) / 2.0))


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(p, axis=1) == y))


def score_all(p: np.ndarray, y: np.ndarray) -> dict:
    return {"log_loss": log_loss(p, y), "brier": brier(p, y),
            "rps": rps(p, y), "accuracy": accuracy(p, y), "n": int(len(y))}


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
class BaseRate:
    """Home/draw/away base rates from the training data. The bar every football
    model has to clear, and a surprising number do not."""
    name = "Home-field base rate"

    def __init__(self, train: list[Match]):
        c = np.array([sum(1 for m in train if _result_index(m) == k) for k in range(3)], float)
        self.p = c / max(c.sum(), 1)

    def predict(self, m):
        return self.p


class Elo:
    """Standard club Elo with a goal-difference multiplier, mapped to 1X2 by a
    draw curve fitted on training data only."""
    name = "Elo"

    def __init__(self, k: float = 20.0, home: float = 65.0):
        self.k, self.home = k, home
        self.r = defaultdict(lambda: 1500.0)
        self.draw_a, self.draw_b = 0.30, 1.0

    def expected(self, h, a):
        return 1.0 / (1.0 + 10 ** (-(self.r[h] - self.r[a] + self.home) / 400.0))

    def predict(self, m):
        e = self.expected(m.home, m.away)
        d = np.clip(self.draw_a - self.draw_b * (e - 0.5) ** 2, 0.06, 0.40)
        hw = np.clip(e - d / 2, 1e-3, 1 - 1e-3)
        aw = np.clip(1 - hw - d, 1e-3, 1 - 1e-3)
        v = np.array([hw, d, aw])
        return v / v.sum()

    def update(self, m):
        e = self.expected(m.home, m.away)
        s = 1.0 if m.hg > m.ag else (0.5 if m.hg == m.ag else 0.0)
        gd = abs(m.hg - m.ag)
        mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (1 + 3 / 4 + (gd - 3) / 8))
        delta = self.k * mult * (s - e)
        self.r[m.home] += delta
        self.r[m.away] -= delta

    def fit_draw(self, train: list[Match]):
        """Choose the draw curve that best fits training results."""
        es, ys = [], []
        tmp = Elo(self.k, self.home)
        for m in sorted(train, key=lambda x: x.date):
            es.append(tmp.expected(m.home, m.away))
            ys.append(_result_index(m))
            tmp.update(m)
        es, ys = np.array(es), np.array(ys)
        best, ba, bb = 1e9, self.draw_a, self.draw_b
        for a in np.arange(0.24, 0.34, 0.01):
            for b in np.arange(0.4, 2.2, 0.2):
                d = np.clip(a - b * (es - 0.5) ** 2, 0.06, 0.40)
                hw = np.clip(es - d / 2, 1e-3, 1)
                aw = np.clip(1 - hw - d, 1e-3, 1)
                p = np.stack([hw, d, aw], 1)
                p /= p.sum(1, keepdims=True)
                ll = log_loss(p, ys)
                if ll < best:
                    best, ba, bb = ll, a, b
        self.draw_a, self.draw_b = float(ba), float(bb)
        for m in sorted(train, key=lambda x: x.date):
            self.update(m)


class SeasonForm:
    """Season-to-date goal difference per game, mapped through a fitted logistic.
    This is 'just read the league table', quantified."""
    name = "Season-to-date form"

    def __init__(self, train: list[Match]):
        self.gd = defaultdict(list)
        self.scale = 0.55
        self.base = BaseRate(train).p

    def predict(self, m):
        h = np.mean(self.gd[m.home][-38:]) if self.gd[m.home] else 0.0
        a = np.mean(self.gd[m.away][-38:]) if self.gd[m.away] else 0.0
        edge = self.scale * (h - a) + 0.30
        d = np.clip(0.29 - 0.06 * abs(edge), 0.10, 0.32)
        hw = np.clip(1 / (1 + np.exp(-edge)) * (1 - d), 1e-3, 1)
        aw = np.clip(1 - hw - d, 1e-3, 1)
        v = np.array([hw, d, aw])
        return v / v.sum()

    def update(self, m):
        self.gd[m.home].append(m.hg - m.ag)
        self.gd[m.away].append(m.ag - m.hg)


# --------------------------------------------------------------------------
# The model under test
# --------------------------------------------------------------------------
def _rounds(matches: list[Match]) -> list[tuple[dt.date, list[Match]]]:
    """Group a season into prediction rounds by date, so the model refits about
    as often as a real forecast would."""
    by_date: dict[dt.date, list[Match]] = defaultdict(list)
    for m in matches:
        by_date[m.date].append(m)
    days = sorted(by_date)
    out, cur, start = [], [], None
    for d in days:
        if start is None:
            start = d
        if (d - start).days > 3 and cur:
            out.append((start, cur))
            cur, start = [], d
        cur.extend(by_date[d])
    if cur:
        out.append((start, cur))
    return out


#: Seasons whose results were used to choose GOALS_WEIGHT and TIME_DECAY. The
#: grid search ran on the Premier League only, so for the other four leagues
#: every season here is genuinely out of sample -- but the parameters were still
#: not chosen on them, and the site says so rather than claiming a clean split.
TUNED_ON = ("2022-23", "2023-24", "2024-25")


def run(ds: Dataset, *, seasons: list[str] | None = None,
        goals_weight: float = config.GOALS_WEIGHT,
        decay: float = config.TIME_DECAY,
        history_years: int = 5, quiet: bool = False) -> dict:
    all_seasons = sorted({m.season for m in ds.top if m.season != ds.season})
    seasons = seasons or [s for s in all_seasons if s >= ds.league.backtest_from]
    shot_conv = ratings.fit_shot_conversion(
        [m for m in ds.top if m.season < min(seasons)])

    preds: list[np.ndarray] = []
    ys: list[int] = []
    seasons_of: list[str] = []
    base_preds = {k: [] for k in ("base", "elo", "form")}

    warm: dict | None = None
    train0 = [m for m in ds.top if m.season < min(seasons)]
    elo = Elo()
    elo.fit_draw(train0)
    form = SeasonForm(train0)
    for m in sorted(train0, key=lambda x: x.date):
        form.update(m)
    base = BaseRate(train0)

    for season in seasons:
        season_matches = sorted([m for m in ds.top if m.season == season],
                                key=lambda x: x.date)
        for start, chunk in _rounds(season_matches):
            cutoff = min(m.date for m in chunk)
            lo = dt.date(cutoff.year - history_years, 1, 1)
            hist = [m for m in ds.top if lo <= m.date < cutoff] + \
                   [m for m in ds.second if lo <= m.date < cutoff]
            pool = sorted({m.home for m in hist} | {m.away for m in hist})
            need = {m.home for m in chunk} | {m.away for m in chunk}
            if not need <= set(pool) or len(hist) < 500:
                for m in chunk:
                    elo.update(m); form.update(m)
                continue
            fit = ratings.fit(hist, pool, cutoff, decay=decay, ridge=config.RIDGE,
                              goals_weight=goals_weight, shot_conv=shot_conv,
                              warm=warm)
            warm = fit.warm
            for m in chunk:
                lh, la = fit.lambdas(m.home, m.away)
                ph, pd, pa = outcome_probs(score_matrix(lh, la, fit.rho))
                preds.append(np.array([ph, pd, pa]))
                ys.append(_result_index(m))
                seasons_of.append(season)
                base_preds["base"].append(base.predict(m))
                base_preds["elo"].append(elo.predict(m))
                base_preds["form"].append(form.predict(m))
            for m in chunk:
                elo.update(m)
                form.update(m)
        if not quiet:
            print(f"  · {season}: {len(ys)} matches scored so far")

    P = np.array(preds)
    Y = np.array(ys)
    S = np.array(seasons_of)
    held = ~np.isin(S, TUNED_ON)
    out = {"model": score_all(P, Y),
           "held_out": score_all(P[held], Y[held]) if held.any() else None,
           "tuned_on": list(TUNED_ON),
           "by_season": [
               {"season": s, **score_all(P[S == s], Y[S == s]),
                "held_out": s not in TUNED_ON}
               for s in seasons if (S == s).any()],
           "baselines": {k: score_all(np.array(v), Y) for k, v in base_preds.items()},
           "baselines_held_out": {k: score_all(np.array(v)[held], Y[held])
                                  for k, v in base_preds.items()} if held.any() else None,
           "names": {"base": BaseRate.name, "elo": Elo.name, "form": SeasonForm.name},
           "seasons": seasons,
           "params": {"goals_weight": goals_weight, "decay": decay}}
    out["calibration"] = calibration(P, Y)
    return out


# --------------------------------------------------------------------------
# European walk-forward
# --------------------------------------------------------------------------
class LeagueAverage:
    """The bar the pooled fit has to clear in Europe.

    Every club is replaced by the average strength of its own domestic league,
    so the forecast knows only 'a Dutch club is playing a Norwegian one' and
    nothing whatever about which Dutch club. If a pooled fit over four thousand
    European matches cannot beat that, its club-level ratings are decoration.

    Strengths are attack/defence pairs fitted the same way as the real model but
    with clubs collapsed into their leagues, so the comparison isolates the one
    thing under test -- club resolution -- rather than the machinery around it.
    """
    name = "League-average strength"

    def __init__(self, train: list[Match], club_league: dict[str, str],
                 ref_date, decay: float = config.TIME_DECAY):
        self.club_league = club_league
        agg: list[Match] = []
        for m in train:
            lh = club_league.get(m.home, "other")
            la = club_league.get(m.away, "other")
            if lh == la:
                continue          # a domestic match says nothing across leagues
            agg.append(Match(date=m.date, home=lh, away=la, hg=m.hg, ag=m.ag,
                             played=True))
        pool = sorted({m.home for m in agg} | {m.away for m in agg})
        self.fit = ratings.fit(agg, pool, ref_date, decay=decay) if agg else None

    def predict(self, m):
        if self.fit is None:
            return np.array([0.44, 0.25, 0.31])
        h = self.club_league.get(m.home, "other")
        a = self.club_league.get(m.away, "other")
        if h not in self.fit.index or a not in self.fit.index:
            return np.array([0.44, 0.25, 0.31])
        lh, la = self.fit.lambdas(h, a)
        return np.array(outcome_probs(score_matrix(lh, la, self.fit.rho)))


def run_european(corpus, seasons: list[str], *, comps=("cl",),
                 history_years: int = 6, decay: float = config.TIME_DECAY,
                 ridge: float = config.RIDGE, quiet: bool = False) -> dict:
    """Walk forward through European seasons the fit has never seen.

    Refits before every matchday on everything played before it -- domestic and
    European alike -- then scores the matchday. The two Swiss seasons are the
    honest holdout: they are the format actually being forecast.
    """
    club_league = corpus.club_leagues()
    target = sorted([m for m in corpus.matches
                     if m.season in seasons and m.comp in comps and m.played],
                    key=lambda m: (m.date, m.home))
    preds, base_preds, ys = [], [], []
    for start, chunk in _rounds(target):
        cutoff = min(m.date for m in chunk)
        lo = dt.date(cutoff.year - history_years, 1, 1)
        hist = [m for m in corpus.matches if lo <= m.date < cutoff]
        pool = sorted({m.home for m in hist} | {m.away for m in hist})
        need = {m.home for m in chunk} | {m.away for m in chunk}
        if not need <= set(pool) or len(hist) < 2000:
            continue
        fit = ratings.fit_pooled(hist, pool, cutoff, group_of=corpus.group_of,
                                 club_league=club_league, decay=decay,
                                 ridge=ridge, default_group=europe.EUROPE)
        base = LeagueAverage(hist, club_league, cutoff, decay=decay)
        for m in chunk:
            lh, la = fit.lambdas(m.home, m.away, group=europe.EUROPE)
            preds.append(np.array(outcome_probs(score_matrix(lh, la, fit.rho))))
            base_preds.append(base.predict(m))
            ys.append(_result_index(m))
        if not quiet:
            print(f"  · {start}: {len(ys)} European matches scored")
    P, B, Y = np.array(preds), np.array(base_preds), np.array(ys)
    return {"model": score_all(P, Y),
            "baselines": {"league_average": score_all(B, Y)},
            "names": {"league_average": LeagueAverage.name},
            "seasons": seasons, "comps": list(comps),
            "calibration": calibration(P, Y)}


def calibration(P: np.ndarray, Y: np.ndarray, bins: int = 10) -> list[dict]:
    """Do things the model calls 60% likely happen 60% of the time?"""
    flat_p = P.ravel()
    o = np.zeros_like(P)
    o[np.arange(len(Y)), Y] = 1.0
    flat_o = o.ravel()
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i in range(bins):
        sel = (flat_p >= edges[i]) & (flat_p < edges[i + 1])
        if sel.sum() < 20:
            continue
        rows.append({"lo": float(edges[i]), "hi": float(edges[i + 1]),
                     "predicted": float(flat_p[sel].mean()),
                     "observed": float(flat_o[sel].mean()),
                     "n": int(sel.sum())})
    return rows
