"""Walk-forward evaluation.

The only honest way to claim a forecast is any good: at every point in the past,
fit on what was knowable then, predict what happened next, and score it. Nothing
here ever sees a result before predicting it.

Scored against three baselines, because "beats a coin flip" is not a claim worth
making. If the model does not beat all three out of sample, the model is wrong.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from collections import defaultdict

import numpy as np

from . import config, europe, ratings
from .data import Dataset
from .parse import Match
from .simulate import _lambdas, outcome_probs, score_matrix, sharpen_probs

EPS = 1e-12

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, ".cache")

#: Below this many scored matches a sharpening exponent is not fitted at all.
#: One number over four thousand matches is a measurement; the same number over
#: the three hundred and seventy-eight of the two Swiss Champions League seasons
#: is the last two seasons' luck, and applying it would be exactly the hidden
#: thumb on the scale this correction is meant not to be. So a cup runs at
#: k = 1 until its own walk-forward is long enough to say otherwise.
MIN_SHARPEN_N = 750

#: How far from 1 a fitted exponent may go. The measured values are 1.18
#: (Premier League) and 0.80 (Belgium); anything outside this band is a fit
#: that has found something other than calibration.
SHARPEN_BAND = (0.70, 1.40)

#: The fraction of a league's scored matches, chronologically, that the honest
#: evaluation of the exponent is fitted on. The rest is the held-out half the
#: reported gain is measured over.
SHARPEN_TRAIN = 0.60


def sharpen_path(slug: str) -> str:
    return os.path.join(CACHE, f"sharpen-{slug}.json")


def load_sharpen(slug: str) -> float:
    """The exponent the last backtest of this league fitted, or 1.0.

    Cached rather than recomputed because the exponent is worth about 0.001 of
    log-loss and the walk-forward that measures it costs twenty-five seconds:
    on a `SKIP_BACKTEST=1` build the honest choice is last week's measured
    number, and on a checkout that has never run one it is 1.0, which is the
    uncalibrated model this correction started from.
    """
    try:
        with open(sharpen_path(slug), encoding="utf-8") as fh:
            k = float(json.load(fh)["k"])
    except (OSError, ValueError, KeyError, TypeError):
        return 1.0
    lo, hi = SHARPEN_BAND
    return float(min(max(k, lo), hi))


def save_sharpen(slug: str, block: dict) -> None:
    try:
        os.makedirs(CACHE, exist_ok=True)
        with open(sharpen_path(slug), "w", encoding="utf-8") as fh:
            json.dump(block, fh, indent=1)
    except OSError:
        pass


def _best_k(P: np.ndarray, Y: np.ndarray) -> float:
    """The exponent that minimises log-loss on these matches."""
    from scipy.optimize import minimize_scalar
    lo, hi = SHARPEN_BAND

    def obj(k):
        return log_loss(sharpen_probs(P, float(k)), Y)

    res = minimize_scalar(obj, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-3})
    return float(res.x)


def fit_sharpen(P: np.ndarray, Y: np.ndarray, slug: str = "") -> dict:
    """Fit the per-league sharpening exponent, and measure it honestly.

    Two numbers come out of this and they are not the same number.

    `k` is fitted on **every** match in the walk-forward -- all of them in the
    past, none of them in the season being forecast -- and is the one applied to
    the live forecast. `held_out` is the evidence that it is worth applying at
    all: an exponent fitted on the chronologically first 60% of the same
    matches, scored on the last 40%, which is a split the shipped number never
    sees. Reporting the in-sample gain alone would be a model marking its own
    homework, and shipping the 60% exponent would throw away four years of
    evidence to make a headline number look cleaner.
    """
    n = len(Y)
    band = {"lo": SHARPEN_BAND[0], "hi": SHARPEN_BAND[1]}
    if n < MIN_SHARPEN_N:
        return {"k": 1.0, "n": int(n), "band": band,
                "reason": f"{n} matches is below the {MIN_SHARPEN_N} needed to "
                          "fit a calibration exponent; left uncalibrated"}
    cut = int(n * SHARPEN_TRAIN)
    k_train = _best_k(P[:cut], Y[:cut])
    before = log_loss(P[cut:], Y[cut:])
    after = log_loss(sharpen_probs(P[cut:], k_train), Y[cut:])
    k_all = _best_k(P, Y)
    # The gate. An exponent fitted on every match in the walk-forward will
    # always improve the log-loss of those same matches -- that is what fitting
    # is -- so the only number that decides whether it ships is the one measured
    # on matches its own fit never saw. Where that is negative the league is
    # left alone: the Primeira Liga fits k = 1.10 in sample and *loses* 0.0033
    # out of it, which is a correction that has been measured not to
    # generalise, and applying it anyway would be the hidden thumb on the scale
    # this whole mechanism exists to avoid.
    ships = (before - after) > 0
    return {
        "k": round(k_all, 4) if ships else 1.0,
        "measured_k": round(k_all, 4),
        "applied": bool(ships),
        **({} if ships else
           {"reason": "the exponent fitted on the first 60% of these matches "
                      f"lost {after - before:.4f} of log-loss on the last 40%, "
                      "so this league ships uncalibrated"}),
        "n": int(n),
        "band": band,
        "in_sample": {"log_loss": round(log_loss(P, Y), 5),
                      "calibrated": round(log_loss(sharpen_probs(P, k_all), Y), 5)},
        "held_out": {"k": round(k_train, 4), "n": int(n - cut),
                     "train_n": int(cut),
                     "log_loss": round(before, 5),
                     "calibrated": round(after, 5),
                     "gain": round(before - after, 5)},
        "note": ("p -> p^k, renormalised. k is fitted on every match in this "
                 "walk-forward and applied to the live forecast; `held_out` is "
                 "the same fit made on the first 60% of these matches and "
                 "scored on the last 40%, which is the only number that says "
                 "whether the correction generalises."),
    }


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


def _score_masked(rows: list, P: np.ndarray, Y: np.ndarray,
                  extra_mask: np.ndarray | None = None) -> dict | None:
    """Score a baseline over exactly the matches it actually has a price for.

    The model's own score over the same subset rides along as `model_here`, so
    "the market beat us" is never a comparison between two different samples.
    """
    have = np.array([r is not None for r in rows])
    if extra_mask is not None:
        have = have & extra_mask
    if have.sum() < 100:
        return None
    B = np.array([r for r, ok in zip(rows, have) if ok], float)
    out = score_all(B, Y[have])
    out["coverage"] = round(float(have.sum() / max(len(rows), 1)), 3)
    out["model_here"] = score_all(P[have], Y[have])
    return out


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


#: Frozen external baselines: the closing bookmaker price and ClubElo's own
#: published rating, extracted once by `tools/extract_baselines.py`. Absent file
#: means those two baselines are simply not reported for that league.
BASELINE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "baselines")


def _snapshot_span(external: dict) -> dict:
    """First and last match date in the frozen baseline extract.

    The page says the snapshot "stops in June 2025". It stops when it stops, and
    the extract knows: quoting a date that was typed by hand is a claim that
    goes stale silently the first time somebody refreshes the file.
    """
    if not external:
        return {}
    days = sorted(k[0] for k in external)
    return {"from": days[0], "to": days[-1], "matches": len(days)}


def load_external(league) -> dict:
    """`{(date, home, away): row}` for one league, or {} when not extracted."""
    path = os.path.join(BASELINE_DIR, f"{league.slug}.json")
    if not os.path.exists(path):
        return {}
    try:
        doc = json.load(open(path))
    except (OSError, ValueError):
        return {}
    return {(r[0], r[1], r[2]): r for r in doc.get("rows", [])}


class Market:
    """The closing bookmaker price, de-vigged.

    The only baseline on this page that is genuinely hard to beat, and the
    reason it is here: everything else compares the model with something nobody
    would actually use. Bookmakers price with injuries, lineups, weather and
    money on the line, none of which this model can see, so beating them is not
    the expectation — knowing the size of the gap is the point.
    """
    name = "Closing bookmaker odds"

    def __init__(self, external: dict):
        self.ext = external

    def predict(self, m):
        row = self.ext.get((m.date.isoformat(), m.home, m.away))
        p = row[3] if row else None
        return np.array(p, float) if p else None


class ClubEloBaseline:
    """ClubElo's published rating, mapped to 1X2.

    A published, independently maintained rating — a much sterner test than the
    toy Elo above, which learns only from the same match list the model sees.
    The rating comes from the frozen snapshot; only the mapping from a rating
    difference to three probabilities is fitted here, and it is fitted purely on
    seasons *before* the backtest window so that no result is used to predict
    itself.
    """
    name = "ClubElo rating"

    def __init__(self, external: dict, train: list[Match]):
        self.ext = external
        self.home = 65.0
        self.draw_a, self.draw_b = 0.30, 1.0
        pairs = []
        for m in train:
            row = external.get((m.date.isoformat(), m.home, m.away))
            if row and row[4] and row[5] and m.hg is not None:
                pairs.append((row[4] - row[5], _result_index(m)))
        if len(pairs) < 500:
            return
        diff = np.array([p[0] for p in pairs], float)
        y = np.array([p[1] for p in pairs])
        best = 1e9
        for home in (40.0, 55.0, 65.0, 75.0, 90.0):
            e = 1.0 / (1.0 + 10 ** (-(diff + home) / 400.0))
            for a in np.arange(0.24, 0.35, 0.01):
                for b in np.arange(0.4, 2.4, 0.2):
                    p = self._curve(e, a, b)
                    ll = log_loss(p, y)
                    if ll < best:
                        best = ll
                        self.home, self.draw_a, self.draw_b = home, float(a), float(b)

    @staticmethod
    def _curve(e, a, b):
        d = np.clip(a - b * (e - 0.5) ** 2, 0.06, 0.40)
        hw = np.clip(e - d / 2, 1e-3, 1 - 1e-3)
        aw = np.clip(1 - hw - d, 1e-3, 1 - 1e-3)
        p = np.stack([hw, d, aw], -1)
        return p / p.sum(-1, keepdims=True)

    def predict(self, m):
        row = self.ext.get((m.date.isoformat(), m.home, m.away))
        if not row or not row[4] or not row[5]:
            return None
        e = 1.0 / (1.0 + 10 ** (-((row[4] - row[5]) + self.home) / 400.0))
        return self._curve(np.array([e]), self.draw_a, self.draw_b)[0]


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
        history_years: int = 5, quiet: bool = False,
        cal: dict | None = None, adj: bool = True) -> dict:
    """Walk forward through this league's past seasons and score every match.

    `adj` decides which of two models is scored. With it (the default) the
    prediction goes through `simulate._lambdas` carrying the same preseason
    correction the pipeline applies to the live forecast -- `priors.preseason_net`
    minus the fit's own centred net, recomputed at every refit exactly as a
    build recomputes it. Without it the prediction comes straight off the fit.

    It matters because until this existed the site published the accuracy of a
    model it does not run: the promoted-club and carryover corrections had never
    been scored out of sample even once, which is precisely why a Primeira Liga
    promoted-club slope of 2.01 measured over eight pairs shipped unnoticed.
    Both are reported -- `model` is what the site publishes, `model_ratings_only`
    is the fit alone -- so the correction has to earn its place every run.

    The market anchor, the other half of the live `adj`, is *not* here and
    cannot be: it is solved against a preseason odds snapshot, this repository
    holds exactly one of those (2026-27, Premier League), and there is no
    historical equivalent to walk forward over.
    """
    from . import priors

    all_seasons = sorted({m.season for m in ds.top if m.season != ds.season})
    seasons = seasons or [s for s in all_seasons if s >= ds.league.backtest_from]
    shot_conv = ratings.fit_shot_conversion(
        [m for m in ds.top if m.season < min(seasons)])
    if adj and cal is None:
        cal = priors.calibrate(ds, shot_conv)

    preds: list[np.ndarray] = []
    raw_preds: list[np.ndarray] = []
    ys: list[int] = []
    seasons_of: list[str] = []
    base_preds = {k: [] for k in ("base", "elo", "form", "market", "clubelo")}

    warm: dict | None = None
    train0 = [m for m in ds.top if m.season < min(seasons)]
    elo = Elo()
    elo.fit_draw(train0)
    form = SeasonForm(train0)
    for m in sorted(train0, key=lambda x: x.date):
        form.update(m)
    base = BaseRate(train0)
    # Two external baselines, from a frozen snapshot. Both may be absent -- for
    # a league nobody has extracted, or for matches after the snapshot's own
    # cut-off -- and a baseline scored on a different set of matches from the
    # model would be a comparison of two different questions. So each carries
    # its own mask and its own `n`, and the site prints both.
    external = load_external(ds.league)
    market = Market(external)
    clubelo = ClubEloBaseline(external, train0)

    for season in seasons:
        season_matches = sorted([m for m in ds.top if m.season == season],
                                key=lambda x: x.date)
        # The clubs in this division that season, and where each of them came
        # from -- which is all `preseason_net` needs to know which of the three
        # measured corrections applies to whom.
        season_teams = sorted({m.home for m in season_matches})
        prior_season = ([s for s in all_seasons if s < season] or [None])[-1]
        for start, chunk in _rounds(season_matches):
            cutoff = min(m.date for m in chunk)
            lo = dt.date(cutoff.year - history_years, 1, 1)
            hist = [m for m in ds.before(cutoff) if m.date >= lo]
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
            # The correction the pipeline would have applied on this date. Its
            # inputs are the fit, this season's clubs and the measured
            # regression -- all of them already here, which is the whole reason
            # this was worth doing rather than approximating.
            adj_map = None
            if adj and cal and prior_season:
                try:
                    raw_net = priors._centred_net(fit, season_teams)
                    prior_net = priors.preseason_net(ds, fit, cal, season_teams,
                                                     prior_season)
                    adj_map = {t: prior_net[t] - raw_net.get(t, 0.0)
                               for t in season_teams if t in raw_net}
                except (KeyError, ValueError):
                    adj_map = None
            for m in chunk:
                lh, la = fit.lambdas(m.home, m.away)
                ph, pd, pa = outcome_probs(score_matrix(lh, la, fit.rho))
                raw_preds.append(np.array([ph, pd, pa]))
                if adj_map:
                    alh, ala = _lambdas(fit, m.home, m.away, adj_map)
                    aph, apd, apa = outcome_probs(score_matrix(alh, ala, fit.rho))
                    preds.append(np.array([aph, apd, apa]))
                else:
                    preds.append(np.array([ph, pd, pa]))
                ys.append(_result_index(m))
                seasons_of.append(season)
                base_preds["base"].append(base.predict(m))
                base_preds["elo"].append(elo.predict(m))
                base_preds["form"].append(form.predict(m))
                base_preds["market"].append(market.predict(m))
                base_preds["clubelo"].append(clubelo.predict(m))
            for m in chunk:
                elo.update(m)
                form.update(m)
        if not quiet:
            print(f"  · {season}: {len(ys)} matches scored so far")

    P = np.array(preds)
    P0 = np.array(raw_preds)
    Y = np.array(ys)
    S = np.array(seasons_of)
    held = ~np.isin(S, TUNED_ON)
    sharp = fit_sharpen(P, Y, ds.league.slug)
    out = {"model": score_all(P, Y),
           # The same walk-forward with the preseason correction switched off.
           # `model` is the model the site publishes; this is the fit alone.
           "model_ratings_only": score_all(P0, Y),
           "adjusted": bool(adj),
           "adjustment_note": (
               "`model` carries the measured carryover / promotion / relegation "
               "correction the live forecast applies (priors.preseason_net), "
               "recomputed at every refit. `model_ratings_only` is the same "
               "walk-forward straight off the ratings. The market anchor is not "
               "included: it is solved against a preseason odds snapshot and no "
               "historical snapshot exists to walk forward over. One caveat "
               "the site should print: the regression `cal` itself is measured "
               "over every season this walk-forward scores, so the two "
               "constants it contributes are not strictly out of sample -- the "
               "per-match prediction is, and the correction's shape is two "
               "numbers over ~50 club-seasons."),
           "sharpen": sharp,
           "held_out": score_all(P[held], Y[held]) if held.any() else None,
           "held_out_ratings_only": (score_all(P0[held], Y[held])
                                     if held.any() else None),
           "tuned_on": list(TUNED_ON),
           "by_season": [
               {"season": s, **score_all(P[S == s], Y[S == s]),
                "held_out": s not in TUNED_ON}
               for s in seasons if (S == s).any()],
           "baselines": {k: v for k, v in
                         ((k, _score_masked(v, P, Y)) for k, v in base_preds.items())
                         if v is not None},
           "baselines_held_out": ({k: v for k, v in
                                   ((k, _score_masked(v, P, Y, held))
                                    for k, v in base_preds.items()) if v is not None}
                                  if held.any() else None),
           "names": {"base": BaseRate.name, "elo": Elo.name, "form": SeasonForm.name,
                     "market": Market.name, "clubelo": ClubEloBaseline.name},
           "external": {"source": "xgabora/Club-Football-Match-Data-2000-2025",
                        "url": ("https://github.com/xgabora/"
                                "Club-Football-Match-Data-2000-2025"),
                        "available": bool(external),
                        # The snapshot's own span, read off the extract rather
                        # than written into the page as a month that will be
                        # wrong the first time the file is refreshed.
                        **_snapshot_span(external)},
           "seasons": seasons,
           "params": {"goals_weight": goals_weight, "decay": decay,
                      "ridge": config.RIDGE,
                      # The calibration exponent the live forecast applies.
                      # Published so the method page can quote the number
                      # rather than describe the idea.
                      "sharpen": sharp["k"]}}
    out["calibration"] = calibration(P, Y)
    out["by_outcome"] = by_outcome(P, Y)
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
    # Shape parity with the domestic backtest, including the calibration
    # exponent -- which for two Swiss seasons of one competition comes back at
    # 1.0 with its reason, because 378 matches is not enough to fit one.
    sharp = fit_sharpen(P, Y, "european")
    return {"model": score_all(P, Y),
            "baselines": {"league_average": score_all(B, Y)},
            "names": {"league_average": LeagueAverage.name},
            "seasons": seasons, "comps": list(comps),
            "sharpen": sharp,
            # Shape parity with the domestic backtest. The pooled European fit
            # has no goals/shots blend, so that key is honestly absent.
            "params": {"decay": decay, "ridge": ridge, "sharpen": sharp["k"]},
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


def by_outcome(P: np.ndarray, Y: np.ndarray) -> list[dict]:
    """Calibration split three ways, because the pooled curve hides a draw bias.

    The overall reliability curve mixes home wins, draws and away wins, and a
    model that is 2 points light on draws and 1 point heavy on each of the other
    two looks perfectly calibrated in the pooled view: the errors cancel. Draws
    are the outcome a goals model is most likely to get wrong -- they are the
    thin middle of a distribution over two counts -- so they are worth reporting
    on their own rather than being averaged away.

    Reported, not corrected. A recalibration fitted on the same matches used to
    detect the bias would be fitting noise; if the gap is real it will still be
    here next season, measured on matches this one never saw.
    """
    names = ("home", "draw", "away")
    rows = []
    for k, name in enumerate(names):
        pk = P[:, k]
        ok = (Y == k).astype(float)
        rows.append({
            "outcome": name,
            "n": int(len(pk)),
            "predicted": round(float(pk.mean()), 4),
            "observed": round(float(ok.mean()), 4),
            # The mean gap in probability points, and how big it is next to the
            # scatter of a single match's outcome -- a bias of half a point on
            # 4,000 matches is not the same finding as one on 200.
            "bias": round(float(pk.mean() - ok.mean()), 4),
            "z": round(float((pk.mean() - ok.mean())
                             / max(1e-9, ok.std(ddof=1) / np.sqrt(len(ok)))), 2),
        })
    return rows


# --------------------------------------------------------------------------
# Scoring the table, not just the matches
# --------------------------------------------------------------------------
def table_accuracy(ds: Dataset, seasons: list[str], shot_conv,
                   *, quiet: bool = True) -> dict:
    """How good was the preseason projection of the final table?

    Match log-loss is the right way to score a forecast and the wrong way to
    argue about one. Nobody has an intuition for 0.97, and the number the model
    is actually judged on in public is whether it called the season.

    So: at the start of each past season, fit on what was knowable then, project
    every fixture, sum expected points, and compare that ordering with what
    happened. Reported as the champion hit rate, the share of the qualifying and
    relegation places called correctly, the rank correlation and the mean points
    error -- none of which the model has ever been allowed to see.
    """
    from .priors import _season_start
    rows = []
    for season in seasons:
        ref = _season_start(season)
        hist = [m for m in ds.before(ref)]
        played = [m for m in ds.top if m.season == season]
        teams = sorted({m.home for m in played})
        n = len(teams)
        if len(hist) < 2000 or n < 16 or len(played) != n * (n - 1):
            continue
        pool = sorted({m.home for m in hist} | {m.away for m in hist})
        if not set(teams) <= set(pool):
            continue
        fit = ratings.fit(hist, pool, ref, shot_conv=shot_conv)
        exp = {t: 0.0 for t in teams}
        for m in played:
            lh, la = fit.lambdas(m.home, m.away)
            ph, pd, pa = outcome_probs(score_matrix(lh, la, fit.rho))
            exp[m.home] += 3 * ph + pd
            exp[m.away] += 3 * pa + pd
        actual = {t: 0 for t in teams}
        for m in played:
            if m.hg > m.ag:
                actual[m.home] += 3
            elif m.hg == m.ag:
                actual[m.home] += 1
                actual[m.away] += 1
            else:
                actual[m.away] += 3
        proj = sorted(teams, key=lambda t: -exp[t])
        real = sorted(teams, key=lambda t: -actual[t])
        p_rank = {t: i for i, t in enumerate(proj)}
        r_rank = {t: i for i, t in enumerate(real)}
        d = np.array([p_rank[t] - r_rank[t] for t in teams], float)
        # Spearman on ranks is Pearson on the rank vectors.
        pr = np.array([p_rank[t] for t in teams], float)
        rr = np.array([r_rank[t] for t in teams], float)
        rho = float(np.corrcoef(pr, rr)[0, 1]) if n > 2 else 0.0
        ucl = ds.league.ucl_places
        rel = ds.league.releg_places
        rows.append({
            "season": season, "n_teams": n,
            "champion_called": proj[0] == real[0],
            "top_overlap": len(set(proj[:ucl]) & set(real[:ucl])) / max(ucl, 1),
            "releg_overlap": len(set(proj[-rel:]) & set(real[-rel:])) / max(rel, 1),
            "rank_corr": round(rho, 4),
            "mean_rank_error": round(float(np.abs(d).mean()), 2),
            "mean_points_error": round(
                float(np.mean([abs(exp[t] - actual[t]) for t in teams])), 2),
        })
        if not quiet:
            print(f"  · {season}: champion "
                  f"{'hit' if rows[-1]['champion_called'] else 'miss'}, "
                  f"rank corr {rho:.2f}")
    if not rows:
        return {}
    return {
        "seasons": rows,
        "n": len(rows),
        "champion_rate": round(sum(r["champion_called"] for r in rows) / len(rows), 3),
        "top_overlap": round(float(np.mean([r["top_overlap"] for r in rows])), 3),
        "releg_overlap": round(float(np.mean([r["releg_overlap"] for r in rows])), 3),
        "rank_corr": round(float(np.mean([r["rank_corr"] for r in rows])), 3),
        "mean_rank_error": round(float(np.mean([r["mean_rank_error"] for r in rows])), 2),
        "mean_points_error": round(
            float(np.mean([r["mean_points_error"] for r in rows])), 2),
    }
