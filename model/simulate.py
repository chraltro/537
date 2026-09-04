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


#: The three outcome classes of a score grid, as an index into the flattened
#: (MAXG+1)x(MAXG+1) matrix: 0 home win, 1 draw, 2 away win. Precomputed
#: because the season simulation reweights by it once per scenario.
_GI, _GJ = np.meshgrid(np.arange(MAXG + 1), np.arange(MAXG + 1), indexing="ij")
OUTCOME_CLASS = np.where(_GI > _GJ, 0, np.where(_GI == _GJ, 1, 2))


def sharpen_probs(p, k: float):
    """`p ** k`, renormalised: the one-parameter calibration of a 1X2 forecast.

    k > 1 pushes probabilities away from the base rate (the model is
    under-confident), k < 1 pulls them toward it (over-confident), k = 1 is the
    identity. It is the only family that can fix the shape the reliability
    curves in `backtest.calibration` actually show -- every bin above 0.5
    under-predicted and the 0.1-0.2 bin over-predicted -- with a single number
    that costs nothing at prediction time and is visible in the published
    backtest rather than being a hidden thumb on the scale.

    Fitted per league, on that league's own earlier seasons: the Premier League
    wants 1.2 and the Belgian Pro League wants 0.8, so one global constant
    would help one and hurt the other. See `backtest.fit_sharpen`.
    """
    if k == 1.0:
        return p
    q = np.asarray(p, float) ** float(k)
    return q / q.sum(axis=-1, keepdims=True)


def sharpen_grid(m: np.ndarray, k: float) -> np.ndarray:
    """Reweight a score grid so its 1X2 margin is `sharpen_probs(p, k)`.

    The calibration is measured on match outcomes, so it is applied to those
    three numbers and the conditional distribution of scorelines *within* each
    outcome is left exactly as the Dixon-Coles grid had it. That keeps the grid,
    the top scorelines, over-2.5 and both-teams-to-score consistent with the
    win/draw/loss probabilities printed next to them, which they would not be
    if the three were sharpened and the grid were not.
    """
    if k == 1.0:
        return m
    cls = OUTCOME_CLASS[:m.shape[0], :m.shape[1]]
    p = np.array([float(m[cls == c].sum()) for c in range(3)])
    q = sharpen_probs(p, k)
    scale = np.where(p > 1e-12, q / np.maximum(p, 1e-12), 0.0)
    out = m * scale[cls]
    return out / out.sum()


def match_report(fit: Fit, home: str, away: str, adj: dict[str, float] | None = None,
                 *, sharpen: float = 1.0) -> dict:
    lh, la = _lambdas(fit, home, away, adj)
    m = sharpen_grid(score_matrix(lh, la, fit.rho), sharpen)
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
                    rating_sd=config.RATING_SD,
                    seed: int = config.SEED,
                    scenarios: int = 200,
                    leverage: bool = False,
                    events: tuple[str, ...] | None = None,
                    keep_orders: bool = False,
                    sharpen: float = 1.0,
                    curves: bool = True) -> dict:
    """Play the rest of the season `n_sims` times.

    Results already on the board are carried in as fact; only the remaining
    fixtures are simulated. `league` decides where the European and relegation
    lines fall; it defaults to the Premier League for callers with none in hand.

    `rating_sd` may be a scalar or one value per club, which is how a club whose
    freshest result is fifteen months old gets a wider interval than one playing
    every week. `events` names the three leverage events; a cup's league phase
    swings qualification rather than the title, so it passes `CUP_EVENTS`.

    `sharpen` is the per-league calibration exponent (see `sharpen_probs`),
    applied to each simulated fixture's outcome probabilities so the table the
    simulation produces is built from the same numbers the match pages print.

    `curves` additionally tallies, for every club, how often each of the three
    events happened *conditional on the club's own final points total*. That is
    the "how many do we need" question, and counting it inside the loop that
    already sorts every table costs four bincounts per scenario.
    """
    lg = league or leagues.DEFAULT
    ucl, europa, releg = lg.ucl_places, lg.europa_places, lg.releg_places
    events = events or EVENTS
    # `rating_sd` stays exactly as the caller passed it into `rng.normal`: a
    # scalar and a constant vector do NOT draw the same numbers from numpy's
    # generator, and switching one for the other would move every published
    # forecast for no reason.
    any_sd = bool(np.any(np.asarray(rating_sd, float) > 0))
    # The three leverage lines, as finishing positions. A domestic league swings
    # the title, the European places and relegation; a cup league phase swings
    # direct qualification, qualification at all, and elimination.
    if lg.kind == "cup" and lg.advance_direct and lg.advance_playoff:
        lev_top = lg.advance_direct
        lev_qual = lg.advance_direct + lg.advance_playoff
        lev_out = lg.n_teams - lev_qual
    else:
        lev_top, lev_qual, lev_out = 1, ucl, releg
    rng = np.random.default_rng(seed)
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    # Spain and Italy separate clubs level on points by the mini-table among
    # those clubs, not by goal difference. That needs a running tally of the
    # points each club has taken off each other club, which is cheap to keep and
    # was simply never kept.
    h2h_rule = lg.tiebreak == "h2h"
    base_h2h = np.zeros((n, n)) if h2h_rule else None

    base_pts = np.zeros(n)
    base_gf = np.zeros(n)
    base_ga = np.zeros(n)
    remaining = []
    for f in fixtures:
        i, j = idx[f.home], idx[f.away]
        if f.played:
            base_gf[i] += f.hg; base_ga[i] += f.ag
            base_gf[j] += f.ag; base_ga[j] += f.hg
            hp = 3 if f.hg > f.ag else (1 if f.hg == f.ag else 0)
            ap = 3 if f.ag > f.hg else (1 if f.hg == f.ag else 0)
            base_pts[i] += hp
            base_pts[j] += ap
            if h2h_rule:
                base_h2h[i, j] += hp
                base_h2h[j, i] += ap
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
    # The finishing order of every simulated season, kept only when a caller
    # needs to play something on top of the table -- a knockout bracket has to
    # be redrawn per simulated season or a club's route is independent of where
    # it finished, which is the opposite of the truth.
    orders_out = (np.empty((scenarios * per, n), dtype=np.int16)
                  if keep_orders else None)
    # Which scenario -- which draw of "how good is everyone really" -- each
    # simulated season came from, and the draws themselves. A knockout bracket
    # played on top of these tables has to use the same rating shock the table
    # was produced under, or the club that finished top because its true rating
    # is higher goes into the last 16 at its point estimate again and four
    # rounds compound a certainty the league phase never had.
    scen_of = (np.empty(scenarios * per, dtype=np.int32)
               if keep_orders else None)
    shocks_out = (np.zeros((scenarios, n)) if keep_orders else None)
    gd_sum = np.zeros(n)
    # Conditional tallies for match importance: for every remaining fixture and
    # every possible result, how often each club ends up champion / in the top
    # five / relegated. Counting this inside the existing simulation is nearly
    # free, and it answers the question a fan actually has -- does this game
    # matter? -- without a second set of conditional runs.
    n_ev = 3
    lev_hits = np.zeros((3, len(remaining), n * n_ev), dtype=np.float64) if leverage else None
    lev_n = np.zeros((3, len(remaining)), dtype=np.float64) if leverage else None
    # Points-conditional tallies: how often each club ends up champion / in the
    # qualifying places / relegated, given the points it finished on. Same idea
    # as leverage -- the counting is nearly free next to the sort that is
    # already happening, and it answers a question no average can.
    per_team_games = (2 * len(fixtures)) // n if n else 0
    n_pts = 3 * per_team_games + 1
    cur_hits = np.zeros((n * n_pts, n_ev)) if curves else None
    cur_n = np.zeros(n * n_pts) if curves else None
    team_off = np.arange(n) * n_pts
    kk = np.arange(MAXG + 1)
    lgam = np.array([0.0] + list(np.cumsum(np.log(np.arange(1, MAXG + 1)))))
    cls_flat = OUTCOME_CLASS.ravel()
    cursor = 0

    for scen in range(scenarios):
        # One draw of "how good is everyone really", held fixed for `per` seasons.
        shock = rng.normal(0.0, rating_sd, n) if any_sd else np.zeros(n)
        if shocks_out is not None:
            shocks_out[scen] = shock
        lh = lh0 * np.exp(shock[hi] / 2 - shock[ai] / 2)
        la = la0 * np.exp(shock[ai] / 2 - shock[hi] / 2)

        if len(remaining):
            ph = np.exp(-lh[:, None] + kk[None, :] * np.log(lh)[:, None] - lgam[None, :])
            pa = np.exp(-la[:, None] + kk[None, :] * np.log(la)[:, None] - lgam[None, :])
            grid = ph[:, :, None] * pa[:, None, :]
            gi, gj = np.meshgrid(kk, kk, indexing="ij")
            grid *= tau(gi, gj, lh[:, None, None], la[:, None, None], fit.rho)
            flat = grid.reshape(len(remaining), -1)
            flat /= flat.sum(axis=1, keepdims=True)
            if sharpen != 1.0:
                # Same reweighting as `sharpen_grid`, vectorised over every
                # remaining fixture at once: the 1X2 margin is sharpened and
                # the scoreline distribution inside each outcome is untouched.
                p3 = np.stack([flat[:, cls_flat == c].sum(axis=1)
                               for c in range(3)], axis=1)
                q3 = sharpen_probs(p3, sharpen)
                sc = np.where(p3 > 1e-12, q3 / np.maximum(p3, 1e-12), 0.0)
                flat = flat * sc[:, cls_flat]
                flat /= flat.sum(axis=1, keepdims=True)
            cdf = np.cumsum(flat, axis=1)

            u = rng.random((per, len(remaining)))
            pick = np.array([np.searchsorted(cdf[m], u[:, m])
                             for m in range(len(remaining))]).T
            hg = (pick // (MAXG + 1)).astype(np.int16)
            ag = (pick % (MAXG + 1)).astype(np.int16)
        else:
            # A finished league phase. The table is settled and only the
            # tie-break below is still random -- which is the state the
            # Champions League is in from the end of matchday 8 until the final,
            # and the state every domestic league reaches in May.
            hg = ag = np.zeros((per, 0), dtype=np.int16)

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
        # Points, then goal difference, then goals scored -- and, where the
        # league says so, the mini-table among the clubs level on points before
        # goal difference is looked at.
        #
        # That mini-table is the whole reason the pairwise tally exists: a club's
        # tie-break score is the points it took off the clubs it is level with,
        # which is `same` (who is on my points) masked over `h2h` (what I took
        # off them). Vectorised, it costs one (per, n, n) product per scenario.
        # It is the primary criterion rather than the full recursive rule --
        # Spain and Italy then go to head-to-head goal difference -- and the
        # method page says which part is modelled.
        if h2h_rule:
            h2h = np.tile(base_h2h.ravel(), (per, 1))
            if len(remaining):
                np.add.at(h2h, (slice(None), hi * n + ai), hw)
                np.add.at(h2h, (slice(None), ai * n + hi), aw)
            h2h = h2h.reshape(per, n, n)
            same = (pts[:, :, None] == pts[:, None, :])
            mini_key = (h2h * same).sum(axis=2)
        else:
            mini_key = np.zeros((per, n))
        # A genuine tie is settled by a play-off, so break it at random here --
        # in a lexsort rather than in one packed float64 key. The packed key was
        # `pts * 1e12 + ... + rng.random(...) * 1e-3`, and at 1.14e14 the ULP is
        # 0.0156: the random term was three orders of magnitude below the
        # smallest representable difference and was discarded entirely, so every
        # dead-level tie went to `np.argsort`'s stable order, which is the order
        # of `teams`, which is alphabetical by club id. Sorting on the columns
        # themselves has no precision to lose.
        rand = rng.random((per, n))
        # Ascending by the last key first, so reverse for the finishing order.
        order = np.ascontiguousarray(
            np.lexsort((rand, gf, gd, mini_key, pts), axis=1)[:, ::-1])
        # bincount, not fancy-index +=: repeated (team, position) pairs across the
        # seasons in this scenario must accumulate, and `+= 1` would count each once.
        flat_idx = order.ravel() * n + np.tile(np.arange(n), per)
        pos_counts += np.bincount(flat_idx, minlength=n * n).reshape(n, n)
        pts_team[cursor:cursor + per] = pts
        pos_pts[cursor:cursor + per] = np.take_along_axis(pts, order, axis=1)
        if orders_out is not None:
            orders_out[cursor:cursor + per] = order
            scen_of[cursor:cursor + per] = scen
        gd_sum += gd.sum(axis=0)
        cursor += per

        if curves or (leverage and len(remaining)):
            rank = np.argsort(order, axis=1)            # rank[s, t] = finish (0-based)
            ev = np.empty((per, n, n_ev), dtype=np.float32)
            ev[:, :, 0] = rank < lev_top
            ev[:, :, 1] = rank < lev_qual
            ev[:, :, 2] = rank >= n - lev_out

        if curves:
            # bin index = team * n_pts + that team's final points
            bins = (team_off[None, :] + np.clip(pts, 0, n_pts - 1).astype(np.int64)).ravel()
            cur_n += np.bincount(bins, minlength=n * n_pts)
            for e in range(n_ev):
                cur_hits[:, e] += np.bincount(bins, weights=ev[:, :, e].ravel(),
                                              minlength=n * n_pts)

        if leverage and len(remaining):
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
        "events": list(events),
        "lines": _lines(pos_pts[:total], lg),
        "curves": (_curves(cur_hits, cur_n, teams, n_pts, total, events)
                   if curves else None),
        "orders": orders_out[:total] if orders_out is not None else None,
        "scenario": scen_of[:total] if scen_of is not None else None,
        "shocks": shocks_out,
        "leverage": (_leverage(lev_hits, lev_n, remaining, teams, events)
                     if leverage else None),
        # The same conditional tallies the leverage score is squeezed out of,
        # kept as probabilities: P(club c reaches event e | this match ends
        # home / draw / away). `rooting.json` is the rest of that question --
        # what a supporter of a club NOT playing should want to happen -- and
        # recomputing it would mean a second set of conditional simulations.
        "conditional": (_conditional(lev_hits, lev_n, remaining, teams, events)
                        if leverage else None),
        "unconditional": (p[:, :lev_top].sum(axis=1),
                          p[:, :lev_qual].sum(axis=1),
                          p[:, n - lev_out:].sum(axis=1)) if leverage else None,
        "remaining": [(f.home, f.away, f.matchday, f.date.isoformat())
                      for f in remaining] if leverage else None,
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


#: A points bin is only published once enough simulated seasons landed in it to
#: mean something. Below this the curve is one season's noise drawn as a fact.
MIN_CURVE_SEASONS = 40


def _curves(hits, counts, teams, n_pts, total, events) -> list[dict]:
    """"How many points do we need?", per club, read off the same simulations.

    For each club, the points totals it actually reached across the simulated
    seasons, and how often each of the three events happened when it finished
    on that many. Thin bins are dropped rather than smoothed: a bin holding
    three seasons out of fifty thousand is not a probability, it is an anecdote.
    """
    out = []
    for i, t in enumerate(teams):
        lo, hi = i * n_pts, (i + 1) * n_pts
        cnt = counts[lo:hi]
        keep = np.nonzero(cnt >= MIN_CURVE_SEASONS)[0]
        if not len(keep):
            out.append({"id": t, "pts": [], "n": [],
                        **{e: [] for e in events}})
            continue
        row = {"id": t,
               "pts": [int(p) for p in keep],
               "n": [int(cnt[p]) for p in keep]}
        for e_i, name in enumerate(events):
            row[name] = [round(float(hits[lo + p, e_i] / cnt[p]), 4) for p in keep]
        out.append(row)
    return out


def _conditional(hits, counts, remaining, teams, events):
    """P(event | result) for every remaining fixture, club and event.

    Shape (3, n_remaining, n_teams, 3): the first axis is the match result
    (home win / draw / away win), the last the three events. A bin with no
    simulated seasons in it -- a result a match essentially cannot produce --
    comes back as `nan` and not as zero, because the honest reading of "this
    never happened in fifty thousand seasons" is "this tells us nothing", and a
    caller that treated it as a probability would publish a swing of the club's
    whole title chance off an empty cell. `run.write_rooting` drops them.
    """
    if hits is None or not len(remaining):
        return None
    n = len(teams)
    with np.errstate(invalid="ignore", divide="ignore"):
        probs = hits / np.maximum(counts, 1)[:, :, None]
    probs = probs.reshape(3, len(remaining), n, len(events))
    empty = counts <= 0
    if empty.any():
        for k in range(3):
            for m in np.nonzero(empty[k])[0]:
                probs[k, m] = np.nan
    return probs


#: Leverage events. Domestic leagues swing title/UCL/relegation; a cup's
#: league phase swings direct qualification / the play-off cut / elimination.
#: The keys are part of the site contract -- the front end maps them to words.
EVENTS = ("title", "ucl", "releg")
CUP_EVENTS = ("top8", "qualify", "out")


def _leverage(hits, counts, remaining, teams,
              events: tuple[str, ...] = EVENTS) -> list[dict]:
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
    probs = probs.reshape(3, len(remaining), n, len(events))
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
            t_i, e_i = divmod(int(f), len(events))
            swings.append({"team": teams[t_i], "event": events[e_i],
                           "home": round(float(probs[0, m, t_i, e_i]), 4),
                           "away": round(float(probs[2, m, t_i, e_i]), 4),
                           "swing": round(float(swing[t_i, e_i]), 4)})
        out.append({"score": round(float(flat.max()), 4), "swings": swings})
    return out
