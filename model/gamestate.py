"""What the results feed says beyond the final score.

Every football-data mirror CSV has carried eleven columns this project never
read: the half-time score, corners, fouls, cards and the referee's name. They
are worth reading for two different reasons.

*Half time is a second, cheaper season.* Fitting the same Dixon-Coles model to
half-time goals costs one extra fit and answers questions the full-time model
cannot: how often a club leads at the break, and how often that lead survives.
A club that leads at half-time in half its matches and wins two thirds of those
is telling you something a points total hides.

*Discipline is a real signal with a fake-looking one next to it.* Cards and
fouls per match are stable club properties. Referee card rates are not the same
kind of fact: a referee who books more is usually working more physical
fixtures, so the numbers here are a record of what happened in that referee's
matches, adjusted for nothing, and the site says exactly that. Nothing in this
module feeds the forecast -- no fixture list anywhere on GitHub publishes the
referee in advance, so a referee term could never be applied to a match that
has not been played.

Coverage, measured across all 33 season files per league:

* half-time goals -- complete from 2000-01 in all five leagues
* cards, corners, fouls -- complete from 2000-01 in all five leagues
* referee -- effectively Premier League only. Counted season by season across
  every file in the mirror: 26 seasons carry it in `premier-league`, two in
  `bundesliga` (2000-01, 2001-02), two in `serie-a` (2005-06, 2006-07) and none
  in `la-liga` or `ligue-1`. So `referees()` returns an empty list for four of
  the five, and the site hides the section rather than printing a stub.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import replace

from .parse import Match

#: Club-level game-state and discipline rates are read off recent seasons only.
#: A club's discipline in 2004 says nothing about the side taking the field now,
#: and averaging twenty-five years of it would mostly measure squad turnover.
RECENT_SEASONS = 5

#: A referee needs a real record before a card rate means anything.
MIN_REFEREE_MATCHES = 30
MAX_REFEREES = 40


def _recent(matches: list[Match], seasons: int = RECENT_SEASONS) -> list[Match]:
    labels = sorted({m.season for m in matches})[-seasons:]
    keep = set(labels)
    return [m for m in matches if m.season in keep]


# --------------------------------------------------------------------------
# Half time
# --------------------------------------------------------------------------
def half_time_matches(matches: list[Match]) -> list[Match]:
    """The same matches, with the half-time score standing in for the result.

    A shallow copy per match rather than a mutation: the caller's list is the
    one the full-time rating is fitted on, and swapping goals underneath it
    would be the most expensive kind of quiet bug.
    """
    out = []
    for m in matches:
        if m.hthg is None or m.htag is None:
            continue
        out.append(replace(m, hg=m.hthg, ag=m.htag,
                           hs=None, as_=None, hst=None, ast=None))
    return out


def half_time_fit(matches: list[Match], ref_date: dt.date):
    """A Dixon-Coles fit on half-time goals, or None if the feed has none.

    Goals only, deliberately: shot counts are full-match totals and cannot be
    split across the interval, so blending them in would attribute a second-half
    shot to the first half.
    """
    from . import ratings
    ht = half_time_matches(matches)
    if len(ht) < 2000:
        return None
    pool = sorted({m.home for m in ht} | {m.away for m in ht})
    # goals_weight=1.0 -> no shot blend; there is nothing to blend with.
    return ratings.fit(ht, pool, ref_date, goals_weight=1.0)


def half_time_report(fit, home: str, away: str, adj=None) -> dict | None:
    """Half-time lead probabilities and the likeliest half-time score."""
    if fit is None or home not in fit.index or away not in fit.index:
        return None
    from .simulate import _lambdas, outcome_probs, score_matrix
    lh, la = _lambdas(fit, home, away, adj)
    m = score_matrix(lh, la, fit.rho, maxg=6)
    ph, pd, pa = outcome_probs(m)
    flat = m.flatten()
    best = int(flat.argmax())
    return {"ph": ph, "pd": pd, "pa": pa,
            "sc": [best // m.shape[1], best % m.shape[1]],
            "scp": float(flat[best])}


# --------------------------------------------------------------------------
# Game state: leads, comebacks, and which half a club plays in
# --------------------------------------------------------------------------
def game_state(matches: list[Match], teams) -> dict[str, dict]:
    """Per club, what happens either side of the interval.

    Counted over the last few seasons of the club's top-flight record, so a
    promoted club's row covers its second-tier seasons only if the caller passed
    them in -- which it does not, and that is why `n` is published alongside
    every rate.
    """
    blank = {"n": 0, "led": 0, "led_won": 0, "level": 0, "level_won": 0,
             "behind": 0, "behind_saved": 0, "behind_won": 0,
             "gf1": 0, "gf2": 0, "ga1": 0, "ga2": 0}
    out = {t: dict(blank) for t in teams}
    for m in _recent(matches):
        if m.hthg is None or m.htag is None or m.hg is None or m.ag is None:
            continue
        for t, hf, ha, ff, fa in ((m.home, m.hthg, m.htag, m.hg, m.ag),
                                  (m.away, m.htag, m.hthg, m.ag, m.hg)):
            r = out.get(t)
            if r is None:
                continue
            r["n"] += 1
            r["gf1"] += hf
            r["ga1"] += ha
            r["gf2"] += ff - hf
            r["ga2"] += fa - ha
            won, drew = ff > fa, ff == fa
            if hf > ha:
                r["led"] += 1
                r["led_won"] += won
            elif hf == ha:
                r["level"] += 1
                r["level_won"] += won
            else:
                r["behind"] += 1
                r["behind_saved"] += won or drew
                r["behind_won"] += won
    for r in out.values():
        n = max(r["n"], 1)
        r["led_pct"] = round(r["led"] / n, 3)
        r["behind_pct"] = round(r["behind"] / n, 3)
        r["hold_pct"] = round(r["led_won"] / r["led"], 3) if r["led"] else None
        r["recover_pct"] = (round(r["behind_saved"] / r["behind"], 3)
                            if r["behind"] else None)
        r["first_half_gd"] = round((r["gf1"] - r["ga1"]) / n, 2)
        r["second_half_gd"] = round((r["gf2"] - r["ga2"]) / n, 2)
    return out


# --------------------------------------------------------------------------
# Discipline
# --------------------------------------------------------------------------
def discipline(matches: list[Match], teams) -> dict[str, dict]:
    """Cards, fouls and corners per match, for and against, per club."""
    blank = {"n": 0, "yellow": 0, "red": 0, "fouls": 0, "corners": 0,
             "yellow_against": 0, "fouls_against": 0, "corners_against": 0}
    out = {t: dict(blank) for t in teams}
    for m in _recent(matches):
        if m.hy is None or m.ay is None:
            continue
        for t, y, r_, f, c, y2, f2, c2 in (
                (m.home, m.hy, m.hr, m.hf, m.hc, m.ay, m.af, m.ac),
                (m.away, m.ay, m.ar, m.af, m.ac, m.hy, m.hf, m.hc)):
            row = out.get(t)
            if row is None:
                continue
            row["n"] += 1
            row["yellow"] += y or 0
            row["red"] += r_ or 0
            row["fouls"] += f or 0
            row["corners"] += c or 0
            row["yellow_against"] += y2 or 0
            row["fouls_against"] += f2 or 0
            row["corners_against"] += c2 or 0
    for row in out.values():
        n = max(row["n"], 1)
        for k in ("yellow", "red", "fouls", "corners",
                  "yellow_against", "fouls_against", "corners_against"):
            row[f"{k}_pm"] = round(row[k] / n, 2)
    return out


def referees(matches: list[Match]) -> list[dict]:
    """One row per referee with a real record, busiest first.

    Deliberately *not* a model of a referee effect. A referee is assigned to
    fixtures, not drawn at random, so a high card rate may be a hard beat rather
    than a strict official — and no source we can reach publishes appointments
    before kick-off anyway, so nothing here could be applied to a fixture that
    has not happened. It is a record of what occurred in their matches.
    """
    tally: dict[str, dict] = {}
    for m in _recent(matches, seasons=8):
        if not m.referee or m.hy is None or m.ay is None:
            continue
        r = tally.setdefault(m.referee, {
            "name": m.referee, "n": 0, "yellow": 0, "red": 0, "fouls": 0,
            "home_wins": 0, "draws": 0, "goals": 0, "pens": 0})
        r["n"] += 1
        r["yellow"] += (m.hy or 0) + (m.ay or 0)
        r["red"] += (m.hr or 0) + (m.ar or 0)
        r["fouls"] += (m.hf or 0) + (m.af or 0)
        r["goals"] += (m.hg or 0) + (m.ag or 0)
        if m.hg is not None and m.ag is not None:
            r["home_wins"] += m.hg > m.ag
            r["draws"] += m.hg == m.ag
    rows = [r for r in tally.values() if r["n"] >= MIN_REFEREE_MATCHES]
    for r in rows:
        n = r["n"]
        r["yellow_pm"] = round(r["yellow"] / n, 2)
        r["red_pm"] = round(r["red"] / n, 3)
        r["fouls_pm"] = round(r["fouls"] / n, 1)
        r["goals_pm"] = round(r["goals"] / n, 2)
        r["home_win_pct"] = round(r["home_wins"] / n, 3)
        r["draw_pct"] = round(r["draws"] / n, 3)
    rows.sort(key=lambda r: -r["n"])
    return rows[:MAX_REFEREES]


def league_average(matches: list[Match], seasons: int = 8) -> dict:
    """The bar every referee and every club row is read against."""
    sel = [m for m in _recent(matches, seasons)
           if m.hy is not None and m.ay is not None]
    n = max(len(sel), 1)
    hw = sum(1 for m in sel if m.hg is not None and m.ag is not None and m.hg > m.ag)
    dr = sum(1 for m in sel if m.hg is not None and m.ag is not None and m.hg == m.ag)
    return {
        "n": len(sel),
        "yellow_pm": round(sum((m.hy or 0) + (m.ay or 0) for m in sel) / n, 2),
        "red_pm": round(sum((m.hr or 0) + (m.ar or 0) for m in sel) / n, 3),
        "fouls_pm": round(sum((m.hf or 0) + (m.af or 0) for m in sel) / n, 1),
        "goals_pm": round(sum((m.hg or 0) + (m.ag or 0) for m in sel) / n, 2),
        "home_win_pct": round(hw / n, 3),
        "draw_pct": round(dr / n, 3),
    }


def build(matches: list[Match], teams, ref_date: dt.date) -> dict:
    """The whole `gamestate.json` payload for one league."""
    return {
        "seasons": sorted({m.season for m in _recent(matches)}),
        "average": league_average(matches),
        "state": game_state(matches, teams),
        "discipline": discipline(matches, teams),
        "referees": referees(matches),
    }
