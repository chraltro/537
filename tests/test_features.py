"""Tests for everything added after the Champions League landed.

The theme is the same as the rest of the suite: the failure modes that matter
here are silent ones. An expected-points column that quietly counts matches the
frozen forecast never covered, a points-conditional curve normalised against the
wrong denominator, a calendar that emits a malformed date, a share card that
walks off the end of an ordinal list on a 24-club division — none of those throw,
and all of them ship a wrong number to a reader who has no way to check it.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import (europe, feeds, gamestate, insight, leagues,  # noqa: E402
                   rankings, simulate, social)
from model.parse import Match, TeamRegistry, parse_openfootball  # noqa: E402
from model.ratings import Fit                                    # noqa: E402


# ------------------------------------------------------------------ expected points
def _m(h, a, hg, ag, played=True):
    return {"h": h, "a": a, "hg": hg, "ag": ag, "played": played}


def test_expected_points_uses_only_frozen_forecasts():
    """A match with no pre-kick-off forecast on record contributes to neither
    side of the comparison -- otherwise a club's actual points would cover more
    matches than its expectation and the gap would be an artefact."""
    matches = [_m("a", "b", 1, 0), _m("b", "a", 2, 2), _m("a", "c", 0, 1)]
    frozen = {
        "a|b": {"ph": 0.5, "pd": 0.3, "pa": 0.2},
        "b|a": {"ph": 0.4, "pd": 0.3, "pa": 0.3},
        # no entry for a|c: played before the pipeline ever saw it
    }
    out = insight.expected_points(matches, frozen, ["a", "b", "c"])
    assert out["a"]["played"] == 2 and out["b"]["played"] == 2
    assert out["c"]["played"] == 0 and out["c"]["xp"] == 0.0
    # a: won at home (3) then drew away (1) = 4 actual
    assert out["a"]["pts"] == 4
    # expected: (3*0.5 + 0.3) + (3*0.3 + 0.3) = 1.8 + 1.2 = 3.0
    assert out["a"]["xp"] == pytest.approx(3.0, abs=1e-9)
    assert out["a"]["diff"] == pytest.approx(1.0, abs=1e-9)
    assert out["a"]["per_match"] == pytest.approx(0.5, abs=1e-3)


def test_expected_points_conserves_points():
    """Actual and expected points must both be conserved across a match: three
    for a decisive result, two for a draw, on both sides of the ledger."""
    matches = [_m("a", "b", 3, 1), _m("a", "b", 1, 1)]
    frozen = {"a|b": {"ph": 0.55, "pd": 0.25, "pa": 0.20}}
    out = insight.expected_points(matches[:1], frozen, ["a", "b"])
    assert out["a"]["pts"] + out["b"]["pts"] == 3
    assert out["a"]["xp"] + out["b"]["xp"] == pytest.approx(3 * 0.75 + 2 * 0.25, abs=1e-6)
    drawn = insight.expected_points(matches[1:], frozen, ["a", "b"])
    assert drawn["a"]["pts"] == 1 and drawn["b"]["pts"] == 1


# ------------------------------------------------------------------ scenario curves
def _toy_fixtures(teams, n_rounds=2):
    out = []
    day = dt.date(2026, 8, 1)
    for r in range(n_rounds):
        for i in range(0, len(teams), 2):
            h, a = (teams[i], teams[i + 1]) if r % 2 == 0 else (teams[i + 1], teams[i])
            out.append(Match(date=day + dt.timedelta(days=7 * r), home=h, away=a,
                             matchday=r + 1, played=False))
    return out


def _toy_fit(teams):
    n = len(teams)
    return Fit(teams, np.zeros(n), np.zeros(n), np.log(1.35), 0.2, 0.0)


def test_points_curves_are_probabilities_over_the_right_denominator():
    teams = ["a", "b", "c", "d"]
    lg = leagues.League(slug="toy", name="Toy", country="X", n_teams=4, n_matches=4,
                        ucl_places=1, releg_places=1, releg_note=None)
    sim = simulate.simulate_season(_toy_fit(teams), _toy_fixtures(teams), teams,
                                   league=lg, n_sims=4000, scenarios=20,
                                   rating_sd=0.0, curves=True)
    curves = sim["curves"]
    assert {c["id"] for c in curves} == set(teams)
    for c in curves:
        assert len(c["pts"]) == len(c["n"]) == len(c["title"])
        assert sum(c["n"]) <= sim["n_sims"]
        for key in ("title", "ucl", "releg"):
            assert all(0.0 <= v <= 1.0 for v in c[key]), (c["id"], key)
        # Every published bin must clear the noise floor.
        assert all(k >= simulate.MIN_CURVE_SEASONS for k in c["n"])
        # More points can never make finishing top less likely, up to sampling
        # noise, in a league where everyone is identical.
        assert c["pts"] == sorted(c["pts"])


def test_curves_are_monotone_in_points_for_identical_clubs():
    """With four identical clubs, more points must mean a better finish. This is
    the sanity check that catches a curve indexed against the wrong club."""
    teams = ["a", "b", "c", "d"]
    lg = leagues.League(slug="toy", name="Toy", country="X", n_teams=4, n_matches=12,
                        ucl_places=1, releg_places=1, releg_note=None)
    sim = simulate.simulate_season(_toy_fit(teams), _toy_fixtures(teams, 6), teams,
                                   league=lg, n_sims=20000, scenarios=50,
                                   rating_sd=0.0, curves=True)
    for c in sim["curves"]:
        vals = c["title"]
        if len(vals) < 4:
            continue
        assert vals[-1] > vals[0], c["id"]
        # allow small sampling wobble but not a trend the wrong way
        assert np.corrcoef(c["pts"], vals)[0, 1] > 0.9, c["id"]


def test_curves_can_be_switched_off():
    teams = ["a", "b"]
    lg = leagues.League(slug="toy", name="Toy", country="X", n_teams=2, n_matches=2,
                        ucl_places=1, releg_places=1, releg_note=None)
    sim = simulate.simulate_season(_toy_fit(teams), _toy_fixtures(teams), teams,
                                   league=lg, n_sims=500, scenarios=5, curves=False)
    assert sim["curves"] is None


# ------------------------------------------------------------------ half time / game state
HT_TEXT = """= Toy League 2024/25

▪ Matchday 1
  Sat Aug 10 2024
    15:00  Alpha FC                v Beta FC                  2-1 (1-1)
           Gamma FC                v Delta FC                 0-0
  Sat Aug 17 2024
    15:00  Beta FC                 v Gamma FC                 1-3 (1-0)
"""


def test_openfootball_half_time_is_parsed():
    """The parenthesised score has been in these files all along, and until now
    the reader threw it away."""
    ms = parse_openfootball(HT_TEXT, "2024-25", TeamRegistry())
    assert len(ms) == 3
    by = {(m.home, m.away): m for m in ms}
    a = by[("alpha", "beta")]
    assert (a.hg, a.ag) == (2, 1) and (a.hthg, a.htag) == (1, 1)
    b = by[("beta", "gamma")]
    assert (b.hg, b.ag) == (1, 3) and (b.hthg, b.htag) == (1, 0)
    # No parenthesis at all means no half-time score, not a zero.
    c = by[("gamma", "delta")]
    assert (c.hg, c.ag) == (0, 0) and c.hthg is None and c.htag is None


def test_aet_still_ingests_the_ninety_minute_score_and_its_half_time():
    text = ("= Cup\n\n▪ Finals, Final\n  Sat May 30 2026\n"
            "    20:00  Alpha FC                v Beta FC   "
            "4-3 pen. 1-1 a.e.t. (1-1, 0-1)\n")
    ms = parse_openfootball(text, "2025-26", TeamRegistry(), euro=True)
    assert len(ms) == 1
    m = ms[0]
    assert (m.hg, m.ag) == (1, 1) and m.aet is True
    assert (m.hthg, m.htag) == (0, 1)


def _state_matches():
    """One club that always leads and holds, one that always trails."""
    out = []
    for i in range(12):
        out.append(Match(date=dt.date(2024, 8, 1) + dt.timedelta(days=i * 7),
                         home="up", away="down", hg=2, ag=0, hthg=1, htag=0,
                         hy=1, ay=3, hr=0, ar=0, hf=8, af=14, hc=7, ac=2,
                         referee="A Referee", season="2024-25", played=True))
    return out


def test_game_state_reads_the_two_sides_of_the_interval():
    st = gamestate.game_state(_state_matches(), ["up", "down"])
    assert st["up"]["led"] == 12 and st["up"]["hold_pct"] == 1.0
    assert st["up"]["behind"] == 0 and st["up"]["recover_pct"] is None
    assert st["down"]["behind"] == 12 and st["down"]["recover_pct"] == 0.0
    assert st["up"]["first_half_gd"] == pytest.approx(1.0)
    assert st["up"]["second_half_gd"] == pytest.approx(1.0)


def test_discipline_and_referees_are_records_not_models():
    d = gamestate.discipline(_state_matches(), ["up", "down"])
    assert d["up"]["yellow_pm"] == 1.0 and d["down"]["yellow_pm"] == 3.0
    assert d["up"]["corners_pm"] == 7.0 and d["up"]["corners_against_pm"] == 2.0
    refs = gamestate.referees(_state_matches())
    assert len(refs) == 0, "a 12-match record is below the publication threshold"
    many = _state_matches() * 3
    refs = gamestate.referees(many)
    assert refs and refs[0]["name"] == "A Referee"
    assert refs[0]["yellow_pm"] == pytest.approx(4.0)
    assert refs[0]["home_win_pct"] == 1.0


def test_a_feed_without_half_time_yields_no_half_time_model():
    """Three of the eight competitions have no half-time column at all. The
    build must degrade to no half-time section rather than to a fitted zero."""
    bare = [Match(date=dt.date(2024, 8, 1), home="a", away="b", hg=1, ag=0,
                  season="2024-25", played=True)] * 50
    assert gamestate.half_time_fit(bare, dt.date(2025, 1, 1)) is None
    assert gamestate.half_time_report(None, "a", "b") is None


# ------------------------------------------------------------------ calendars and feeds
def test_calendar_is_well_formed_and_carries_the_forecast():
    meta = {"a": {"name": "Alpha FC", "short": "ALP"},
            "b": {"name": "Beta FC", "short": "BET"}}
    rows = [
        {"h": "a", "a": "b", "date": "2026-08-21", "time": "20:00", "md": 1,
         "ph": 0.5, "pd": 0.3, "pa": 0.2, "xgh": 1.7, "xga": 1.1,
         "sc": [2, 1], "scp": 0.11, "played": False, "swings": []},
        {"h": "b", "a": "a", "date": "2026-12-26", "time": None, "md": 18,
         "ph": 0.3, "pd": 0.3, "pa": 0.4, "xgh": 1.1, "xga": 1.4,
         "sc": [1, 1], "scp": 0.13, "played": True, "hg": 0, "ag": 3, "swings": []},
    ]
    ics = feeds.calendar(rows, meta, title="Toy", uid_ns="toy.537")
    assert ics.startswith("BEGIN:VCALENDAR\r\n") and ics.endswith("END:VCALENDAR\r\n")
    assert ics.count("BEGIN:VEVENT") == 2 and ics.count("END:VEVENT") == 2
    # A timed fixture is floating local time; an untimed one is an all-day event.
    assert "DTSTART:20260821T200000" in ics
    assert "DTSTART;VALUE=DATE:20261226" in ics
    assert "Alpha FC v Beta FC" in ics
    assert "Beta FC 0–3 Alpha FC" in ics          # played rows carry the score
    # Every line is CRLF-terminated and within the 75-octet limit.
    for line in ics.split("\r\n"):
        assert len(line.encode("utf-8")) <= 75, line


def test_calendar_can_be_filtered_to_one_club():
    meta = {"a": {"name": "A"}, "b": {"name": "B"}, "c": {"name": "C"}}
    rows = [{"h": "a", "a": "b", "date": "2026-08-21", "time": "15:00", "md": 1,
             "ph": .4, "pd": .3, "pa": .3, "xgh": 1, "xga": 1, "sc": [1, 1],
             "scp": .1, "played": False, "swings": []},
            {"h": "c", "a": "b", "date": "2026-08-28", "time": "15:00", "md": 2,
             "ph": .4, "pd": .3, "pa": .3, "xgh": 1, "xga": 1, "sc": [1, 1],
             "scp": .1, "played": False, "swings": []}]
    assert feeds.calendar(rows, meta, title="t", uid_ns="t", team="a").count("BEGIN:VEVENT") == 1
    assert feeds.calendar(rows, meta, title="t", uid_ns="t", team="b").count("BEGIN:VEVENT") == 2


def test_the_feed_stays_quiet_when_nothing_moved():
    """Four items a day saying nothing is how a feed gets unsubscribed from."""
    meta = {"a": {"name": "Alpha"}}
    quiet = [{"slug": "x", "name": "X", "url": "u", "meta": meta,
              "recap": {"asof": "2026-09-01",
                        "movers": [{"id": "a", "metric": "title", "delta": 0.004,
                                    "before": 0.10, "after": 0.104}]}}]
    assert feeds.feed_items(quiet) == []
    loud = [{"slug": "x", "name": "X", "url": "u", "meta": meta,
             "recap": {"asof": "2026-09-01",
                       "movers": [{"id": "a", "metric": "title", "delta": 0.12,
                                   "before": 0.10, "after": 0.22}]}}]
    items = feeds.feed_items(loud)
    assert len(items) == 1 and "Alpha" in items[0]["content_text"]
    doc = feeds.json_feed(items, home="https://example.test/", title="t")
    assert doc["version"].startswith("https://jsonfeed.org/")
    xml = feeds.rss(items, home="https://example.test/", title="t")
    assert xml.startswith("<?xml") and "<item>" in xml


# ------------------------------------------------------------------ share cards
def test_ordinals_survive_a_twenty_four_club_division():
    """A four-element suffix list and a `20 <= v < 30` branch is how the
    Championship's 24th-placed club crashed the card renderer."""
    got = [social._ord(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 24, 111)]
    assert got == ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th",
                   "21st", "22nd", "23rd", "24th", "111th"]


def test_cards_are_optional_and_never_fatal():
    """Pillow and a font are a nice-to-have; the forecast is not."""
    assert issubclass(social.Unavailable, RuntimeError)


# ------------------------------------------------------------------ pooled ranking
def test_club_leagues_follows_the_club_not_its_history():
    """Brentford has played twenty seasons in the second tier and four in the
    first. The league it belongs to is the one it is in now."""
    old = [Match(date=dt.date(2016, 8, 1) + dt.timedelta(days=i), home="brentford",
                 away="x", hg=1, ag=0, played=True, comp="premier-league-2")
           for i in range(200)]
    new = [Match(date=dt.date(2026, 5, 1) + dt.timedelta(days=i), home="brentford",
                 away="y", hg=1, ag=0, played=True, comp="premier-league")
           for i in range(20)]
    got = europe.club_leagues(old + new)
    assert got["brentford"] == "premier-league"
    # With nothing recent at all, the whole record decides it again.
    assert europe.club_leagues(old)["brentford"] == "premier-league-2"


def test_spi_from_offence_and_defence_matches_the_pipeline_definition():
    """`rankings.spi_from` and `run.spi` must agree, or the published `spi` and
    the published `off`/`def` would tell different stories."""
    from model.run import spi as run_spi
    teams = ["a", "b"]
    fit = Fit(teams, np.array([0.25, -0.25]), np.array([0.1, -0.1]),
              np.log(1.3), 0.2, -0.02)
    for t in teams:
        mine = rankings.spi_from(fit.offence(t), fit.defence(t), fit.home, fit.rho)
        assert mine == pytest.approx(run_spi(fit, t), abs=1e-9)


def test_head_to_head_is_bounded_to_the_clubs_that_can_be_compared():
    reg = TeamRegistry()
    corpus = europe.Corpus(reg)
    corpus.matches = [
        Match(date=dt.date(2024, 9, 1), home="arsenal", away="psv", hg=2, ag=0,
              played=True, comp="cl"),
        Match(date=dt.date(2025, 3, 12), home="psv", away="arsenal", hg=2, ag=2,
              played=True, comp="cl"),
        Match(date=dt.date(2025, 3, 12), home="arsenal", away="not-featured", hg=1,
              ag=0, played=True, comp="cl"),
    ]
    pairs = europe and rankings.head_to_head(corpus, {"arsenal", "psv"})
    assert set(pairs) == {"arsenal|psv"}
    rec = pairs["arsenal|psv"]
    assert rec["n"] == 2 and rec["aw"] == 1 and rec["d"] == 1 and rec["bw"] == 0
    assert rec["agf"] == 4 and rec["bgf"] == 2
    assert rec["last"] == "2025-03-12"
    # The last-result string carries display names, not raw ids.
    assert "PSV" in rec["lastr"] and "Arsenal" in rec["lastr"]


# ------------------------------------------------------------------ site contract
SITE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "site")


def test_every_page_declares_a_share_image():
    """The site has always told scrapers to expect a large image. Until the card
    renderer landed it shipped none, so every shared link rendered blank."""
    import glob
    for path in sorted(glob.glob(os.path.join(SITE, "*.html"))):
        html = open(path, encoding="utf-8").read()
        if "twitter:card" not in html:
            continue                       # embed.html is deliberately chromeless
        assert 'property="og:image"' in html, os.path.basename(path)
        assert 'name="twitter:image"' in html, os.path.basename(path)


def test_the_manifest_and_worker_exist_and_agree():
    man = json.load(open(os.path.join(SITE, "manifest.webmanifest")))
    assert man["start_url"].endswith("index.html")
    assert os.path.exists(os.path.join(SITE, man["icons"][0]["src"]))
    sw = open(os.path.join(SITE, "sw.js"), encoding="utf-8").read()
    # Network-first, always: a cache-first worker on a six-hourly rebuild would
    # serve April's numbers in May and look entirely fine doing it.
    assert "await fetch(e.request)" in sw
    assert "caches.match" in sw
    assert sw.index("await fetch(e.request)") < sw.index("caches.match(e.request")
