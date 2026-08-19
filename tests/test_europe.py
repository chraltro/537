"""Tests for the European corpus: the parser defects, and the bridge itself.

The failure that matters here is silent. A European file that parses 187 of 189
matches still produces a plausible table; a club that resolves to `arsenal-eng`
in Europe and `arsenal` at home still produces a plausible rating, half of it
missing. Both look fine on the page. So the counts are pinned exactly, and the
club identities are checked against the domestic feeds rather than eyeballed.
"""
from __future__ import annotations

import collections
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import europe, fetch, leagues                               # noqa: E402
from model.parse import (Match, TeamRegistry, parse_openfootball,      # noqa: E402
                         parse_openfootball_euro)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def reg():
    return TeamRegistry()


def _file(season: str, comp: str) -> str | None:
    return fetch.get(europe.euro_url(season, comp), required=False)


# ------------------------------------------------------------------ 1.4 (1)
def test_digit_guard_keeps_clubs_with_numbers_in_their_names():
    """'Bayer 04 Leverkusen' is a club, '4-3' is a score. The old guard rejected
    any side containing a digit and silently lost twelve lines of one file."""
    reg = TeamRegistry()
    text = "\n".join([
        "▪ League, Matchday 1",
        "  Tue Sep 16 2025",
        "    18:45  Bayer 04 Leverkusen (GER) v Como 1907 (ITA)        2-1 (1-0)",
        "    18:45  FC Basel 1893 (SUI)     v Stade Brestois 29 (FRA)  0-0",
        "    18:45  1. FC Heidenheim 1846 (GER) v SK Dnipro-1 (UKR)    3-2 (2-1)",
    ])
    got = parse_openfootball_euro(text, "2025-26", reg)
    assert len(got) == 3
    assert got[0].home == reg.resolve("Bayer Leverkusen")


# ------------------------------------------------------------------ 1.4 (2)
def test_compound_shootout_result_yields_the_ninety_minute_score():
    """'4-3 pen. 1-1 a.e.t. (1-1, 0-1)' was 0-1 at half time, 1-1 at full time,
    still 1-1 after extra time, and a shootout after that. The fit wants 1-1."""
    reg = TeamRegistry()
    text = "\n".join([
        "▪ Finals, Final",
        "  Sat May 30 2026",
        "    18:00  Paris Saint-Germain FC (FRA) v Arsenal FC (ENG)  "
        "4-3 pen. 1-1 a.e.t. (1-1, 0-1)",
        "    18:00  Juventus FC (ITA)       v Galatasaray SK (TUR)     3-2 a.e.t. (3-0, 1-0)",
        "    18:00  ACF Fiorentina (ITA)    v Viktoria Plzeň (CZE)     2-0 a.e.t. (0-0)",
        "    18:00  Athletic Club (ESP)     v Arsenal FC (ENG)         0-2 (0-0)",
    ])
    got = parse_openfootball_euro(text, "2025-26", reg)
    assert [(m.hg, m.ag) for m in got] == [(1, 1), (3, 0), (0, 0), (0, 2)]
    assert [m.aet for m in got] == [True, True, True, False]


# ------------------------------------------------------------------ 1.4 (3)
def test_stage_headers_carry_matchday_and_do_not_leak_into_the_knockout():
    """'▪ League, Matchday 3' is a matchday; '▪ Playoffs, Matchday 2' is a
    second leg. Letting the second one set `matchday` would put knockout ties
    in the league table."""
    reg = TeamRegistry()
    text = "\n".join([
        "▪ League, Matchday 3",
        "  Tue Oct 20 2026",
        "    18:45  Arsenal FC (ENG)        v PSV (NED)                1-0 (0-0)",
        "▪ Playoffs, Matchday 2",
        "  Tue Feb 24 2027",
        "    18:45  Juventus FC (ITA)       v Galatasaray SK (TUR)     1-0 (0-0)",
        "▪ Finals, Quarterfinals",
        "  Tue Apr 6 2027",
        "    18:45  Arsenal FC (ENG)        v Real Madrid CF (ESP)     2-2 (1-1)",
    ])
    got = parse_openfootball_euro(text, "2026-27", reg)
    assert [(m.stage, m.matchday, m.leg) for m in got] == [
        ("league", 3, None), ("playoff", None, 2), ("qf", None, None)]


# ------------------------------------------------------------------ 1.4 (4)
def test_country_suffix_is_stripped_but_kept(reg):
    """`arsenal-eng` != `arsenal` is the bug that quietly disconnects the whole
    bridge, so the id must match the domestic feed and the code must survive."""
    text = "\n".join([
        "▪ League, Matchday 1",
        "  Tue Sep 16 2025",
        "    18:45  Athletic Club (ESP)     v Arsenal FC (ENG)         0-2 (0-0)",
    ])
    got = parse_openfootball_euro(text, "2025-26", TeamRegistry())
    assert got[0].away == reg.resolve("Arsenal")
    assert got[0].home == reg.resolve("Athletic Bilbao")
    assert (got[0].home_assoc, got[0].away_assoc) == ("ESP", "ENG")


def test_domestic_parsing_is_untouched_by_the_european_reader():
    """The European extras are opt-in: a domestic file must parse exactly as it
    did before, or the five published forecasts move for no reason."""
    reg = TeamRegistry()
    text = "\n".join([
        "▪ Matchday 1",
        "  Fri Aug 21 2026",
        "    20:00  Arsenal FC              v Chelsea FC               2-1 (1-0)",
    ])
    got = parse_openfootball(text, "2026-27", reg)
    assert (got[0].matchday, got[0].stage, got[0].comp) == (1, None, "")
    assert got[0].home_assoc is None


# ------------------------------------------------------------------ the gate
#: docs/european-competitions-plan.md §5, phase 1. Exact counts, because an
#: approximately-parsed season is the failure this whole file exists to catch.
EXPECTED = {
    ("2025-26", "cl"): (189, 36),
    ("2024-25", "cl"): (189, 36),
    ("2024-25", "el"): (189, 36),
    ("2024-25", "conf"): (153, 36),
}


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_swiss_seasons_parse_completely(reg, key):
    season, comp = key
    n_matches, n_teams = EXPECTED[key]
    text = _file(season, comp)
    if not text:
        pytest.skip(f"{season}/{comp}.txt unreachable")
    got = parse_openfootball_euro(text, season, reg, comp)
    assert len(got) == n_matches, f"{season}/{comp}: parsed {len(got)}"
    teams = {m.home for m in got} | {m.away for m in got}
    assert len(teams) == n_teams


def test_2025_26_champions_league_has_all_eight_matchdays(reg):
    text = _file("2025-26", "cl")
    if not text:
        pytest.skip("2025-26/cl.txt unreachable")
    got = parse_openfootball_euro(text, "2025-26", reg, "cl")
    phase = [m for m in got if m.stage == "league"]
    assert len(phase) == 144
    assert sorted({m.matchday for m in phase}) == [1, 2, 3, 4, 5, 6, 7, 8]
    for md in range(1, 9):
        assert sum(1 for m in phase if m.matchday == md) == 18
    teams = {m.home for m in phase} | {m.away for m in phase}
    assert len(teams) == 36
    for t in teams:
        assert sum(1 for m in phase if m.home == t) == 4
        assert sum(1 for m in phase if m.away == t) == 4


BIG_FIVE_IN_EUROPE = [
    ("Arsenal FC (ENG)", "arsenal"), ("Real Madrid CF (ESP)", "real-madrid"),
    ("FC Bayern München (GER)", "bayern"), ("Juventus FC (ITA)", "juventus"),
    ("Paris Saint-Germain FC (FRA)", "psg"), ("AS Monaco FC (MCO)", "monaco"),
    ("Club Atlético de Madrid (ESP)", "atletico-madrid"),
    ("Bayer 04 Leverkusen (GER)", "leverkusen"),
    ("FC Internazionale Milano (ITA)", "inter"),
    ("Manchester City FC (ENG)", "man-city"),
    ("1899 Hoffenheim (GER)", "hoffenheim"),
    ("Lazio Roma (ITA)", "lazio"),
]


@pytest.mark.parametrize("spelling,want", BIG_FIVE_IN_EUROPE)
def test_european_spellings_resolve_to_the_domestic_club_id(reg, spelling, want):
    """The bridge is only a bridge if both ends are the same club."""
    text = "\n".join(["▪ League, Matchday 1", "  Tue Sep 16 2025",
                      f"    18:45  {spelling} v Ajax Amsterdam (NED)   1-0 (0-0)"])
    got = parse_openfootball_euro(text, "2025-26", reg)
    assert got[0].home == want


def test_no_european_participant_forks_a_curated_club(reg):
    """No club in either Swiss season may be auto-registered: an auto id means
    the corpus invented a club the curated metadata does not know, which is what
    an `-eng`-style fork looks like from the inside."""
    text = _file("2025-26", "cl")
    if not text:
        pytest.skip("2025-26/cl.txt unreachable")
    r = TeamRegistry()
    got = parse_openfootball_euro(text, "2025-26", r, "cl")
    teams = {m.home for m in got} | {m.away for m in got}
    unmapped = sorted(t for t in teams if r.meta[t].get("auto"))
    assert unmapped == [], f"unmapped European clubs: {unmapped}"


def test_our_fixture_file_is_the_primary_source(tmp_path, monkeypatch):
    """Risk 1, and the difference between shipping and not: openfootball may be
    weeks behind, so our own file wins on pairings and yields only on results,
    and only when upstream has strictly more of them."""
    r = TeamRegistry()
    ours = "\n".join([
        "▪ League, Matchday 1",
        "  Tue Sep 8 2026",
        "    18:45  Arsenal FC (ENG)        v PSV (NED)",
        "    18:45  Real Madrid CF (ESP)    v Club Brugge (BEL)",
    ])
    theirs = "\n".join([
        "▪ League, Matchday 1",
        "  Tue Sep 8 2026",
        "    18:45  Arsenal FC (ENG)        v PSV (NED)                2-0 (1-0)",
    ])
    p = tmp_path / "fixtures-2026-27.txt"
    p.write_text(ours, encoding="utf-8")
    monkeypatch.setattr(europe, "our_fixture_path", lambda s: str(p))
    monkeypatch.setattr(europe.fetch, "get", lambda *a, **k: theirs)
    got, meta = europe.load_cup_fixtures(r, "2026-27")
    assert meta["source"] == "ours+openfootball-results"
    assert len(got) == 2, "our pairings must survive an incomplete upstream file"
    assert (got[0].hg, got[0].ag) == (2, 0), "upstream results must be folded in"
    assert got[1].played is False

    # Upstream with no more played matches than ours must not displace us.
    monkeypatch.setattr(europe.fetch, "get", lambda *a, **k: ours)
    got, meta = europe.load_cup_fixtures(TeamRegistry(), "2026-27")
    assert meta["source"] == "ours"


def test_an_empty_fixture_file_and_no_upstream_is_awaiting_draw(tmp_path,
                                                               monkeypatch):
    r = TeamRegistry()
    p = tmp_path / "fixtures-2026-27.txt"
    p.write_text("", encoding="utf-8")
    monkeypatch.setattr(europe, "our_fixture_path", lambda s: str(p))
    monkeypatch.setattr(europe.fetch, "get", lambda *a, **k: None)
    with pytest.raises(europe.AwaitingDraw):
        europe.load_cup_fixtures(r, "2026-27")


# --------------------------------------------------------- the built output
def _cup_forecast():
    path = os.path.join(HERE, "site", "data", "champions-league", "forecast.json")
    if not os.path.exists(path):
        pytest.skip("champions-league has not been built in this checkout")
    return json.load(open(path))


def test_cup_forecast_advancement_probabilities_are_coherent():
    fc = _cup_forecast()
    lg = leagues.CHAMPIONS_LEAGUE
    rows = fc["teams"]
    assert len(rows) == lg.n_teams
    for key, want in (("p_top8", 8), ("p_playoff", 16), ("p_out", 12),
                      ("p_r16", 16), ("p_qf", 8), ("p_sf", 4),
                      ("p_final", 2), ("p_win", 1)):
        assert abs(sum(r[key] for r in rows) - want) < 0.03, key
    for r in rows:
        assert abs(r["p_top8"] + r["p_playoff"] + r["p_out"] - 1) < 1e-3
        assert abs(sum(r["pos"]) - 1) < 1e-3
        assert r["p_win"] <= r["p_final"] <= r["p_sf"] <= r["p_qf"] <= r["p_r16"] + 1e-9


def test_cup_matches_carry_league_phase_matchdays_then_round_labels():
    path = os.path.join(HERE, "site", "data", "champions-league", "matches.json")
    if not os.path.exists(path):
        pytest.skip("champions-league has not been built in this checkout")
    ms = json.load(open(path))["matches"]
    mds = [m["md"] for m in ms]
    assert sorted({m for m in mds if isinstance(m, int)}) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert sum(1 for m in mds if isinstance(m, int)) == 144
    assert set(m for m in mds if isinstance(m, str)) <= {"KPO", "R16", "QF", "SF", "F"}
    for m in ms:
        assert abs(m["ph"] + m["pd"] + m["pa"] - 1) < 1e-3


def test_a_replay_build_is_never_marked_ready():
    """Staging data and a live forecast live in the same directory and have the
    same shape. The only thing keeping them apart is this stamp."""
    fc = _cup_forecast()
    if fc.get("replay") is None:
        pytest.skip("champions-league holds a live forecast, not a replay")
    man = json.load(open(os.path.join(HERE, "site", "data", "leagues.json")))
    entry = next(e for e in man["leagues"] if e["slug"] == "champions-league")
    assert entry["ready"] is False
    assert entry["kind"] == "cup"
    assert entry["advance_direct"] == 8 and entry["advance_playoff"] == 16


def test_participants_file_resolves_and_is_internally_consistent():
    path = europe.participants_path("2026-27")
    doc = json.load(open(path))
    assert doc["status"] == "provisional"
    r = TeamRegistry()
    ids = [c["id"] for c in doc["clubs"]]
    assert len(ids) == len(set(ids)), "duplicate club in the participant list"
    for c in doc["clubs"]:
        assert c["id"] in r.meta, f"{c['id']} has no team_meta entry"
        assert not r.meta[c["id"]].get("auto"), f"{c['id']} is auto-registered"
        assert c["pot"] in (1, 2, 3, 4)
    for tie in doc["playoff"]["ties"]:
        for side in tie:
            assert side["id"] in r.meta and not r.meta[side["id"]].get("auto")
    pots = collections.Counter(c["pot"] for c in doc["clubs"])
    assert pots[1] == 9, "pot 1 is always full: nine clubs including the holder"
    for p in (1, 2, 3, 4):
        assert pots[p] <= 9, f"pot {p} has {pots[p]} clubs"
    assert len(doc["clubs"]) + doc["unfilled"] == leagues.CHAMPIONS_LEAGUE.n_teams
    # No association may exceed the places its league actually has.
    per = collections.Counter(c["assoc"] for c in doc["clubs"])
    for assoc, slug in (("ENG", "premier-league"), ("ESP", "la-liga"),
                        ("ITA", "serie-a"), ("GER", "bundesliga")):
        assert per[assoc] == leagues.get(slug).ucl_places, assoc


def test_group_keys_separate_europe_from_every_domestic_league():
    d = europe.BY_ASSOC["NED"]
    assert d.group == "dom-ned"
    m = Match(date=None, home="a", away="b", comp="cl")
    assert europe.group_key(m) == europe.EUROPE
    m2 = Match(date=None, home="a", away="b", comp="dom-ned")
    assert europe.group_key(m2) == "dom-ned"
    m3 = Match(date=None, home="a", away="b")
    assert europe.group_key(m3, "premier-league") == "premier-league"
