"""The gate on second feeds.

None of these tests touch the network. That is the point: the readers in
`model/footballdata.py` and `model/wikifootball.py` were written in a sandbox
that cannot reach either host, so the only thing worth testing here is the part
that decides whether to believe a feed once a runner has fetched it.

Each test builds a trusted feed (what openfootball already gave us) and a second
feed (what the new source claims), and checks that the gate reaches the right
verdict. The failures modelled are the ones that would actually happen: a second
spelling of a club we already hold, a correctly-parsed file of the wrong
competition, a feed that stopped being updated, and a club promoted after the
old feed went quiet.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import external, footballdata, wikifootball  # noqa: E402
from model.parse import Match, TeamRegistry  # noqa: E402

CLUBS = ["Raków Częstochowa", "Legia Warszawa", "Lech Poznan", "Pogoń Szczecin",
         "Górnik Zabrze", "Piast Gliwice", "Widzew Łódź", "Zagłębie Lubin",
         "Korona Kielce", "Radomiak Radom", "Motor Lublin", "Śląsk Wrocław"]
GROUP = "dom-pol"


def _round_robin(clubs, season, start, *, score=lambda i: (i % 3, (i + 1) % 2)):
    """Every ordered pair, so a 12-club league gives 132 fixtures."""
    out, i = [], 0
    for h in clubs:
        for a in clubs:
            if h == a:
                continue
            hg, ag = score(i)
            out.append((h, a, hg, ag, start + dt.timedelta(days=i % 300)))
            i += 1
    return out


def trusted(reg, clubs=CLUBS, season="2024-25", start=dt.date(2024, 8, 1), **kw):
    """What the GitHub feed already gave us, with ids resolved as it resolves them."""
    return [Match(date=d, home=reg.resolve(h), away=reg.resolve(a), hg=hg, ag=ag,
                  season=season, played=True, comp=GROUP)
            for h, a, hg, ag, d in _round_robin(clubs, season, start, **kw)]


def second(clubs=CLUBS, season="2024-25", start=dt.date(2024, 8, 1), **kw):
    """What a second source claims, as the reader hands it over: raw names."""
    return [(Match(date=d, home=h, away=a, hg=hg, ag=ag, season=season, played=True), h, a)
            for h, a, hg, ag, d in _round_robin(clubs, season, start, **kw)]


def make(rows, assoc="POL"):
    return external.ExternalSource(
        source="test", assoc=assoc, league="Ekstraklasa", group=GROUP,
        load=lambda reg, r=rows: list(r))


TODAY = dt.date(2026, 8, 20)


# ------------------------------------------------------------------ the happy path
def test_a_second_feed_arms_when_it_reproduces_a_season_we_already_hold():
    """The overlap season is the evidence. Two independent feeds agreeing on 132
    scores is what makes the seasons after it worth believing."""
    reg = TeamRegistry()
    have = trusted(reg)
    src = make(second() + second(season="2025-26", start=dt.date(2025, 8, 1)))
    v = external.probe(src, reg, have, today=TODAY)
    assert v.ok, v.reason
    assert v.overlap_season == "2024-25"
    assert v.compared == 132 and v.agreed == 132
    assert v.unresolved == () and v.new_clubs == ()


def test_only_the_seasons_we_do_not_already_have_are_added():
    """Where both feeds describe a season the GitHub one wins: it is the feed
    with real dates on it and the one whose spellings defined the club ids."""
    reg = TeamRegistry()
    have = trusted(reg)
    src = make(second() + second(season="2025-26", start=dt.date(2025, 8, 1)))
    external.probe(src, reg, have, today=TODAY)
    got = external.matches(src, reg, have)
    assert {m.season for m in got} == {"2025-26"}
    assert len(got) == 132
    assert {m.comp for m in got} == {GROUP}
    assert all(m.home in reg.meta and m.away in reg.meta for m in got)


def test_a_pair_that_meets_twice_at_the_same_ground_still_agrees():
    """A fixture is not unique. Romania's Liga I splits into a championship round
    after the regular season and Switzerland's twelve clubs play each other three
    times, so the same pair meets twice at the same ground in one season.

    Indexed as a dict the second meeting overwrites the first, every first
    meeting then reads as a disagreement, and both leagues are refused with
    their feeds entirely correct. Which is what happened: Poland armed, because
    eighteen clubs meeting twice gives one fixture per ordered pair, and the two
    leagues after it did not.
    """
    reg = TeamRegistry()
    twice = _round_robin(CLUBS, "2024-25", dt.date(2024, 8, 1))
    twice += [(h, a, hg + 1, ag, d + dt.timedelta(days=150))
              for h, a, hg, ag, d in _round_robin(CLUBS, "2024-25", dt.date(2024, 8, 1))]
    have = [Match(date=d, home=reg.resolve(h), away=reg.resolve(a), hg=hg, ag=ag,
                  season="2024-25", played=True, comp=GROUP) for h, a, hg, ag, d in twice]
    theirs = [(Match(date=d, home=h, away=a, hg=hg, ag=ag, season="2024-25",
                     played=True), h, a) for h, a, hg, ag, d in twice]
    src = make(theirs + second(season="2025-26", start=dt.date(2025, 8, 1)))
    v = external.probe(src, reg, have, today=TODAY)
    assert v.ok, v.reason
    assert v.compared == 264 and v.agreed == 264


def test_the_same_score_cannot_be_matched_twice():
    """The counting is a multiset, not a set: if they record a pair meeting twice
    and we record it once, only one of the two can agree with us."""
    reg = TeamRegistry()
    have = trusted(reg)
    dupes = second() + second()[:60]
    src = make(dupes + second(season="2025-26", start=dt.date(2025, 8, 1)))
    v = external.probe(src, reg, have, today=TODAY)
    assert v.compared == 192, "every fixture they list that we also have"
    assert v.agreed == 132, "but each of our results can only be spent once"
    assert not v.ok and "not the competition" in v.reason


# ------------------------------------------------------------------ the refusals
def test_a_second_spelling_in_the_overlap_blocks_the_league():
    """`Rakow` is not a new club, it is `Raków Częstochowa` written by someone
    without a keyboard for it. Minting an id would put a duplicate in the global
    ranking with half a record, so the league waits for an alias instead."""
    reg = TeamRegistry()
    have = trusted(reg)
    theirs = ["Rakow" if c.startswith("Raków") else c for c in CLUBS]
    src = make(second(clubs=theirs))
    v = external.probe(src, reg, have, today=TODAY)
    assert not v.ok
    assert v.unresolved == ("Rakow",)
    assert "spellings rather than new clubs" in v.reason


def test_the_right_shape_of_the_wrong_competition_is_refused():
    """The dangerous failure: a file that parses perfectly, carries the clubs we
    expect, and is the second division. Only the scores catch it."""
    reg = TeamRegistry()
    have = trusted(reg)
    src = make(second(score=lambda i: ((i + 1) % 4, i % 3)))
    v = external.probe(src, reg, have, today=TODAY)
    assert not v.ok
    assert "not the competition it claims to be" in v.reason
    assert v.compared == 132 and v.agreed < 60


def test_a_feed_with_no_season_in_common_cannot_be_checked_and_does_not_arm():
    """A source that only carries seasons we have never seen has nothing to be
    verified against, and an unverifiable source is one we do not use."""
    reg = TeamRegistry()
    have = trusted(reg)
    src = make(second(season="2025-26", start=dt.date(2025, 8, 1)))
    v = external.probe(src, reg, have, today=TODAY)
    assert not v.ok
    assert "no season in common" in v.reason


def test_a_thin_overlap_is_not_enough_to_call_it_the_same_competition():
    """Half a dozen matching fixtures is a coincidence, not a verification."""
    reg = TeamRegistry()
    have = trusted(reg)
    src = make(second()[:20] + second(season="2025-26", start=dt.date(2025, 8, 1)))
    v = external.probe(src, reg, have, today=TODAY)
    assert not v.ok
    assert "line up against" in v.reason


def test_a_feed_that_stopped_being_updated_does_not_arm():
    """This whole exercise exists because a feed went quiet without saying so.
    A second feed that has done the same is not an improvement."""
    reg = TeamRegistry()
    have = trusted(reg)
    src = make(second())
    v = external.probe(src, reg, have, today=dt.date(2027, 6, 1))
    assert not v.ok
    assert "older than" in v.reason


def test_an_unreachable_source_is_skipped_rather_than_fatal():
    """A blocked host, a 500, a DNS failure: the build carries on with the data
    it already had. This is the path every local run takes."""
    reg = TeamRegistry()

    def boom(_reg):
        raise footballdata.FormatError("unreachable: https://example.invalid/x.csv")

    src = external.ExternalSource(source="test", assoc="POL", league="Ekstraklasa",
                                  group=GROUP, load=boom)
    v = external.probe(src, reg, trusted(reg), today=TODAY)
    assert not v.ok and v.reason == "could not be fetched"
    assert "unreachable" in v.detail, "the exception belongs in the log, not on the site"
    assert external.matches(src, reg) == []


# ------------------------------------------------------------------ promotions
def test_a_club_promoted_after_the_old_feed_went_quiet_is_allowed_through():
    """The distinction the gate exists to draw: a name absent from the overlap
    but present after it is a promoted club, and minting an id for it is right."""
    reg = TeamRegistry()
    have = trusted(reg)
    newer = CLUBS[:-1] + ["Arka Gdynia"]
    src = make(second() + second(clubs=newer, season="2025-26",
                                 start=dt.date(2025, 8, 1)))
    v = external.probe(src, reg, have, today=TODAY)
    assert v.ok, v.reason
    assert v.new_clubs == ("Arka Gdynia",)
    got = external.matches(src, reg, have)
    assert reg.known("Arka Gdynia") is not None
    assert any(m.home == reg.known("Arka Gdynia") for m in got)


def test_a_source_that_did_not_arm_contributes_nothing():
    reg = TeamRegistry()
    have = trusted(reg)
    src = make(second(clubs=["Rakow" if c.startswith("Raków") else c for c in CLUBS]))
    external.probe(src, reg, have, today=TODAY)
    assert external.matches(src, reg, have) == []


# ------------------------------------------------------------------ the resolver
def test_the_strict_resolver_refuses_to_invent_a_club():
    """`resolve` mints an id for anything, which is right for a Championship
    opponent from 2004 and wrong for a second feed of a league we already hold."""
    reg = TeamRegistry()
    before = len(reg.meta)
    assert reg.known("Nonesuch Athletic FC") is None
    assert len(reg.meta) == before
    assert reg.resolve("Nonesuch Athletic FC") is not None
    assert len(reg.meta) == before + 1
    assert reg.known("Nonesuch Athletic FC") is not None


# ------------------------------------------------------------------ the readers
FD_CSV = """Country,League,Season,Date,Time,Home,Away,HG,AG,Res,PH,PD,PA
Poland,Ekstraklasa,2025/2026,18/07/2025,18:00,Lech Poznan,Cracovia,1,0,H,2.1,3.4,3.5
Poland,Ekstraklasa,2025/2026,19/07/2025,15:00,Legia,Rakow,2,2,D,2.4,3.2,3.0
Poland,Ekstraklasa,2025/2026,26/07/2025,,Rakow,Lech Poznan,,,,2.6,3.1,2.9
"""


def test_the_football_data_reader_reads_the_published_shape(monkeypatch):
    """Header-driven, so the reader survives a column being added, and strict,
    so it does not survive the columns it needs going missing."""
    monkeypatch.setattr(footballdata.fetch, "get", lambda *a, **k: FD_CSV)
    rows = footballdata.load("POL", TeamRegistry())
    assert len(rows) == 2                      # the third has no score yet
    m, home, away = rows[0]
    # "Cracovia" comes back as "KS Cracovia": the publisher's short names are
    # mapped onto the spellings the GitHub feed defined the club ids from,
    # inside the reader, so no caller can forget to do it.
    assert (home, away) == ("Lech Poznan", "KS Cracovia")
    assert (m.hg, m.ag) == (1, 0)
    assert m.date == dt.date(2025, 7, 18)
    assert m.season == "2025-26"
    assert m.played


def test_the_football_data_reader_refuses_a_file_it_does_not_recognise(monkeypatch):
    """A reader that guesses at an unexpected header is how a feed silently
    starts producing nonsense, so this one stops instead."""
    monkeypatch.setattr(footballdata.fetch, "get",
                        lambda *a, **k: "Date,Something,Else\n01/01/2026,a,b\n")
    with pytest.raises(footballdata.FormatError, match="no home column"):
        footballdata.load("POL", TeamRegistry())


def test_an_unreachable_file_raises_rather_than_returning_nothing(monkeypatch):
    """Nothing-parsed and could-not-be-fetched are different failures and the
    verdict should say which."""
    monkeypatch.setattr(footballdata.fetch, "get", lambda *a, **k: None)
    with pytest.raises(footballdata.FormatError, match="unreachable"):
        footballdata.load("POL", TeamRegistry())


GRID = """
{{#invoke:sports results|main|style=WDL
|update=complete
|team1=LIN|team2=EUR|team3=MAG
|name_LIN=Lincoln Red Imps
|name_EUR=Europa
|name_MAG=Magpies
|match_LIN_EUR=3–0
|match_LIN_MAG=2–1
|match_EUR_LIN=1–1
|match_EUR_MAG=0-2
|match_MAG_LIN=0–4
|match_MAG_EUR=2–2
}}
"""


def test_the_wikipedia_reader_reads_a_results_grid():
    got = wikifootball.parse_grid(GRID)
    assert len(got) == 6
    assert ("Lincoln Red Imps", "Europa", 3, 0) in got
    assert ("Europa", "Magpies", 0, 2) in got     # a hyphen, not an en dash
    assert ("Magpies", "Lincoln Red Imps", 0, 4) in got


def test_a_page_with_no_grid_says_so_rather_than_returning_an_empty_season():
    with pytest.raises(wikifootball.GridError, match="no match_X_Y cells"):
        wikifootball.parse_grid("== Results ==\nThe season was played.\n")


def test_every_match_from_a_results_grid_is_dated_at_its_season_midpoint():
    """The grid carries no dates, and inventing a run of them would be inventing
    precision. One epoch, one weight, flagged as approximate."""
    assert wikifootball.season_midpoint("2025-26") == dt.date(2026, 1, 1)
    assert wikifootball.season_midpoint("2025") == dt.date(2025, 7, 1)


def test_the_wikipedia_reader_asks_for_a_season_it_can_be_checked_on():
    """It fetches one season more than it needs. The extra one is not data, it
    is the season `probe` lines the two feeds up on."""
    got = wikifootball.seasons_for(("2023-24", "2024-25"), through="2026-27")
    assert got[0] == "2024-25", "the newest season the GitHub feed carries"
    assert "2025-26" in got and "2026-27" in got


def test_no_wikipedia_league_is_armed_without_a_receipt():
    """Arming a league is a deliberate act taken after reading a green probe on a
    runner, and the probe is the only thing that can be read: nothing here can
    reach Wikipedia. So every armed code carries a copy of the verdict that
    armed it -- the date, the season it was lined up on, and the counts -- which
    can be checked against the run it names. A code without one was added from
    an armchair."""
    import json
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "armed.json")
    seen = {}
    if os.path.exists(path):
        seen = {r["assoc"]: r for r in json.load(open(path))["probes"]}
    for assoc in wikifootball.ARMED | wikifootball.PROJECTED:
        r = seen.get(assoc)
        assert r, f"{assoc} is armed with no probe recorded in data/armed.json"
        assert r["agreed"] == r["compared"] > 0, r
        assert r["overlap_season"] and r["run"], r


def test_a_league_feeds_the_corpus_or_only_its_own_projection():
    """Poland has two verified feeds. Only one may put matches in the pooled
    corpus -- two would count the same season twice -- while the other is still
    the place the season's entrant list comes from."""
    assert "POL" not in wikifootball.ARMED, (
        "Poland's results come from football-data.co.uk, which carries dates; "
        "arming the grid as well would add the same season twice")


# ------------------------------------------------------------------ real names
#: The diacritics football-data.co.uk drops. It writes ASCII: 'Rakow' for
#: 'Raków', 'Otelul' for 'Oţelul', 'Zurich' for 'Zürich'.
_ASCII = str.maketrans({"ó": "o", "ą": "a", "ę": "e", "ł": "l", "ń": "n", "ś": "s",
                        "ż": "z", "ź": "z", "ć": "c", "Ł": "L", "Ś": "S", "Ż": "Z",
                        "ș": "s", "ț": "t", "ş": "s", "ţ": "t", "ă": "a", "â": "a",
                        "î": "i", "Ă": "A", "Ș": "S", "ü": "u", "ö": "o", "ä": "a",
                        "é": "e", "è": "e", "à": "a"})


@pytest.mark.parametrize("assoc", ["POL", "ROU", "SUI"])
def test_the_whole_path_works_on_real_club_names(assoc):
    """The one thing the tests above cannot show.

    Everything else here runs on twelve invented clubs, which proves the gate
    reasons correctly and proves nothing about whether a second feed's spellings
    line up with ours. So this takes the real openfootball data for one league,
    rewrites it into football-data.co.uk's shape and its ASCII spellings, adds a
    season the GitHub feed never had, and runs the unmodified probe over it.

    It is how the fixture-counting bug was found: eighteen Polish clubs meeting
    twice give one fixture per ordered pair and armed cleanly, while Romania's
    championship round and Switzerland's triple round-robin put the same pair at
    the same ground twice, and every first meeting read as a disagreement.
    """
    from model import europe

    reg = TeamRegistry()
    dom = europe.load_domestic(reg, assocs=[assoc], quiet=True)
    if len(dom) < 200:
        pytest.skip(f"{assoc}'s GitHub feed is not reachable in this checkout")
    group = europe.BY_ASSOC[assoc].group

    header = ["Country", "League", "Season", "Date", "Time", "Home", "Away",
              "HG", "AG", "Res"]
    rows, clubs = [header], set()
    for m in dom:
        h = reg.display(m.home).translate(_ASCII)
        a = reg.display(m.away).translate(_ASCII)
        clubs |= {h, a}
        rows.append([assoc, "x", m.season.replace("-", "/20"),
                     m.date.strftime("%d/%m/%Y"), "", h, a, m.hg, m.ag, "H"])
    newest = max(m.season for m in dom)
    later = sorted(clubs)[:-1] + ["Newly Promoted SK"]
    nxt = f"{int(newest[:4]) + 1}-{str(int(newest[:4]) + 2)[-2:]}"
    d0 = dt.date(int(nxt[:4]), 8, 1)
    for i, (h, a) in enumerate((h, a) for h in later for a in later if h != a):
        rows.append([assoc, "x", nxt.replace("-", "/20"),
                     (d0 + dt.timedelta(days=i % 280)).strftime("%d/%m/%Y"), "",
                     h, a, i % 3, (i + 1) % 2, "H"])

    import csv as _csv
    import io as _io
    buf = _io.StringIO()
    _csv.writer(buf).writerows(rows)
    text = buf.getvalue()

    src = footballdata.source(assoc, "x", group)
    src.load = lambda _reg, t=text: _read(t, assoc)
    v = external.probe(src, reg, dom, today=dt.date(int(nxt[:4]) + 1, 6, 1))

    assert v.ok, f"{v.reason} {v.unresolved[:8]}"
    assert v.agreed == v.compared, (
        f"{v.compared - v.agreed} of {v.compared} results in {v.overlap_season} "
        "disagree with the feed they were generated from")
    assert v.new_clubs == ("Newly Promoted SK",), v.new_clubs

    got = external.matches(src, reg, dom)
    assert {m.season for m in got} == {nxt}, "only the seasons we did not have"
    assert {m.comp for m in got} == {group}
    before = {reg.known(reg.display(m.home)) for m in dom}
    assert all(m.home in before or m.home == reg.known("Newly Promoted SK")
               for m in got), "a second feed must not mint a club we already hold"


def _read(text, assoc):
    """`footballdata.load` with the fetch replaced by a string."""
    real = footballdata.fetch.get
    footballdata.fetch.get = lambda *a, **k: text
    try:
        return footballdata.load(assoc, TeamRegistry())
    finally:
        footballdata.fetch.get = real


# ------------------------------------------------------------------ the wiring
def test_every_second_feed_extends_a_league_rather_than_founding_one():
    """A group id that does not match the GitHub source's is not a second feed
    for that league, it is a parallel league with the same clubs in it: the
    pooled fit would give it its own home-advantage term and its own ridge
    centre, and the club would appear twice in the ranking."""
    from model import europe
    for src in europe.external_sources():
        assert src.assoc in europe.BY_ASSOC, f"{src.assoc} is not a league we carry"
        assert src.group == europe.BY_ASSOC[src.assoc].group, (
            f"{src.source}/{src.assoc} would found the group {src.group!r} rather "
            f"than extend {europe.BY_ASSOC[src.assoc].group!r}")


def test_a_blocked_network_leaves_the_corpus_exactly_as_it_was():
    """Every local run takes this path, and so does every runner if the hosts go
    down. It must be indistinguishable from not having the feature."""
    reg = TeamRegistry()
    have = trusted(reg)
    srcs = [make(second(), assoc="POL")]
    for src in srcs:
        src.load = lambda _r: (_ for _ in ()).throw(OSError("blocked"))
    out = [m for src in srcs for m in external.matches(src, reg, have)]
    assert out == []


# ------------------------------------------------------------------- spellings
def test_a_publishers_short_name_finds_the_club_we_already_hold():
    """The gap the ASCII test above cannot see.

    That test builds football-data.co.uk's file by folding accents out of our
    own club names, so every name it produces resolves by construction. The real
    file drops the town as well as the accents -- "Rakow", not "Rakow
    Czestochowa" -- and on 2026-08-21 that refused all three leagues on the
    runner. An alias is what fixes it, and this is the test that the alias is
    actually applied to what the reader returns.
    """
    reg = TeamRegistry()
    held = reg.resolve("Raków Częstochowa")
    header = "Country,League,Season,Date,Time,Home,Away,HG,AG,Res"
    text = (f"{header}\n"
            "POL,x,2024/2025,10/08/2024,18:00,Rakow,Legia,2,1,H\n"
            "POL,x,2024/2025,17/08/2024,18:00,Legia,Rakow,0,0,D\n")

    rows = _read(text, "POL")
    names = {raw for _, h, a in rows for raw in (h, a)}
    assert "Rakow" not in names, "the publisher's short name reached the gate raw"
    assert "Raków Częstochowa" in names
    assert reg.known("Raków Częstochowa") == held
    assert all(reg.known(n) is not None for n in names), sorted(names)


def test_an_alias_never_points_at_itself_or_at_another_alias():
    """A table of spellings is only useful while every entry moves a name onto a
    different one, exactly once. An entry that maps a name to itself is dead
    weight that reads as a fix, and two entries whose keys fold to the same key
    silently disagree about which club a name means."""
    from model.parse import normalise
    for assoc, table in footballdata.ALIASES.items():
        seen: dict[str, str] = {}
        for src_name, held in table.items():
            assert normalise(src_name) != normalise(held), (
                f"{assoc}: {src_name!r} -> {held!r} changes nothing")
            assert held not in table, (
                f"{assoc}: {held!r} is both an alias target and an alias key")
            key = normalise(src_name)
            assert key not in seen, (
                f"{assoc}: {src_name!r} and {seen[key]!r} are the same key")
            seen[key] = src_name


@pytest.mark.parametrize("assoc", ["POL", "ROU", "SUI"])
def test_every_alias_names_a_club_that_league_actually_had(assoc):
    """An alias written from a probe's output is a guess about the target as
    well as the source: "Sepsi OSK" and "Sepsi OSK Sfantu Gheorghe" are both
    plausible spellings of the club we hold, and only one of them is the one the
    GitHub feed wrote. A target that resolves to nothing mints a duplicate club
    the moment the league arms, which is the exact failure the gate exists to
    prevent -- so the target is checked against the feed that defined the ids."""
    from model import europe

    reg = TeamRegistry()
    dom = europe.load_domestic(reg, assocs=[assoc], quiet=True)
    if len(dom) < 200:
        pytest.skip(f"{assoc}'s GitHub feed is not reachable in this checkout")
    for src_name, held in footballdata.ALIASES.get(assoc, {}).items():
        assert reg.known(held) is not None, (
            f"{assoc}: {src_name!r} points at {held!r}, which is not a club the "
            "GitHub feed ever named")


def test_one_alias_covering_two_clubs_is_caught_before_they_merge():
    """The hazard an alias table carries.

    "Zaglebie" is Zagłębie Lubin while Lubin is the only Zagłębie in the league,
    and becomes wrong the season Zagłębie Sosnowiec comes up. The overlap test
    cannot see that: it checks a season we already hold, and the mistake happens
    in a season we do not. What it looks like from here is two spellings landing
    on one club id -- which is worth refusing on its own account, since that
    club would take both sets of results, appear to play itself, and stand in
    the table with twice as many matches as anybody else.
    """
    reg = TeamRegistry()
    have = trusted(reg)
    rows = second()
    # What a too-broad alias does: a second club's name mapped onto the first,
    # so the fixture between them becomes one club against itself.
    merged = [(m, "Legia Warsaw" if h == "Lech Poznan" else h,
               "Legia Warsaw" if a == "Lech Poznan" else a) for m, h, a in rows]
    src = make(merged, assoc="POL")
    v = external.probe(src, reg, have, today=dt.date(2025, 6, 1))
    assert not v.ok
    assert "playing itself" in v.reason, v.reason
    assert external.matches(src, reg, have) == []


def test_two_spellings_of_one_club_are_not_an_error():
    """The false positive the check above was written with and then lost.

    football-data.co.uk writes "Dinamo Bucuresti" in some rows and "Dinamo
    Bucureşti" in others, inside one season of one file. Both are the same club
    by any reading, `normalise` folds them together, and refusing the league for
    it took Romania back out of service after it had armed cleanly on 315 of 315
    results.
    """
    reg = TeamRegistry()
    have = trusted(reg)
    rows = second()
    mixed = [(m, "Legia Warsaw" if i % 2 and h == "Legia Warszawa" else h,
              "Legia Warsaw" if i % 2 and a == "Legia Warszawa" else a)
             for i, (m, h, a) in enumerate(rows)]
    v = external.probe(make(mixed, assoc="POL"), reg, have,
                       today=dt.date(2025, 6, 1))
    assert v.ok, v.reason


# ------------------------------------------------------------------ wiki names
LINKED_GRID = """
{{#invoke:sports results|main|style=WDL
|team1=BRE|team2=GOM|team3=MIN|team4=VIT
|name_BRE=[[FC Dynamo Brest|Dynamo Brest]]
|name_GOM=[[FC Gomel]]
|name_MIN=Minsk
|name_VIT=[[Strømsgodset Toppfotball|Strømsgodset]]
|match_BRE_GOM=3–0
|match_GOM_BRE=1–1
|match_MIN_VIT=0–2
}}
"""


def test_a_club_name_written_as_a_wiki_link_is_read_as_a_club():
    """The whole of the first Wikipedia probe's failure, in one line.

    Half these articles write `|name_BRE=[[FC Dynamo Brest|Dynamo Brest]]`, and
    a reader that stops at the first pipe reads that as "[[FC Dynamo Brest".
    Every one of the ten names Belarus refused on, and four of Norway's and six
    of Ukraine's, was this and nothing else.
    """
    assert wikifootball.name_variants("[[FC Dynamo Brest|Dynamo Brest]]") == (
        "Dynamo Brest", "FC Dynamo Brest")
    assert wikifootball.name_variants("[[FC Gomel]]") == ("FC Gomel",)
    assert wikifootball.name_variants("Minsk") == ("Minsk",)
    got = wikifootball.parse_grid(LINKED_GRID)
    assert ("Dynamo Brest", "FC Gomel", 3, 0) in got
    assert all("[[" not in n for row in got for n in row[:2])


def test_the_grid_hands_over_whichever_spelling_we_already_know(monkeypatch):
    """A link gives two names for one club and the feeds disagree about which
    one to use, so the reader tries both against the registry rather than
    picking one and needing an alias for every club that went the other way."""
    reg = TeamRegistry()
    # The registry holds the article's title and not what the table prints,
    # which is the way round Norway's clubs go.
    formal = reg.resolve("Strømsgodset Toppfotball")
    assert reg.known("Strømsgodset") is None
    monkeypatch.setattr(wikifootball.fetch, "get", lambda *a, **k: LINKED_GRID)
    rows = wikifootball.load("BLR", reg, ("2025",))
    names = {n for _, h, a in rows for n in (h, a)}
    assert "Strømsgodset Toppfotball" in names, (
        "the spelling the registry holds was not chosen")
    assert reg.known("Strømsgodset Toppfotball") == formal
    # "Minsk" is in the alias table for Belarus, so it arrives as the spelling
    # the GitHub feed uses. An alias beats the registry, which is the order that
    # lets one be written for a name the registry would otherwise get wrong.
    assert "FK Minsk" in names, sorted(names)


@pytest.mark.parametrize("assoc", sorted(wikifootball.ALIASES))
def test_every_grid_alias_names_a_club_that_league_actually_had(assoc):
    """Same rule as the football-data table: the target is checked against the
    feed that defined the club ids, so an alias cannot point at a spelling
    nobody uses and quietly mint a duplicate the day the league arms."""
    from model import europe

    reg = TeamRegistry()
    dom = europe.load_domestic(reg, assocs=[assoc], quiet=True)
    if len(dom) < 200:
        pytest.skip(f"{assoc}'s GitHub feed is not reachable in this checkout")
    for src_name, held in wikifootball.ALIASES[assoc].items():
        assert reg.known(held) is not None, (
            f"{assoc}: {src_name!r} points at {held!r}, which is not a club the "
            "GitHub feed ever named")
