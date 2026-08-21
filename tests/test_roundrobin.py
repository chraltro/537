"""The remaining fixtures of a double round-robin, and the shapes that are not one.

A results grid carries no fixture list, so the fixtures a season has left are
derived from the ones it has played: in a competition where every club plays
every other twice, once at each ground, the pairs without a score against them
are exactly the matches still to come. That is the format, published in the
competition's own rules, and not an inference about anybody's schedule.

Which makes the derivation only as safe as the assumption behind it, so the
tests below spend most of their effort on the assumption: real seasons of the
five leagues this is meant for, replayed match by match, and real seasons of
leagues it is not meant for -- Switzerland's triple round-robin, Romania's
championship split -- which have to be refused rather than quietly halved.
"""
import sys

import pytest

sys.path.insert(0, ".")

from model import europe, fetch, roundrobin                    # noqa: E402
from model.parse import TeamRegistry, parse_openfootball       # noqa: E402

PLAIN = ["NOR", "BLR", "LUX", "UKR", "POL"]
SHAPED = ["SUI", "ROU"]


def _season(assoc, complete=True):
    """The newest openfootball season for one association, as (clubs, pairs)."""
    reg = TeamRegistry()
    src = europe.BY_ASSOC[assoc]
    for label in reversed(src.seasons):
        text = fetch.get(src.url(label), required=False, tries=1)
        if not text:
            continue
        ms = [m for m in parse_openfootball(text, label, reg, comp=src.group)
              if m.played]
        clubs = sorted({m.home for m in ms} | {m.away for m in ms})
        if complete and len(ms) < len(clubs) * (len(clubs) - 1):
            continue                      # a season still in progress, or abandoned
        return label, clubs, [(m.home, m.away) for m in ms]
    pytest.skip(f"{assoc}'s GitHub feed is not reachable in this checkout")


# ------------------------------------------------------------------ the format
def test_a_finished_season_has_nothing_left_to_play():
    clubs = list("abcdef")
    assert roundrobin.remaining(clubs, roundrobin.all_pairs(clubs)) == []


def test_a_season_nobody_has_started_is_the_whole_fixture_list():
    clubs = list("abcdefgh")
    got = roundrobin.remaining(clubs, [])
    assert len(got) == roundrobin.season_size(8) == 56
    assert len({tuple(p) for p in got}) == 56
    for club in clubs:
        assert sum(1 for h, _ in got if h == club) == 7
        assert sum(1 for _, a in got if a == club) == 7


def test_a_club_that_has_not_played_yet_still_gets_its_fixtures():
    """The failure this guards against is silent: a club missing from the list
    loses all of its matches and the league is short a team, with a fixture
    count that still looks plausible."""
    clubs = list("abcd")
    got = roundrobin.remaining(clubs, [("a", "b"), ("b", "a")])
    assert len(got) == 10
    assert sum(1 for h, a in got if "d" in (h, a)) == 6


# ------------------------------------------------------------------ real seasons
@pytest.mark.parametrize("assoc", PLAIN)
def test_a_real_season_replayed_gives_back_exactly_the_matches_left(assoc):
    """The whole derivation, against a season whose answer is known.

    Take a finished season from the feed that carries it, hide a slice of it,
    and the derived remaining fixtures must be that slice and nothing else.
    Done at four points in the season, because a bug that drops a club shows up
    at one round and not another.
    """
    label, clubs, pairs = _season(assoc)
    n = len(clubs)
    assert len(pairs) == roundrobin.season_size(n), (
        f"{assoc} {label} is {len(pairs)} matches for {n} clubs, not a plain "
        "double round-robin, and does not belong in CANDIDATES")

    for cut in (0, len(pairs) // 4, len(pairs) // 2, len(pairs) - 1):
        played, hidden = pairs[:cut], pairs[cut:]
        got = roundrobin.remaining(clubs, played)
        assert sorted(got) == sorted(hidden), (
            f"{assoc} {label} at {cut} played: {len(got)} derived against "
            f"{len(hidden)} actually left")


@pytest.mark.parametrize("assoc", SHAPED)
def test_a_league_that_is_not_a_plain_round_robin_is_refused(assoc):
    """Switzerland's twelve clubs meet three times and Romania splits into a
    championship round, so both put the same pair at the same ground twice. A
    derivation that swallowed that would hand back a fixture list for a
    competition that does not exist."""
    _, clubs, pairs = _season(assoc, complete=False)
    with pytest.raises(roundrobin.ShapeError):
        roundrobin.remaining(clubs, pairs)


# ------------------------------------------------------------------ the refusals
def test_the_same_pair_at_the_same_ground_twice_is_not_this_format():
    with pytest.raises(roundrobin.ShapeError, match="appears twice"):
        roundrobin.remaining(list("abcd"), [("a", "b"), ("a", "b")])


def test_a_club_nobody_listed_stops_the_derivation():
    with pytest.raises(roundrobin.ShapeError, match="not in the club list"):
        roundrobin.remaining(list("abcd"), [("a", "z")])


def test_a_club_left_out_of_the_grid_is_caught_by_its_opponents():
    """The shape a grid takes when a relegated club is still in the cells but no
    longer in the entrant list. Nothing else would notice: the fixture count
    comes out plausible and the club quietly plays a season nobody counted."""
    played = [("a", x) for x in "bcd"] + [("a", "e")]
    with pytest.raises(roundrobin.ShapeError, match="not in the club list"):
        roundrobin.remaining(list("abcd"), played)
    assert len(roundrobin.remaining(list("abcde"), played)) == 16


def test_a_club_cannot_play_itself():
    with pytest.raises(roundrobin.ShapeError, match="playing itself"):
        roundrobin.remaining(list("abcd"), [("a", "a")])


# ------------------------------------------------------------------ the grid
def test_the_club_list_comes_from_the_grids_own_entrants():
    from model import wikifootball
    text = ("|team1=AAA|team2=BBB|team3=CCC|team4=DDD\n"
            "|name_AAA=Alpha\n|name_BBB=Beta\n|name_CCC=Gamma\n|name_DDD=Delta\n"
            "|match_AAA_BBB=1–0\n")
    assert wikifootball.grid_clubs(text) == ["Alpha", "Beta", "Gamma", "Delta"]
    played = [(h, a) for h, a, _, _ in wikifootball.parse_grid(text)]
    got = roundrobin.remaining(wikifootball.grid_clubs(text), played)
    assert len(got) == 11 and ("Alpha", "Beta") not in got


def test_a_team_code_with_no_name_beside_it_is_not_a_club():
    """Poland's season article carries a league table whose entrant list uses
    its own short codes, next to the results grid with its own. Reading both
    gave the Ekstraklasa two extra clubs called "G" and "WP"."""
    from model import wikifootball
    text = ("|team1=AAA|team2=BBB|team3=CCC|team4=DDD|team5=G|team6=WP\n"
            "|name_AAA=Alpha\n|name_BBB=Beta\n|name_CCC=Gamma\n|name_DDD=Delta\n"
            "|match_AAA_BBB=1–0\n")
    assert wikifootball.grid_clubs(text) == ["Alpha", "Beta", "Gamma", "Delta"]
