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


def test_a_club_code_that_is_not_ascii_is_still_a_club():
    """Codes are the article editor's own abbreviations and not all of them are
    ASCII. A class of [A-Za-z0-9_] matched neither the entrant list nor the name
    lines for those, their cells were then dropped as belonging to some other
    grid, and the league quietly ran a club short: one in Norway's 2025 season,
    three in Poland's 2026-27."""
    from model import wikifootball
    text = ("|team1=ŚLĄ|team2=BBB|team3=CCC|team4=DDD\n"
            "|name_ŚLĄ=[[Śląsk Wrocław|Śląsk]]\n|name_BBB=Beta\n"
            "|name_CCC=Gamma\n|name_DDD=Delta\n"
            "|match_ŚLĄ_BBB=2–1\n")
    assert wikifootball.grid_clubs(text) == ["Śląsk", "Beta", "Gamma", "Delta"]
    assert wikifootball.parse_grid(text) == [("Śląsk", "Beta", 2, 1)]


def test_a_club_wrapped_in_a_presentation_template_is_still_a_club():
    """Moldova, Scotland, San Marino and Wales all write
    `|name_X={{nowrap|[[Real Club|Club]]}}`, and a reader that stops at the
    first pipe calls that club "{{nowrap". All four reported it as an
    unresolved club name and none of them armed."""
    from model import wikifootball
    assert wikifootball.name_variants("{{nowrap|[[Sheriff Tiraspol|Sheriff]]}}") == (
        "Sheriff", "Sheriff Tiraspol")
    assert wikifootball.name_variants("{{nowrap|Zimbru Chișinău}}") == ("Zimbru Chișinău",)
    text = ("|team1=SHE|team2=ZIM|team3=CCC|team4=DDD\n"
            "|name_SHE={{nowrap|[[Sheriff Tiraspol|Sheriff]]}}\n"
            "|name_ZIM={{nowrap|Zimbru Chișinău}}\n"
            "|name_CCC=Gamma\n|name_DDD=Delta\n"
            "|match_SHE_ZIM=2–1\n")
    assert wikifootball.grid_clubs(text)[:2] == ["Sheriff", "Zimbru Chișinău"]
    assert wikifootball.parse_grid(text) == [("Sheriff", "Zimbru Chișinău", 2, 1)]


#: The shape eleven leagues actually use, taken from what the runner printed of
#: Andorra's and Bosnia's articles. Two differences from the style this reader
#: was written against, and either one on its own is enough to lose the league:
#: the entrants are one comma separated line rather than a parameter each, and
#: a result cell is named after the pair with no `match_` in front of it.
FBR = """{{#invoke:sports results|main |source=[x] |update=complete
|matches_style=FBR |solid_cell=grey |legs=2 |a_note=yes
|team_order=BOR, GOŠ, IGM, POS
|name_BOR = [[FK Borac Banja Luka|Borac]]
|name_GOŠ = [[NK GOŠK Gabela|GOŠK]]
|name_IGM = [[FK Igman Konjic|Igman]]
|name_POS = [[HŠK Posušje|Posušje]]
|win_BOR=9 |draw_BOR=2 |loss_BOR=1 |gf_BOR=24 |ga_BOR=8
|BOR_GOŠ=2–1
|GOŠ_BOR=0–0
|IGM_POS=3–1
}}"""


def test_the_other_cell_style_is_read_and_the_table_beside_it_is_not():
    """Eleven leagues reported no results grid while carrying one.

    The trap is that the same article also runs a league table whose parameters
    look exactly like result cells: `win_BOR`, `gf_BOR`, `ga_BOR`. What keeps
    those out is the rule every cell goes through, that both halves of the pair
    must be codes the article itself declared as entrants.
    """
    from model import wikifootball
    assert wikifootball.grid_clubs(FBR) == ["Borac", "GOŠK", "Igman", "Posušje"]
    got = wikifootball.parse_grid(FBR)
    assert ("Borac", "GOŠK", 2, 1) in got
    assert ("GOŠK", "Borac", 0, 0) in got
    assert ("Igman", "Posušje", 3, 1) in got
    assert len(got) == 3, "a table parameter is not a result"
    assert not any("BOR" == h or "win" in h for h, _, _, _ in got)


def test_a_comma_separated_entrant_list_still_gives_every_club():
    """A club missing from the entrant list loses every one of its fixtures
    silently, so the two ways of writing that list both have to be read."""
    from model import wikifootball
    assert wikifootball._order(FBR) == ["BOR", "GOŠ", "IGM", "POS"]
    assert wikifootball._order("|team1=AAA|team2=BBB") == []


#: Andorra's own markup. Ten clubs meeting three times, so every cell carries
#: the round it belongs to, and a pair appears once per round.
ROUNDS = """{{#invoke:sports results|main |matches_style=FBR |legs=2
|team_order=ACE, ESP, INT, MAS
|name_ACE=[[Atlètic Club d'Escaldes]]
|name_ESP=[[CF Esperança d'Andorra|Esperança]]
|name_INT=[[Inter Club d'Escaldes]]
|name_MAS=[[FS La Massana|La Massana]]
|match1_ACE_ESP=5–1 |match1_ACE_INT=3–1 |match1_ESP_ACE=0–5
|match2_ACE_ESP=1–1
| match1_MAS_INT = 0–3
}}"""


def test_a_cell_numbered_by_its_round_is_still_a_result():
    """Eleven leagues reported no results grid while carrying a full one.

    A league whose clubs meet three or four times cannot name a cell after the
    pair alone, because several cells would share the name, so the round goes
    in: match1_ACE_ESP rather than match_ACE_ESP. A reader that insisted on the
    bare form found nothing in any of them.
    """
    from model import wikifootball
    got = wikifootball.parse_grid(ROUNDS)
    assert ("Atlètic Club d'Escaldes", "Esperança", 5, 1) in got
    assert ("Esperança", "Atlètic Club d'Escaldes", 0, 5) in got
    assert ("La Massana", "Inter Club d'Escaldes", 0, 3) in got, "spaces around ="
    # The same pair in a later round is a different match, not a duplicate.
    assert got.count(("Atlètic Club d'Escaldes", "Esperança", 5, 1)) == 1
    assert ("Atlètic Club d'Escaldes", "Esperança", 1, 1) in got
    assert len(got) == 5
