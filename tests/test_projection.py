"""A projected final table for a league whose fixture list nobody publishes.

There is nothing to fetch here and nothing to mock: a projection is built from a
club list, the results so far, and a rating fit, and all three can be taken from
a season this project already holds. So these tests take a real finished season,
stop it partway, and check that what comes out is a forecast of the season that
actually happened -- the right clubs, the right matches left, a table that adds
up, and probabilities that behave like probabilities.
"""
import sys

import pytest

sys.path.insert(0, ".")

from model import europe, fetch, projection, ratings, roundrobin      # noqa: E402
from model.parse import TeamRegistry, parse_openfootball              # noqa: E402


@pytest.fixture(scope="module")
def season():
    """Norway's last complete season: 16 clubs, 240 matches, no split."""
    reg = TeamRegistry()
    src = europe.BY_ASSOC["NOR"]
    for label in reversed(src.seasons):
        text = fetch.get(src.url(label), required=False, tries=1)
        if not text:
            continue
        ms = [m for m in parse_openfootball(text, label, reg, comp=src.group)
              if m.played]
        clubs = sorted({m.home for m in ms} | {m.away for m in ms})
        if len(ms) == roundrobin.season_size(len(clubs)):
            ms.sort(key=lambda m: (m.date, m.home))
            return reg, label, clubs, ms
    pytest.skip("Norway's GitHub feed is not reachable in this checkout")


def _fit(reg, clubs, matches, ref):
    return ratings.fit(matches, clubs, ref)


def _projection(label, clubs, played, source="wikipedia"):
    return projection.Projection(
        slug="eliteserien", name="Eliteserien", country="Norway",
        season=label, source=source, clubs=clubs, played=played)


# ------------------------------------------------------------------ the fixtures
def test_the_matches_left_plus_the_matches_played_are_the_whole_season(season):
    _, label, clubs, ms = season
    for cut in (0, 60, 120, 239):
        fx = projection.fixtures(clubs, ms[:cut], when=ms[0].date)
        assert len(fx) == roundrobin.season_size(len(clubs))
        assert sum(1 for f in fx if f.played) == cut
        for club in clubs:
            assert sum(1 for f in fx if f.home == club) == len(clubs) - 1
            assert sum(1 for f in fx if f.away == club) == len(clubs) - 1


def test_no_derived_fixture_claims_to_know_when_it_is(season):
    """The one thing a results grid cannot say. Every derived fixture carries a
    date because the pipeline sorts by one, and every derived fixture is flagged
    so that nothing prints it."""
    _, _, clubs, ms = season
    fx = projection.fixtures(clubs, ms[:100], when=ms[99].date)
    for f in fx:
        if not f.played:
            assert f.extra.get("date_unknown") is True
            assert f.extra.get("date_approx") is True


# ------------------------------------------------------------------ the table
def test_the_table_so_far_is_the_table_the_season_had(season):
    _, label, clubs, ms = season
    rows = projection.table(clubs, ms)
    assert sum(r["pld"] for r in rows.values()) == 2 * len(ms)
    assert sum(r["gf"] for r in rows.values()) == sum(r["ga"] for r in rows.values())
    for r in rows.values():
        assert r["pld"] == r["w"] + r["d"] + r["l"] == 2 * (len(clubs) - 1)
        assert r["pts"] == 3 * r["w"] + r["d"]


def test_a_club_that_has_not_kicked_off_yet_is_still_in_the_table(season):
    _, _, clubs, ms = season
    rows = projection.table(clubs, [])
    assert len(rows) == len(clubs)
    assert all(r["pld"] == 0 and r["pts"] == 0 for r in rows.values())


# ------------------------------------------------------------------ the forecast
@pytest.mark.parametrize("cut", [0, 120, 239])
def test_a_projection_is_a_probability_distribution_over_the_table(season, cut):
    reg, label, clubs, ms = season
    played = ms[:cut]
    ref = ms[cut if cut < len(ms) else -1].date
    fit = _fit(reg, clubs, ms[:max(cut, 40)], ref)
    out = projection.run(_projection(label, clubs, played), fit, n_sims=2000)

    assert out["kind"] == "projection"
    assert out["fixtures_known"] is False
    assert out["matches_played"] == cut
    assert out["matches_total"] == roundrobin.season_size(len(clubs))
    assert len(out["teams"]) == len(clubs)
    assert abs(sum(t["title"] for t in out["teams"]) - 1) < 0.02
    assert abs(sum(t["last"] for t in out["teams"]) - 1) < 0.02
    for t in out["teams"]:
        assert len(t["position"]) == len(clubs)
        assert abs(sum(t["position"]) - 1) < 0.02
        assert t["pts_lo"] <= t["pts"] <= t["pts_hi"]
        assert t["pts"] >= t["now"] - 0.01, "a club cannot lose points it has"
    # Every finishing position is taken exactly once per simulated season.
    for pos in range(len(clubs)):
        assert abs(sum(t["position"][pos] for t in out["teams"]) - 1) < 0.02


def test_a_season_already_finished_projects_itself(season):
    """With nothing left to play the projection is not a forecast at all, it is
    the table. Anything else means results are being simulated twice."""
    reg, label, clubs, ms = season
    fit = _fit(reg, clubs, ms, ms[-1].date)
    out = projection.run(_projection(label, clubs, ms), fit, n_sims=500)
    rows = projection.table(clubs, ms)
    order = sorted(rows.values(), key=lambda r: (-r["pts"], -r["gd"], -r["gf"]))
    assert out["teams"][0]["id"] == order[0]["team"]
    assert out["teams"][0]["title"] > 0.99
    for t in out["teams"]:
        assert abs(t["pts"] - t["now"]) < 0.01


def test_the_projection_publishes_no_line_it_cannot_cite(season):
    """No European place, no relegation line, no points threshold for either.
    The simulator computes all three against a placeholder competition; none of
    it may reach the file, because this project has no source for these leagues'
    rules and a drawn line reads as a claim."""
    reg, label, clubs, ms = season
    fit = _fit(reg, clubs, ms[:120], ms[119].date)
    out = projection.run(_projection(label, clubs, ms[:120]), fit, n_sims=500)
    banned = {"ucl", "europa", "relegation", "lines", "ucl_places",
              "releg_places", "releg_note", "top5", "safety"}
    seen = set(out) | {k for t in out["teams"] for k in t}
    assert not (seen & banned), sorted(seen & banned)
