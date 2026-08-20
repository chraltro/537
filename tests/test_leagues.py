"""Tests for the five-league generalisation.

The Premier League pipeline was correct because one set of constants happened to
describe it. Four more leagues means those constants are now parameters, and the
failure mode that matters is a silent one: a league built with another league's
shape, or a club whose two feeds never met and quietly became two clubs with
half a rating each. Everything here is aimed at that.
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import config, insight, leagues, priors, ratings, run, simulate  # noqa: E402
from model.data import Dataset                                          # noqa: E402
from model.parse import TeamRegistry, normalise                         # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLUGS = [lg.slug for lg in leagues.LEAGUES]
#: The five the model was tuned on. Several rules below are true of those and
#: deliberately not of the three added later: only the big five have a
#: football-datasets mirror, twenty-plus seasons of history, or a Champions
#: League line worth pinning to a UEFA decision.
BIG5 = [lg.slug for lg in leagues.BIG_FIVE]

_CACHE: dict[str, Dataset] = {}


@pytest.fixture(scope="session")
def dataset():
    """One loaded Dataset per league, built at most once for the whole session."""
    def get(slug: str) -> Dataset:
        if slug not in _CACHE:
            _CACHE[slug] = Dataset(leagues.get(slug)).load()
        return _CACHE[slug]
    return get


# ---------------------------------------------------------------- registry
def test_registry_is_internally_consistent():
    assert len(leagues.BIG_FIVE) == 5
    assert len(leagues.LEAGUES) == 9
    assert len({lg.slug for lg in leagues.LEAGUES}) == len(leagues.LEAGUES)
    assert set(leagues.BIG_FIVE) <= set(leagues.LEAGUES)
    for lg in leagues.LEAGUES:
        # A double round robin, and nothing else, is a season.
        assert lg.n_matches == lg.n_teams * (lg.n_teams - 1), lg.slug
        assert 1 <= lg.ucl_places < lg.n_teams - lg.releg_places
        assert lg.releg_places >= 1
        assert lg.of_top
        # Every competition names a second tier except Belgium, where
        # openfootball has no `be2` at any season. That is recorded in the
        # registry as an empty string rather than a wrong URL, and the
        # promoted-club correction falls back accordingly.
        assert lg.of_second or lg.slug == "pro-league", lg.slug
        # Only the big five are in the results mirror -- it has exactly five
        # league directories -- so everything else must say it reads goals only.
        if lg in leagues.BIG_FIVE:
            assert lg.source == "mirror" and lg.fd_dir, lg.slug
        else:
            assert lg.source == "openfootball" and not lg.fd_dir, lg.slug


def test_promotion_leagues_declare_their_playoff_band():
    """A second tier is read against a promotion line and a play-off band, and
    the band has to fit between that line and the drop."""
    for lg in leagues.LEAGUES:
        if lg.kind != "promotion":
            continue
        assert lg.advance_direct == lg.ucl_places, lg.slug
        assert lg.advance_playoff and lg.advance_playoff >= 1, lg.slug
        assert lg.advance_direct + lg.advance_playoff < lg.n_teams - lg.releg_places
        assert "advance_direct" in lg.public() and "kind" in lg.public()


def test_2026_27_season_shapes_match_the_contract():
    """380/380/380/306/306 for the big five, in registry order, and the four
    competitions added after them: 306/306/306/552. Belgium expanded from 16
    clubs to 18 for 2026-27, so its shape is the new one, not the old."""
    assert [lg.n_matches for lg in leagues.BIG_FIVE] == [380, 380, 380, 306, 306]
    assert [lg.n_teams for lg in leagues.BIG_FIVE] == [20, 20, 20, 18, 18]
    extra = [lg for lg in leagues.LEAGUES if lg not in leagues.BIG_FIVE]
    assert [lg.slug for lg in extra] == ["eredivisie", "primeira-liga",
                                         "pro-league", "championship"]
    assert [lg.n_matches for lg in extra] == [306, 306, 306, 552]
    assert [lg.n_teams for lg in extra] == [18, 18, 18, 24]


def test_qualification_rules_are_the_verified_2026_27_ones():
    """England and Spain took both UEFA European Performance Spots, so they have
    five; Italy and Germany have four; France is the fifth association and sends
    three straight to the league phase. Play-off relegation only in Germany and
    France."""
    ucl = {lg.slug: lg.ucl_places for lg in leagues.BIG_FIVE}
    assert ucl == {"premier-league": 5, "la-liga": 5, "serie-a": 4,
                   "bundesliga": 4, "ligue-1": 3}
    releg = {lg.slug: lg.releg_places for lg in leagues.BIG_FIVE}
    assert releg == {"premier-league": 3, "la-liga": 3, "serie-a": 3,
                     "bundesliga": 2, "ligue-1": 2}
    with_playoff = {lg.slug for lg in leagues.BIG_FIVE if lg.releg_note}
    assert with_playoff == {"bundesliga", "ligue-1"}
    # The Netherlands and Portugal each send champions and runners-up to the
    # league phase; this repository's own data/europe/participants-2026-27.json
    # lists PSV and Feyenoord, and Porto and Sporting CP, among the 36.
    assert leagues.EREDIVISIE.ucl_places == 2
    assert leagues.PRIMEIRA_LIGA.ucl_places == 2


def test_source_urls_follow_the_documented_shapes():
    assert leagues.PREMIER_LEAGUE.of_url("2026-27", "top").endswith(
        "/openfootball/england/master/2026-27/1-premierleague.txt")
    assert leagues.LA_LIGA.of_url("2020-21", "second").endswith(
        "/openfootball/espana/master/2020-21/2-liga2.txt")
    # France's repository nests the country a second time and puts the season in
    # the filename, which is the whole reason these are templates.
    assert leagues.LIGUE_1.of_url("2026-27", "top").endswith(
        "/openfootball/france/master/france/2026-27_fr1.txt")
    assert leagues.LIGUE_1.of_url("2026-27", "second").endswith(
        "/openfootball/france/master/france/2026-27_fr2.txt")
    assert leagues.SERIE_A.fd_csv_url("2526").endswith("/datasets/serie-a/season-2526.csv")


def test_season_lists_end_at_the_forecast_season():
    lg = leagues.LA_LIGA
    codes = lg.fd_season_codes("2026-27")
    assert codes[0] == "0001" and codes[-1] == "2627"
    labels = lg.second_season_labels("2026-27")
    assert labels[0] == "2012-13" and labels[-1] == "2026-27"


def test_unknown_league_fails_loudly():
    with pytest.raises(KeyError):
        leagues.get("not-a-league")


def test_config_constants_still_describe_the_premier_league():
    """Module-level constants are aliases, not a second source of truth."""
    pl = leagues.PREMIER_LEAGUE
    assert (config.N_TEAMS, config.N_MATCHES) == (pl.n_teams, pl.n_matches)
    assert config.UCL_PLACES == pl.ucl_places
    assert config.RELEGATION_PLACES == pl.releg_places
    assert config.PL_SEASONS == pl.fd_season_codes(config.SEASON)
    assert config.CH_SEASONS == pl.second_season_labels(config.SEASON)


# ---------------------------------------------------------------- naming
@pytest.mark.parametrize("a,b", [
    ("Malaga", "Málaga CF"),
    ("Ath Madrid", "Atlético de Madrid"),
    ("Espanol", "RCD Espanyol"),
    ("Paris SG", "Paris Saint-Germain"),
    ("Ein Frankfurt", "Eintracht Frankfurt"),
    ("M'gladbach", "Borussia Mönchengladbach"),
    ("Inter", "FC Internazionale Milano"),
    ("Sp Gijon", "Sporting Gijón"),
    ("St Etienne", "AS Saint-Étienne"),
    ("FC Koln", "1. FC Köln"),
    ("Bayern Munich", "FC Bayern München"),
    ("La Coruna", "RC Deportivo La Coruña"),
])
def test_cross_feed_spellings_resolve_together(a, b):
    reg = TeamRegistry()
    assert reg.resolve(a) == reg.resolve(b), f"{a!r} and {b!r} must be the same club"


def test_normalise_folds_accents():
    assert normalise("Málaga CF") == normalise("Malaga CF")
    assert normalise("Borussia Mönchengladbach") == "borussia monchengladbach"
    assert normalise("Deportivo Alavés") == "deportivo alaves"
    # Folding must not merge clubs that are genuinely different.
    assert normalise("Milan") != normalise("Inter")


def test_clubs_with_digits_in_their_name_are_not_dropped():
    """'Como 1907' and '1. FC Köln' are clubs, not corrupt lines -- the England
    parser used to reject any side containing a digit."""
    from model.parse import parse_openfootball
    reg = TeamRegistry()
    text = "\n".join([
        "▪ Matchday 1",
        "  Sat Aug 22 2026",
        "    15:30  1. FSV Mainz 05         v SC Paderborn 07",
        "    15:30  Bologna FC 1909         v Como 1907                2-1",
        "    15:30  Bayer 04 Leverkusen  0-0  TSG 1899 Hoffenheim",
    ])
    got = parse_openfootball(text, "2026-27", reg)
    assert len(got) == 3
    assert got[1].hg == 2 and got[1].ag == 1
    assert got[2].home == reg.resolve("Bayer Leverkusen")
    assert got[2].away == reg.resolve("Hoffenheim")


def test_team_meta_covers_every_club_of_every_league(dataset):
    """The completeness test: no auto-registered club may reach a fixture list."""
    missing = {}
    for slug in SLUGS:
        ds = dataset(slug)
        auto = [t for t in ds.teams if ds.reg.meta.get(t, {}).get("auto")]
        if auto:
            missing[slug] = auto
    assert not missing, f"unmapped 2026-27 clubs: {missing}"


@pytest.mark.parametrize("slug", SLUGS)
def test_short_codes_are_unique_within_a_league(dataset, slug):
    ds = dataset(slug)
    counts = collections.Counter(ds.reg.meta[t]["short"] for t in ds.teams)
    assert [c for c in counts if counts[c] > 1] == []
    for t in ds.teams:
        m = ds.reg.meta[t]
        assert len(m["short"]) == 3, f"{t} short code {m['short']!r}"
        for key in ("primary", "secondary"):
            assert len(m[key]) == 7 and m[key][0] == "#", f"{t} {key} {m[key]!r}"


# ---------------------------------------------------------------- fixtures
@pytest.mark.parametrize("slug", SLUGS)
def test_fixture_list_is_a_complete_season(dataset, slug):
    ds = dataset(slug)
    lg = ds.league
    assert len(ds.fixtures) == lg.n_matches
    assert len(ds.teams) == lg.n_teams
    for t in ds.teams:
        home = sum(1 for m in ds.fixtures if m.home == t)
        away = sum(1 for m in ds.fixtures if m.away == t)
        assert home == away == lg.n_teams - 1, f"{slug}: {t} plays {home}H/{away}A"
    pairs = [(m.home, m.away) for m in ds.fixtures]
    assert len(set(pairs)) == len(pairs)


@pytest.mark.parametrize("slug", SLUGS)
def test_history_is_deep_enough_to_fit(dataset, slug):
    ds = dataset(slug)
    lg = leagues.get(slug)
    # The mirror carries the big five back to 1993-94 and the pipeline reads it
    # from 2000-01. openfootball's country files start much later, so the bar is
    # "deep enough to fit", not "as deep as the Premier League".
    # Belgium is the thin one and is allowed to be, on measured grounds:
    # openfootball has no `be1` at all for 2020-21 or 2022-23, and its 2025-26
    # file stopped at 121 of 240 matches when the maintainer moved on to the
    # Wallonian provincial leagues. Six seasons is what exists upstream.
    want = 20 if lg.source == "mirror" else (6 if slug == "pro-league" else 8)
    assert len({m.season for m in ds.top}) >= want, f"{slug}: too few seasons"
    # Enough second-tier football to rate a promoted club at all. The big five
    # have a decade or more; the Eredivisie and Primeira Liga have five seasons
    # upstream, which is thin on purpose -- `priors.regress` falls back to the
    # measured Premier League constants rather than trusting a slope fitted on
    # a handful of promoted clubs, and says so in the output.
    if not lg.of_second:
        # Belgium: no second tier upstream, so there is nothing to load and the
        # promoted-club correction is a declared fallback, not a measurement.
        assert len(ds.second) == 0, f"{slug}: no second tier is declared"
        return
    floor = 2000 if lg.source == "mirror" else 1200
    assert len(ds.second) > floor, f"{slug}: second tier needed to rate promoted clubs"


# ---------------------------------------------------------------- goals-only
def test_some_leagues_genuinely_lack_early_shot_data(dataset):
    """This is the premise of the next test, so it is checked rather than assumed:
    the mirror carries no shots at all for La Liga, Serie A or Ligue 1 before
    2005-06, while the Premier League has them from 2000-01."""
    def with_shots(ds, season):
        ms = [m for m in ds.top if m.season == season]
        return sum(1 for m in ms if m.hst is not None), len(ms)

    for slug in ("la-liga", "serie-a", "ligue-1"):
        got, total = with_shots(dataset(slug), "2002-03")
        assert total > 250 and got == 0, f"{slug} 2002-03: {got}/{total} with shots"
    got, total = with_shots(dataset("premier-league"), "2002-03")
    assert got == total > 250


def test_fitting_tolerates_a_league_with_goals_only(dataset):
    """A season with no shot columns must still produce usable ratings.

    The blend is coverage-weighted, so a club the shot fit never saw keeps its
    goals-only rating instead of being dragged to the league average by a fit
    built on no evidence. Nothing is allowed to raise, and nothing is allowed to
    come back flat.
    """
    ds = dataset("la-liga")
    early = [m for m in ds.top if m.season < "2005-06"]
    assert early and all(m.hst is None for m in early)
    teams = sorted({m.home for m in early})
    shot_conv = ratings.fit_shot_conversion(early)      # falls back to defaults
    fit = ratings.fit(early, teams, dt.date(2005, 7, 1), shot_conv=shot_conv)
    assert np.isfinite(fit.att).all() and np.isfinite(fit.dfn).all()
    assert fit.att.std() > 0.05, "goals-only fit collapsed to a flat league"
    lh, la = fit.lambdas(teams[0], teams[1])
    assert 0.1 < lh < 6 and 0.1 < la < 6
    # And the mixed case: shots for the later seasons only, none for the earlier.
    mixed = [m for m in ds.top if m.season < "2008-09"]
    pool = sorted({m.home for m in mixed} | {m.away for m in mixed})
    f2 = ratings.fit(mixed, pool, dt.date(2008, 7, 1),
                     shot_conv=ratings.fit_shot_conversion(mixed))
    assert np.isfinite(f2.att).all() and f2.att.std() > 0.05


# ---------------------------------------------------------------- simulation
@pytest.mark.parametrize("slug", SLUGS)
def test_simulation_probabilities_are_coherent_for_every_league(dataset, slug):
    """Exactly one champion, `ucl_places` qualifiers and `releg_places`
    relegations per simulated season -- with each league's own numbers."""
    ds = dataset(slug)
    lg = ds.league
    fit = ratings.fit([m for m in ds.top if m.season >= "2024-25"],
                      ds.teams, ds.kickoff)
    sim = simulate.simulate_season(fit, ds.fixtures, ds.teams, league=lg,
                                   n_sims=4000, scenarios=20)
    assert abs(sim["title"].sum() - 1) < 0.02
    assert abs(sim["ucl"].sum() - lg.ucl_places) < 0.05
    assert abs(sim["relegation"].sum() - lg.releg_places) < 0.05
    assert np.allclose(sim["position"].sum(axis=1), 1, atol=1e-6)
    assert np.allclose(sim["position"].sum(axis=0), 1, atol=1e-6)
    # The points lines are read off the boundary positions of the same tables.
    # Where a competition has exactly one European place -- Belgium -- the
    # title line and the qualification line are read off the same boundary
    # position, so they are equal by construction rather than ordered.
    if lg.ucl_places > 1:
        assert sim["lines"]["title"]["p50"] > sim["lines"]["top5"]["p50"]
    else:
        assert sim["lines"]["title"]["p50"] == sim["lines"]["top5"]["p50"]
    assert sim["lines"]["top5"]["p50"] > sim["lines"]["safety"]["p50"]


def test_the_ucl_line_actually_moves_with_the_league(dataset):
    """Same fixtures, same ratings, different rule: a four-place league must
    hand out fewer Champions League slots than a five-place one."""
    ds = dataset("la-liga")
    fit = ratings.fit([m for m in ds.top if m.season >= "2024-25"],
                      ds.teams, ds.kickoff)
    kw = dict(n_sims=4000, scenarios=20, seed=11)
    five = simulate.simulate_season(fit, ds.fixtures, ds.teams,
                                    league=leagues.LA_LIGA, **kw)
    four = simulate.simulate_season(fit, ds.fixtures, ds.teams,
                                    league=leagues.SERIE_A, **kw)
    assert abs(five["ucl"].sum() - 5) < 0.05
    assert abs(four["ucl"].sum() - 4) < 0.05
    assert (five["ucl"] >= four["ucl"] - 1e-9).all()


def test_strength_of_schedule_covers_every_fixture_in_every_league(dataset):
    for slug in SLUGS:
        ds = dataset(slug)
        n = ds.league.n_teams
        spi = {t: 50.0 + i for i, t in enumerate(ds.teams)}
        sos = insight.strength_of_schedule(ds.fixtures, ds.teams, spi, 0.18)
        for t in ds.teams:
            assert len(sos[t]["fixtures"]) == 2 * (n - 1), slug
        assert sorted(sos[t]["rank"] for t in ds.teams) == list(range(1, n + 1))


# ---------------------------------------------------------------- priors
def test_market_anchor_is_per_league_and_optional():
    assert os.path.exists(priors.market_path(leagues.PREMIER_LEAGUE)), \
        "the Premier League anchor moved to data/market_priors/premier-league.json"
    assert priors.load_market(priors.market_path(leagues.PREMIER_LEAGUE))["title"]
    for lg in leagues.LEAGUES:
        if lg is leagues.PREMIER_LEAGUE:
            continue
        # Absent file must read as "no anchor", not as an error.
        assert priors.load_market(priors.market_path(lg)) == {}
    assert not os.path.exists(os.path.join(HERE, "data", "market_priors.json")), \
        "the old flat market_priors.json should be gone"


def test_market_weight_uses_the_league_season_length():
    """The anchor decays over matchweeks, not matches. Ninety matches is ten
    matchweeks of an 18-club league but only nine of a 20-club one, so the same
    match count must leave the smaller league with less weight -- exactly none,
    at MARKET_DECAY_MW."""
    played = 90
    assert priors.market_weight(played, leagues.BUNDESLIGA) == 0.0
    assert priors.market_weight(played, leagues.PREMIER_LEAGUE) > 0.0
    # Equal matchweeks, equal weight, whatever the division's size.
    assert priors.market_weight(9 * 5, leagues.BUNDESLIGA) == pytest.approx(
        priors.market_weight(10 * 5, leagues.PREMIER_LEAGUE))
    assert priors.market_weight(0, leagues.LIGUE_1) == pytest.approx(config.MARKET_WEIGHT)
    assert priors.market_weight(10_000, leagues.SERIE_A) == 0.0


def test_thin_second_tier_falls_back_to_the_premier_league_constants():
    """Below MIN_PAIRS the league's own regression is noise, so the Premier
    League's measured carryover is used and the result says where it came from."""
    thin = [(0.2, 0.1), (-0.3, -0.2), (0.5, 0.4)]
    got = priors.regress(thin, "promoted", "serie-a")
    assert len(thin) < priors.MIN_PAIRS
    assert got["source"] == "premier-league"
    assert got["n"] == 3
    assert {k: v for k, v in got.items() if k != "reason"} == {
        **priors.PL_FALLBACK["promoted"], "n": 3}
    assert got["reason"] == "too few pairs"
    assert 0.2 < got["slope"] < 0.6, "a promoted club keeps a fraction of its edge"

    # With enough pairs the league measures its own: y = 0.5x exactly.
    plenty = [(x / 10, x / 20) for x in range(-6, 6)]
    got = priors.regress(plenty, "promoted", "serie-a")
    assert got["source"] == "serie-a"
    assert got["slope"] == pytest.approx(0.5, abs=1e-6)
    assert got["intercept"] == pytest.approx(0.0, abs=1e-6)


def test_an_implausible_slope_is_clamped_not_discarded():
    """Enough pairs is not the same as a believable answer. Eight seasons of a
    smaller league can fit a promoted-club slope of 3.0, which would triple a
    promoted club's rating gap instead of shrinking it.

    Above the band the slope is pulled back to the ceiling and the fit is kept:
    the intercept was measured on the right population -- promoted clubs in this
    competition -- and throwing it away to take the Premier League's would
    discard a good number along with a bad one. The output has to say what was
    measured and what was done about it, so the site can too.
    """
    silly = [(0.1 * i, 0.3 * i) for i in range(-6, 6)]      # slope 3.0
    got = priors.regress(silly, "promoted", "primeira-liga")
    hi = priors.SLOPE_BAND["promoted"][1]
    assert len(silly) >= priors.MIN_PAIRS
    assert got["source"] == "primeira-liga"
    assert got["measured_slope"] > hi
    assert got["slope"] == pytest.approx(hi)
    assert "clamped" in got["reason"]


def test_a_backwards_slope_is_discarded_entirely():
    """Below the floor the fit is not noisy, it is pointing the wrong way: a
    club's preseason edge predicting the opposite of what happens. There is
    nothing in it worth keeping, so the whole correction falls back."""
    backwards = [(0.1 * i, -0.3 * i) for i in range(-6, 6)]      # slope -3.0
    got = priors.regress(backwards, "promoted", "primeira-liga")
    assert len(backwards) >= priors.MIN_PAIRS
    assert got["source"] == "premier-league"
    assert got["measured_slope"] < priors.SLOPE_BAND["promoted"][0]
    assert 0 < got["slope"] < 1


def test_every_league_calibration_reports_its_source(dataset):
    """A cached calibration file must name the league it was measured on, so a
    fallback can never be mistaken for a measurement."""
    for slug in SLUGS:
        ds = dataset(slug)
        cal = priors.calibrate(ds, (0.30, 0.03))
        for key in ("continuing", "promoted"):
            assert cal[key]["source"] in (slug, "premier-league")
            assert 0.0 < cal[key]["slope"] < 1.5, (slug, key)


# ---------------------------------------------------------------- outputs
def test_built_output_matches_the_manifest():
    """Whatever the last run produced must agree with the registry: a league
    marked ready has a forecast, and that forecast carries its own league block."""
    path = os.path.join(HERE, "site", "data", "leagues.json")
    if not os.path.exists(path):
        pytest.skip("pipeline has not been run in this checkout")
    man = json.load(open(path))
    assert man["default"] == leagues.DEFAULT.slug
    assert [e["slug"] for e in man["leagues"]] == \
        [lg.slug for lg in leagues.LEAGUES + leagues.EUROPEAN]
    for entry in man["leagues"]:
        lg = leagues.get(entry["slug"])
        assert entry["n_teams"] == lg.n_teams
        assert entry["ucl_places"] == lg.ucl_places
        assert entry["releg_places"] == lg.releg_places
        assert entry["releg_note"] == lg.releg_note
        fc_path = os.path.join(HERE, "site", "data", lg.slug, "forecast.json")
        # A forecast on disk is not the same thing as a ready league: the
        # Champions League directory holds a REPLAY of a finished season, built
        # so the site has something to develop against before the 27 August
        # draw exists. It is stamped as a replay and must never read as live.
        replay = (os.path.exists(fc_path)
                  and json.load(open(fc_path)).get("replay") is not None)
        assert entry["ready"] == (os.path.exists(fc_path) and not replay)
        if not entry["ready"]:
            continue
        fc = json.load(open(fc_path))
        assert fc["league"] == lg.public()
        assert len(fc["teams"]) == lg.n_teams
        assert abs(sum(t["title"] for t in fc["teams"]) - 1) < 0.02
        assert abs(sum(t["ucl"] for t in fc["teams"]) - lg.ucl_places) < 0.02
        assert abs(sum(t["releg"] for t in fc["teams"]) - lg.releg_places) < 0.02
        si = json.load(open(os.path.join(HERE, "site", "data", lg.slug,
                                         "sim_input.json")))
        assert (si["n_teams"], si["ucl_places"], si["releg_places"]) == \
            (lg.n_teams, lg.ucl_places, lg.releg_places)


def test_the_flat_duplicates_of_one_league_are_gone():
    """Every page reads `site/data/<slug>/<name>.json`. The flat copies at
    `site/data/<name>.json` were a second Premier League that nothing fetched
    and every build rewrote, so they are deleted -- and stay deleted, because a
    stale duplicate of a forecast is worse than no duplicate at all.

    The site-wide files are a different thing and must survive: they are not
    scoped to a competition, which is their whole point.
    """
    base = os.path.join(HERE, "site", "data")
    if not os.path.exists(os.path.join(base, "premier-league", "forecast.json")):
        pytest.skip("pipeline has not been run in this checkout")
    for name in run.RETIRED_FLAT_FILES:
        assert not os.path.exists(os.path.join(base, name)), \
            f"{name} is a flat duplicate and should have been removed"
    for name in ("leagues.json", "global.json", "h2h.json"):
        assert os.path.exists(os.path.join(base, name)), \
            f"{name} belongs to the whole site and must not be swept up"


# ------------------------------------------------- arriving from above
def test_only_a_second_tier_loads_the_division_above(dataset):
    """A top flight loads the tier below to rate the clubs coming up. A second
    tier needs the mirror image, because three of its clubs each season arrive
    from above with no recent record in it at all."""
    for slug in SLUGS:
        ds = dataset(slug)
        lg = leagues.get(slug)
        if lg.above_slug:
            assert ds.above, f"{slug}: nothing loaded from {lg.above_slug}"
            # Deep enough to actually rate a club that has just come down.
            assert len({m.season for m in ds.above}) >= 10, slug
        else:
            assert ds.above == [], f"{slug}: has no division above but loaded one"


def test_the_corpus_is_assembled_in_one_place(dataset):
    """Four call sites used to build the fit corpus by hand and had to agree.
    `before` is now the only one that decides what is in it."""
    ds = dataset("championship")
    cutoff = dt.date(2026, 7, 1)
    got = ds.before(cutoff)
    assert all(m.date < cutoff for m in got)
    assert len(got) == sum(1 for m in ds.top + ds.second + ds.above if m.date < cutoff)
    # The tier above is genuinely in there, which is the whole point.
    assert any(m in got for m in ds.above)


def test_relegated_clubs_are_told_apart_from_promoted_ones(dataset):
    """The bug this pins: every club new to a division was called 'promoted',
    so a club dropping out of the Premier League was shrunk by a correction
    measured on clubs arriving from League One. In 2025-26 the Premier League
    relegated West Ham, Burnley and Wolves, and the Championship promoted
    Cardiff, Lincoln and Bolton."""
    ds = dataset("championship")
    teams = ds.teams
    how = priors.arrivals(ds, teams, "2025-26")
    down = {t for t, v in how.items() if v == "down"}
    up = {t for t, v in how.items() if v == "up"}
    assert {"west-ham", "burnley", "wolves"} <= down, sorted(down)
    assert down & up == set()
    assert "west-ham" not in up
    # Everyone else was already there.
    assert how["millwall"] == "stayed"


def test_a_top_flight_never_sees_an_arrival_from_above(dataset):
    """`ds.above` is empty there, so the three-way split collapses back to the
    two-way one and nothing about the big five changes."""
    ds = dataset("premier-league")
    how = priors.arrivals(ds, ds.teams, "2025-26")
    assert set(how.values()) <= {"stayed", "up"}


def test_each_kind_of_arrival_gets_its_own_correction():
    """Routing, without the network: a relegated club must not be handed the
    promoted club's level shift."""
    cal = {
        "continuing": {"slope": 1.0, "intercept": 0.0},
        "promoted": {"slope": 1.0, "intercept": -1.0},
        "relegated": {"slope": 1.0, "intercept": +0.5},
    }

    class FakeFit:
        index = {"stay": 0, "came_up": 1, "came_down": 2}

        @staticmethod
        def offence(t):
            return np.e          # every club identical, so only the correction shows

        @staticmethod
        def defence(t):
            return 1.0

    class FakeDs:
        top = [type("M", (), {"home": "stay", "season": "2025-26"})()]
        above = [type("M", (), {"home": "came_down", "season": "2025-26"})()]

    teams = ["stay", "came_up", "came_down"]
    out = priors.preseason_net(FakeDs(), FakeFit(), cal, teams, "2025-26")
    # Ranking is what matters: down above stayed above up.
    assert out["came_down"] > out["stay"] > out["came_up"]
    # And the gaps are exactly the intercepts, recentred.
    assert out["came_down"] - out["came_up"] == pytest.approx(1.5)


def test_a_thin_relegated_sample_falls_back_to_the_carryover_not_the_shrink():
    """With too few relegated cases to measure, a club coming down is treated
    like a continuing one. Its rating is a real measurement on a shared scale,
    so shrinking it like a promoted club would be the same error again."""
    thin = [(0.2, 0.1), (-0.3, -0.2)]
    got = priors.regress(thin, "relegated", "championship", fallback="continuing")
    assert got["slope"] == pytest.approx(priors.PL_FALLBACK["continuing"]["slope"])
    assert got["slope"] > priors.PL_FALLBACK["promoted"]["slope"]
    assert got["intercept"] > priors.PL_FALLBACK["promoted"]["intercept"]
