"""The seams, and the bugs that lived in them.

Every test here exists because something shipped. The suite was strong on the
steps -- parsing, name resolution, simulation invariants -- and had nothing at
all on the joins between them: that one build step's output is another's input,
that a constant matches the corpus it claims to describe, that the model the
backtest scores is the model the site publishes. Each of the following would
have caught a bug that was live on 2026-09-04.

Deliberately network-free. Everything is either a pure function, a synthetic
fit or a temporary directory.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import (backtest, config, europe, insight, knockout, leagues,   # noqa: E402
                   priors, ratings, run, simulate)
from model.parse import Match                                              # noqa: E402


def _league(n_teams: int, n_matches: int, **kw) -> leagues.League:
    """A synthetic competition of the right size, so the simulation's European
    and relegation lines fall inside the table it is given."""
    import dataclasses
    return dataclasses.replace(leagues.PREMIER_LEAGUE, slug="test", name="Test",
                               n_teams=n_teams, n_matches=n_matches,
                               ucl_places=kw.get("ucl", 1),
                               europa_places=kw.get("europa", 1),
                               releg_places=kw.get("releg", 1))


def _fit(n: int = 6, seed: int = 0, spread: float = 0.3) -> ratings.Fit:
    rng = np.random.default_rng(seed)
    teams = [f"c{i}" for i in range(n)]
    return ratings.Fit(teams, rng.normal(0, spread, n), rng.normal(0, spread, n),
                       float(np.log(1.35)), 0.25, 0.02)


def _matches(n_teams=8, seasons=("2024-25", "2025-26"), seed=1, comp="dom"):
    rng = np.random.default_rng(seed)
    teams = [f"c{i}" for i in range(n_teams)]
    out = []
    day = dt.date(2024, 8, 1)
    for s in seasons:
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j:
                    continue
                day += dt.timedelta(days=1)
                out.append(Match(date=day, home=teams[i], away=teams[j],
                                 hg=int(rng.poisson(1.5)), ag=int(rng.poisson(1.1)),
                                 played=True, season=s, comp=comp))
    return teams, out


# ------------------------------------------------------------------ S1
def test_the_corpus_carries_the_season_under_forecast():
    """`load_competitions` walks EURO_FILES, so a season missing from it can
    never enter the corpus however many results upstream publishes.

    This one line is the whole of bug S1: with `SEASON` at 2026-27 and the table
    ending at 2025-26, every Champions League rating would have stayed frozen at
    the 2025-26 final from matchday 1 onwards -- and so would `global.json`,
    `ratings.json` and `trajectory.json`, which every club page reads.
    """
    assert config.SEASON in europe.EURO_FILES
    assert "cl" in europe.EURO_FILES[config.SEASON]


def test_a_played_cup_fixture_reaches_the_corpus_once_and_only_once():
    """The other half of S1: our own fixture file is the primary source, and its
    results have to reach the fit before openfootball publishes them -- without
    being counted twice when it finally does."""
    corpus = europe.Corpus()
    played = Match(date=dt.date(2026, 9, 8), home="arsenal", away="inter",
                   hg=2, ag=1, played=True, season="2026-27", comp="cl")
    unplayed = Match(date=dt.date(2026, 9, 9), home="psg", away="porto",
                     hg=None, ag=None, played=False, season="2026-27", comp="cl")
    assert corpus.add_unique([played, unplayed], europe.EUROPE) == 1
    assert len(corpus.before(dt.date(2026, 9, 9))) == 1
    # openfootball publishes the same match a fortnight later: still one copy.
    same = Match(date=dt.date(2026, 9, 8), home="arsenal", away="inter",
                 hg=2, ag=1, played=True, season="2026-27", comp="cl")
    assert corpus.add_unique([same], europe.EUROPE) == 0
    assert len(corpus.matches) == 1


def test_our_own_league_phase_file_reaches_every_corpus(tmp_path, monkeypatch):
    """The corpus, not just the cup page.

    `global.json`, `ratings.json` and `trajectory.json` are all fitted on the
    shared corpus, and upstream published the last three league-phase files +3,
    +68 and +208 days after the draw (twice, never). So the committed file has
    to be read where the corpus is assembled, or every one of those stays frozen
    at the previous season's final until openfootball catches up.
    """
    from model.parse import TeamRegistry
    path = tmp_path / f"fixtures-{config.SEASON}.txt"
    path.write_text(
        f"= UEFA Champions League {config.SEASON.replace('-', '/')}\n\n"
        "\u25aa League, Matchday 1\n"
        "  Tue Sep 8 2026\n"
        "    21:00  Arsenal FC (ENG)   3-1 (1-0)  FC Internazionale Milano (ITA)\n"
        "    21:00  Real Madrid CF (ESP)          v  FC Porto (POR)\n")
    monkeypatch.setattr(europe, "our_fixture_path", lambda season: str(path))
    got = europe.our_played(TeamRegistry(), config.SEASON)
    assert len(got) == 1, "only the played one"
    assert (got[0].home, got[0].hg, got[0].ag) == ("arsenal", 3, 1)
    assert got[0].season == config.SEASON
    # And a season we do not hold a file for is simply nothing, not a crash.
    monkeypatch.setattr(europe, "our_fixture_path", lambda season: str(tmp_path / "no"))
    assert europe.our_played(TeamRegistry(), config.SEASON) == []


def test_a_played_cup_result_moves_the_pooled_fit():
    """And it must actually change the answer, which is the point of all of it.

    A synthetic 4-0 in a competition the corpus can see has to move the winner's
    rating up and the loser's down. Before the fix the match could not enter the
    corpus at all, so this difference was exactly zero.
    """
    teams, ms = _matches()
    cl = {t: "dom" for t in teams}
    ref = dt.date(2026, 9, 10)

    def group_of(m):
        return m.comp

    before = ratings.fit_pooled(ms, teams, ref, group_of=group_of, club_league=cl,
                                default_group="dom")
    corpus = europe.Corpus()
    corpus.add(ms, "dom")
    big = Match(date=dt.date(2026, 9, 8), home=teams[0], away=teams[1],
                hg=4, ag=0, played=True, season="2026-27", comp="cl")
    assert corpus.add_unique([big], europe.EUROPE) == 1
    hist = corpus.before(run.fit_cutoff(dt.date(2026, 9, 9)))
    cl2 = dict(cl)
    after = ratings.fit_pooled(hist, teams, ref, group_of=group_of, club_league=cl2,
                               default_group="dom")
    net = lambda f, t: np.log(f.offence(t)) - np.log(f.defence(t))   # noqa: E731
    assert net(after, teams[0]) > net(before, teams[0])
    assert net(after, teams[1]) < net(before, teams[1])


# ------------------------------------------------------------------ S2
def test_a_cup_table_is_built_from_the_played_fixtures():
    """S2: the six table columns were hardcoded to zero, so from matchday 1
    every club would have shown P0 W0 D0 under a `matches_played` that said
    otherwise."""
    reg = type("R", (), {"display": staticmethod(lambda t: t.title())})()
    fixtures = [
        Match(date=dt.date(2026, 9, 8), home="a", away="b", hg=2, ag=1, played=True),
        Match(date=dt.date(2026, 9, 8), home="c", away="d", hg=0, ag=0, played=True),
        Match(date=dt.date(2026, 9, 9), home="a", away="c", hg=None, ag=None,
              played=False),
    ]
    table = {r["id"]: r for r in run._table_from([f for f in fixtures if f.played], reg)}
    assert table["a"]["pld"] == 1 and table["a"]["pts"] == 3 and table["a"]["gf"] == 2
    assert table["b"]["l"] == 1 and table["b"]["pts"] == 0 and table["b"]["ga"] == 2
    assert table["c"]["d"] == 1 and table["c"]["pts"] == 1
    # Two rows per played match, which is the invariant `validate_cup` now
    # asserts on every build.
    assert sum(r["pld"] for r in table.values()) == 2 * 2


# ------------------------------------------------------------------ S3
def test_shooting_is_written_before_the_ratings_that_read_it():
    """S3: `build_ratings` reads shooting.json for three of its nine ratings and
    `build_shooting` is what writes it. With the steps the other way round the
    published ratings were always one build stale, and absent on a fresh
    checkout."""
    import inspect
    src = inspect.getsource(run.main)
    assert src.index('("shooting", build_shooting)') < src.index('("pooled ratings", build_ratings)')


def test_ratings_json_reflects_the_shooting_file_from_this_run(tmp_path, monkeypatch):
    """The same seam, functionally: a sentinel shooting.json must reach
    ratings.json in the same build."""
    monkeypatch.setattr(run, "OUT", str(tmp_path))
    json.dump({"generated": "2026-09-04T00:00:00+00:00",
               "clubs": [{"id": "arsenal", "spi": 80.0, "att_r": 70, "def_r": 70}]},
              open(tmp_path / "global.json", "w"))
    json.dump({"clubs": {"arsenal": {"sot_pm": 9.9, "conversion": 0.30,
                                     "foul_index": 2.0, "league": "premier-league"},
                         "chelsea": {"sot_pm": 3.0, "conversion": 0.05,
                                     "foul_index": 2.0, "league": "premier-league"}}},
              open(tmp_path / "shooting.json", "w"))
    run.build_ratings({"premier-league"})
    got = json.load(open(tmp_path / "ratings.json"))["clubs"]
    assert "creation_r" in got["arsenal"], "the shot ratings never reached ratings.json"
    assert got["arsenal"]["creation_r"] > got["chelsea"]["creation_r"]


# ------------------------------------------------------------------ S5
def test_no_promoted_club_is_rated_above_its_divisions_median():
    """The invariant that would have caught the Primeira Liga in one line.

    Reads whatever forecasts are on disk; a competition that has not been built
    is skipped rather than failed, so this passes on a fresh checkout and bites
    the moment a build produces a promoted club in the top half.
    """
    checked = 0
    for lg in leagues.LEAGUES:
        path = os.path.join(run.OUT, lg.slug, "forecast.json")
        if not os.path.exists(path):
            continue
        try:
            rows = json.load(open(path))["teams"]
        except (ValueError, OSError):
            continue
        if lg.kind != "league" or not rows:
            continue
        up = [r for r in rows if r.get("arrived") == "up"]
        if not up:
            continue
        median = float(np.median([r["lg_strength"] for r in rows]))
        best = max(r["lg_strength"] for r in up)
        assert best < median, (
            f"{lg.slug}: a promoted club is rated above the division median "
            f"({best:.1f} vs {median:.1f})")
        checked += 1
    if not checked:
        pytest.skip("no forecasts on disk to check")


# ------------------------------------------------------------------ S6
def test_a_dead_level_tie_is_broken_at_random_not_alphabetically():
    """S6: the tie-break randomiser was packed into a float64 key at 1e14, three
    orders of magnitude below the ULP, so it was discarded and every genuine tie
    went to the alphabetically-first club.

    Two identical clubs, a finished season, level on everything: each must win
    the title about half the time.
    """
    teams = ["aaa", "zzz"]
    fit = ratings.Fit(teams, np.zeros(2), np.zeros(2), float(np.log(1.35)), 0.0, 0.0)
    fixtures = [Match(date=dt.date(2026, 5, 1), home="aaa", away="zzz",
                      hg=1, ag=1, played=True),
                Match(date=dt.date(2026, 5, 8), home="zzz", away="aaa",
                      hg=2, ag=2, played=True)]
    sim = simulate.simulate_season(fit, fixtures, teams, n_sims=4000,
                                   league=_league(2, 2), rating_sd=0.0,
                                   curves=False)
    assert 0.4 < float(sim["title"][0]) < 0.6, sim["title"]


# ------------------------------------------------------------------ S7
def test_recentring_the_fit_leaves_the_goal_total_at_the_mle():
    """S7: `_fit_core` shifted attack and defence onto their own means without
    moving the intercept, which multiplied every lambda by exp(d_bar - a_bar) --
    0.43% on the live Premier League fit, on every published xG, score grid and
    season goal total.

    The intercept is unpenalised, so at the optimum the weighted predicted goals
    must equal the weighted observed goals exactly. That identity is what the
    recentring used to break.
    """
    teams, ms = _matches(n_teams=10, seasons=("2024-25",), seed=7)
    ref = dt.date(2026, 1, 1)
    fit = ratings.fit(ms, teams, ref, shot_conv=None, goals_weight=1.0)
    pred = obs = 0.0
    for m in ms:
        w = float(np.exp(-config.TIME_DECAY * (ref - m.date).days))
        lh, la = fit.lambdas(m.home, m.away)
        pred += w * (lh + la)
        obs += w * (m.hg + m.ag)
    assert abs(pred / obs - 1.0) < 2e-3, f"lambdas are {pred / obs - 1:+.4%} off the MLE"


# ------------------------------------------------------------------ S8
def test_a_fixture_whose_kickoff_has_passed_is_never_refrozen(tmp_path, monkeypatch):
    """S8: the only gate was `played`, which is what the *feed* says. A match
    kicked off on Saturday that openfootball has not committed by Wednesday was
    re-frozen on Wednesday with four days of other results in the fit, and the
    site's own honest in-season log-loss was then measured against probabilities
    that were never pre-match."""
    monkeypatch.setattr(insight, "OUT", str(tmp_path))
    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    tomorrow = (dt.date.today() + dt.timedelta(days=1)).isoformat()
    base = {"md": 1, "xgh": 1.4, "xga": 1.1, "played": False}
    insight.freeze_predictions([
        {**base, "h": "a", "a": "b", "date": yesterday,
         "ph": 0.50, "pd": 0.30, "pa": 0.20},
        {**base, "h": "c", "a": "d", "date": tomorrow,
         "ph": 0.50, "pd": 0.30, "pa": 0.20}])
    # A later build, with the feed still behind on the first match.
    store = insight.freeze_predictions([
        {**base, "h": "a", "a": "b", "date": yesterday,
         "ph": 0.90, "pd": 0.05, "pa": 0.05},
        {**base, "h": "c", "a": "d", "date": tomorrow,
         "ph": 0.90, "pd": 0.05, "pa": 0.05}])
    assert store["a|b"]["ph"] == 0.50, "a match that has kicked off was re-frozen"
    assert store["c|d"]["ph"] == 0.90, "a match still to come must keep updating"


# ------------------------------------------------------------------ S9
def test_the_optional_writers_are_imported_at_module_level():
    """S9: two `except ImportError: pass` blocks swallowed an ImportError raised
    *inside* siminput or recap with no log line, leaving the previous run's
    sim_input.json on disk for the what-if worker to read as current."""
    assert hasattr(run, "siminput") and hasattr(run, "recap")
    import inspect
    src = inspect.getsource(run)
    assert "except ImportError:\n        pass" not in src


# ------------------------------------------------------------------ S10/S11
def test_a_ready_league_does_not_ship_a_ready_note():
    """S11: `ready_note` is the sentence shown *instead of* a forecast."""
    cup = leagues.get("champions-league")
    assert cup.ready_note, "this test needs a league that has a note"
    assert "ready_note" not in cup.manifest_entry(ready=True)
    assert cup.manifest_entry(ready=False)["ready_note"] == cup.ready_note


def test_the_manifest_count_covers_every_competition():
    """S10: 'ready' is built over LEAGUES + EUROPEAN and the denominator was
    LEAGUES alone, which printed '10/9 leagues ready'."""
    import inspect
    src = inspect.getsource(run.main)
    assert "len(leagues.LEAGUES + leagues.EUROPEAN)" in src


# ------------------------------------------------------------------ S12
@pytest.mark.parametrize("lg", list(leagues.LEAGUES) + list(leagues.EUROPEAN),
                         ids=lambda l: l.slug)
def test_the_market_anchor_expires_by_the_end_of_every_competition(lg):
    """S12: the anchor decayed over ten 'matchweeks' derived from a domestic
    season's length. The Champions League league phase is eight matchdays, so it
    carried 0.65 x 0.2 = 0.13 all the way through the knockout."""
    assert priors.market_weight(0, lg) == pytest.approx(config.MARKET_WEIGHT)
    assert priors.market_weight(lg.n_matches, lg) == 0.0
    # And it decays, rather than falling off a cliff at the end.
    assert 0 < priors.market_weight(lg.n_matches // 8, lg) < config.MARKET_WEIGHT


# ------------------------------------------------------------------ S13
def test_regressing_a_relegated_group_without_a_fallback_does_not_raise():
    """S13: `PL_FALLBACK[fallback or key]` with no 'relegated' key."""
    got = priors.regress([], "relegated", "championship")
    assert got["slope"] == priors.PL_FALLBACK["continuing"]["slope"]


# ------------------------------------------------------------------ S14
def test_a_match_played_today_is_visible_to_the_fit():
    """S14: `before()` is strictly-before and the reference date was today, so a
    Saturday result was in the table, in the points and in the frozen scoring
    while being invisible to every rating until Sunday."""
    today = dt.date.today()
    played_today = Match(date=today, home="a", away="b", hg=1, ag=0, played=True)
    tomorrow = Match(date=today + dt.timedelta(days=1), home="c", away="d",
                     hg=None, ag=None, played=False)
    corpus = europe.Corpus()
    corpus.add([played_today], "dom")
    corpus.add_unique([tomorrow], "dom")
    assert corpus.before(today) == []
    assert len(corpus.before(run.fit_cutoff(today))) == 1


# ------------------------------------------------------------- calibration
def test_sharpening_is_the_identity_at_one_and_monotone_around_it():
    p = np.array([0.55, 0.25, 0.20])
    assert simulate.sharpen_probs(p, 1.0) is p
    hot = simulate.sharpen_probs(p, 1.2)
    cold = simulate.sharpen_probs(p, 0.8)
    assert hot[0] > p[0] > cold[0]
    assert hot[2] < p[2] < cold[2]
    for q in (hot, cold):
        assert abs(q.sum() - 1) < 1e-12


def test_sharpening_the_grid_moves_the_result_and_not_the_scoreline():
    """The exponent is measured on results, so it is applied to results: the
    distribution of scorelines *within* a home win must be untouched."""
    m = simulate.score_matrix(1.62, 1.05, 0.03)
    g = simulate.sharpen_grid(m, 1.15)
    assert abs(g.sum() - 1) < 1e-12
    p = np.array(simulate.outcome_probs(m))
    q = np.array(simulate.outcome_probs(g))
    assert np.allclose(q, simulate.sharpen_probs(p, 1.15), atol=1e-12)
    # 2-1 and 3-1 are both home wins, so their ratio cannot have moved.
    assert (g[2, 1] / g[3, 1]) == pytest.approx(m[2, 1] / m[3, 1], rel=1e-12)


def test_the_exponent_is_recovered_from_data_that_has_one():
    rng = np.random.default_rng(3)
    P = rng.dirichlet([4, 3, 3], 4000)
    truth = simulate.sharpen_probs(P, 1.2)
    Y = np.array([rng.choice(3, p=r) for r in truth])
    got = backtest.fit_sharpen(P, Y, "synthetic")
    assert 1.05 < got["k"] < 1.4
    assert got["applied"] is True
    assert got["held_out"]["gain"] > 0
    assert got["held_out"]["n"] + got["held_out"]["train_n"] == 4000


def test_an_exponent_that_does_not_generalise_is_not_shipped():
    """Fitting always improves the matches you fitted on. Only the held-out
    half decides whether the correction ships: the Primeira Liga measures
    k = 1.10 in sample and loses 0.0033 out of it."""
    rng = np.random.default_rng(9)
    P = rng.dirichlet([4, 3, 3], 2000)
    # Outcomes drawn from the probabilities themselves: perfectly calibrated,
    # so any exponent the first 60% fits is noise and must not survive.
    Y = np.array([rng.choice(3, p=r) for r in P])
    got = backtest.fit_sharpen(P, Y, "synthetic")
    if got["held_out"]["gain"] <= 0:
        assert got["k"] == 1.0 and got["applied"] is False
        assert got["measured_k"] != 1.0
        assert "uncalibrated" in got["reason"]
    else:                       # the fit happened to generalise on this draw
        assert got["k"] == got["measured_k"]


def test_a_thin_competition_is_left_uncalibrated():
    """'Never touch cups unless there is data': the two Swiss Champions League
    seasons are 378 matches and one fitted exponent over them is those two
    seasons' luck."""
    rng = np.random.default_rng(4)
    P = rng.dirichlet([4, 3, 3], 378)
    Y = rng.integers(0, 3, 378)
    got = backtest.fit_sharpen(P, Y, "champions-league")
    assert got["k"] == 1.0
    assert "below" in got["reason"]


def test_a_missing_or_unreadable_exponent_is_one(tmp_path, monkeypatch):
    monkeypatch.setattr(backtest, "CACHE", str(tmp_path))
    assert backtest.load_sharpen("nowhere") == 1.0
    backtest.save_sharpen("somewhere", {"k": 1.19})
    assert backtest.load_sharpen("somewhere") == pytest.approx(1.19)
    backtest.save_sharpen("silly", {"k": 9.0})
    assert backtest.load_sharpen("silly") == backtest.SHARPEN_BAND[1]


def test_the_simulation_and_the_match_page_sharpen_alike():
    """One exponent, two code paths: the table the season simulation produces
    has to be built from the numbers the match pages print."""
    fit = _fit(4, seed=5)
    teams = fit.teams
    fixtures = [Match(date=dt.date(2026, 5, 1) + dt.timedelta(days=i),
                      home=teams[i % 4], away=teams[(i + 1) % 4],
                      hg=None, ag=None, played=False) for i in range(12)]
    lg = _league(4, 12)
    plain = simulate.simulate_season(fit, fixtures, teams, n_sims=20000,
                                     league=lg, rating_sd=0.0, curves=False)
    sharp = simulate.simulate_season(fit, fixtures, teams, n_sims=20000,
                                     league=lg, rating_sd=0.0, sharpen=1.3,
                                     curves=False)
    best = int(np.argmax(plain["title"]))
    assert sharp["title"][best] > plain["title"][best] + 0.005


# --------------------------------------------------------------- backtest
def test_the_backtest_predicts_through_the_same_call_the_site_does():
    """5.5: whatever the walk-forward scores has to be `match_report`'s answer
    for the same fit and the same adjustment, or the published accuracy belongs
    to a model the site does not run."""
    fit = _fit(4, seed=11)
    adj = {"c0": 0.3, "c1": -0.2, "c2": 0.0, "c3": -0.1}
    lh, la = simulate._lambdas(fit, "c0", "c1", adj)
    from model.backtest import outcome_probs as bt_outcome, score_matrix as bt_grid
    ph, pd, pa = bt_outcome(bt_grid(lh, la, fit.rho))
    rep = simulate.match_report(fit, "c0", "c1", adj)
    assert (ph, pd, pa) == pytest.approx((rep["home_win"], rep["draw"],
                                          rep["away_win"]), abs=1e-12)


# --------------------------------------------------------------- knockout
def test_the_bracket_resamples_ratings_like_the_league_phase():
    """1.7: the bracket used one tie matrix from the point estimate, so four
    knockout rounds compounded the favourite's edge with zero rating
    uncertainty while the league phase beside it resampled every scenario. The
    published number was Arsenal 30.2% for the trophy."""
    n = 36
    rng = np.random.default_rng(2)
    teams = [f"c{i}" for i in range(n)]
    att = np.linspace(0.45, -0.45, n)
    fit = ratings.Fit(teams, att, att[::-1].copy(), float(np.log(1.35)), 0.25, 0.02)
    orders = np.stack([np.arange(n)] * 2000)                # the seeding is fixed
    scen = np.repeat(np.arange(20), 100)
    shocks = rng.normal(0.0, config.RATING_SD, (20, n))
    fixed = knockout.simulate_bracket(fit, teams, orders)
    drawn = knockout.simulate_bracket(fit, teams, orders, shocks=shocks,
                                      scenario=scen)
    assert fixed["resampled"] is False and drawn["resampled"] is True
    assert drawn["win"][0] < fixed["win"][0], "resampling must flatten the favourite"
    for key in ("r16", "qf", "sf", "final", "win"):
        assert drawn[key].sum() == pytest.approx(fixed[key].sum(), abs=1e-9)


def test_a_zero_shock_reproduces_the_point_estimate_matrices():
    fit = _fit(8, seed=13)
    tie, neu = knockout.shock_matrices(fit, fit.teams, np.zeros((1, 8)))
    assert np.abs(tie[0] - knockout.tie_matrix(fit, fit.teams)).max() < 1e-12
    assert np.abs(neu[0] - knockout.neutral_matrix(fit, fit.teams)).max() < 1e-12


# ------------------------------------------------------- contract §4 and §5
def test_clinch_is_a_conservative_bound_that_says_done_when_it_is_done():
    rows = [{"id": "a", "cur_pts": 30}, {"id": "b", "cur_pts": 10},
            {"id": "c", "cur_pts": 5}]
    # One match left each, and a plays nobody in this list.
    fixtures = [Match(date=dt.date(2026, 5, 1), home="a", away="b",
                      hg=None, ag=None, played=False),
                Match(date=dt.date(2026, 5, 1), home="c", away="c",
                      hg=None, ag=None, played=False)]
    run._clinch(rows, fixtures, {"title": 1, "top": 2, "safe": 2})
    by = {r["id"]: r["clinch"] for r in rows}
    # b's ceiling is 13, c's is 11, so a on 30 has already won it.
    assert by["a"]["title"] == {"done": True, "need": 0}
    # b needs to beat c's ceiling of 11: 12 - 10 = 2 points, and it has 3 left.
    assert by["b"]["title"]["done"] is False
    assert by["b"]["title"]["need"] is None      # a's ceiling is 33, unreachable
    # For a top-two finish b only has to clear the second-best rival ceiling,
    # which is c's 11: two more points does it, and b has three left.
    assert by["b"]["top"] == {"done": False, "need": 2}
    assert by["c"]["title"]["need"] is None


def test_rooting_covers_the_next_round_only_and_omits_the_clubs_playing(tmp_path):
    fit = _fit(6, seed=17, spread=0.45)
    teams = fit.teams
    fixtures = []
    day = dt.date(2026, 5, 1)
    for md in (1, 2):
        for i in range(3):
            fixtures.append(Match(date=day + dt.timedelta(days=7 * md),
                                  home=teams[i], away=teams[5 - i],
                                  hg=None, ag=None, played=False, matchday=md))
    sim = simulate.simulate_season(fit, fixtures, teams, n_sims=20000,
                                   league=_league(6, len(fixtures)),
                                   rating_sd=0.0, leverage=True, curves=False)
    got = run.write_rooting(str(tmp_path), sim, teams)
    doc = json.load(open(tmp_path / "rooting.json"))
    assert doc == got
    assert doc["md"] == 1
    assert doc["events"] == list(simulate.EVENTS), "a league's own event keys"
    assert doc["contract_events"] == list(run.CONTRACT_EVENTS)
    assert len(doc["matches"]) == 3, "next matchweek only"
    for row in doc["matches"]:
        assert set(row) == {"h", "a", "date", "effects"}
        # A cup would carry ("top8", "qualify", "out") here instead: the first
        # event of a cup's league-phase simulation is finishing in the top
        # eight, and calling it "title" would collide with the trophy
        # probability the same competition's forecast.json calls "title".
        assert row["h"] not in row["effects"] and row["a"] not in row["effects"]
        for club, ev in row["effects"].items():
            assert club in teams
            for name, deltas in ev.items():
                assert name in simulate.EVENTS
                assert len(deltas) == 3
                assert max(abs(v) for v in deltas) >= run.ROOTING_FLOOR
    # A result that helps somebody has to hurt somebody: over a whole match the
    # title deltas of the clubs not playing cannot all point the same way.
    hits = [sum(ev["title"][0] for ev in row["effects"].values() if "title" in ev)
            for row in doc["matches"]]
    assert any(h < 0.02 for h in hits)
