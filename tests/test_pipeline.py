"""Tests for the parts that fail silently if they break.

Club-name mapping and fixture integrity get the most attention here: a source
renaming a club, or dropping a match, degrades the forecast without raising
anything, which is the worst kind of bug for a site that publishes numbers.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import config, ratings, simulate                      # noqa: E402
from model.data import Dataset                                    # noqa: E402
from model.parse import TeamRegistry, normalise, parse_openfootball  # noqa: E402
from model.priors import devig                                    # noqa: E402


@pytest.fixture(scope="module")
def ds() -> Dataset:
    return Dataset().load()


# ---------------------------------------------------------------- naming
@pytest.mark.parametrize("a,b", [
    ("Man City", "Manchester City FC"),
    ("Nott'm Forest", "Nottingham Forest FC"),
    ("Brighton", "Brighton & Hove Albion FC"),
    ("Bournemouth", "AFC Bournemouth"),
    ("Hull", "Hull City AFC"),
    ("Wolves", "Wolverhampton Wanderers"),
    ("Spurs", "Tottenham Hotspur FC"),
])
def test_name_variants_resolve_together(a, b):
    reg = TeamRegistry()
    assert reg.resolve(a) == reg.resolve(b), f"{a!r} and {b!r} must be the same club"


def test_distinct_clubs_stay_distinct():
    reg = TeamRegistry()
    assert reg.resolve("Man City") != reg.resolve("Man United")
    assert reg.resolve("Sheffield United") != reg.resolve("Sheffield Wednesday")


def test_normalise_drops_only_club_suffixes():
    assert normalise("Arsenal FC") == "arsenal"
    assert normalise("AFC Bournemouth") == "bournemouth"
    assert normalise("Brighton & Hove Albion") == "brighton and hove albion"


# ---------------------------------------------------------------- parsing
def test_openfootball_handles_all_three_layouts():
    """Fixture-only, fixture-with-score, and score-in-the-middle all appear in
    the files this pipeline reads, sometimes within one season."""
    reg = TeamRegistry()
    text = "\n".join([
        "= Test League 2026/27",
        "",
        "▪ Matchday 1",
        "  Fri Aug 21 2026",
        "    20:00  Arsenal FC              v Coventry City FC",
        "  Sat Aug 22",
        "    15:00  Hull City AFC           v Chelsea FC               2-1 (1-0)",
        "    15:00  Liverpool  3-0 (2-0)  Everton",
        "                  (Someone SCORED 12')",
        "    17:30  Fulham FC               v Leeds United FC          1-0 a.e.t. (0-0)",
    ])
    got = parse_openfootball(text, "2026-27", reg)
    assert len(got) == 4
    assert [m.played for m in got] == [False, True, True, True]
    assert (got[1].hg, got[1].ag) == (2, 1)
    assert (got[2].home, got[2].hg, got[2].ag) == (reg.resolve("Liverpool"), 3, 0)
    assert (got[3].hg, got[3].ag) == (1, 0), "a.e.t. must not swallow the score"
    assert all(m.matchday == 1 for m in got)


def test_openfootball_carries_the_year_across_new_year(ds):
    """Month headers omit the year; January must land in the following one."""
    reg = TeamRegistry()
    text = "\n".join([
        "▪ Matchday 20",
        "  Sat Dec 26 2026",
        "    15:00  Arsenal FC v Chelsea FC  1-1",
        "  Sat Jan 2",
        "    15:00  Chelsea FC v Arsenal FC  0-0",
    ])
    got = parse_openfootball(text, "2026-27", reg)
    assert got[0].date == dt.date(2026, 12, 26)
    assert got[1].date == dt.date(2027, 1, 2)


# ---------------------------------------------------------------- fixtures
def test_fixture_list_is_a_complete_season(ds):
    assert len(ds.fixtures) == config.N_MATCHES
    teams = ds.teams
    assert len(teams) == config.N_TEAMS
    for t in teams:
        home = sum(1 for m in ds.fixtures if m.home == t)
        away = sum(1 for m in ds.fixtures if m.away == t)
        assert home == away == config.N_TEAMS - 1, f"{t} plays {home}H/{away}A"


def test_every_club_in_the_fixture_list_is_mapped(ds):
    unmapped = [t for t in ds.teams if ds.reg.meta.get(t, {}).get("auto")]
    assert not unmapped, f"unmapped club names: {unmapped}"


def test_each_pairing_happens_exactly_once(ds):
    pairs = [(m.home, m.away) for m in ds.fixtures]
    assert len(set(pairs)) == len(pairs)


def test_history_loaded(ds):
    assert len(ds.pl) > 9000, "expected 25+ seasons of Premier League results"
    assert len(ds.ch) > 5000, "Championship history is needed to rate promoted clubs"


# ---------------------------------------------------------------- model
def test_score_matrix_is_a_distribution():
    m = simulate.score_matrix(1.7, 1.1, -0.08)
    assert abs(m.sum() - 1) < 1e-9
    assert (m >= 0).all()
    h, d, a = simulate.outcome_probs(m)
    assert abs(h + d + a - 1) < 1e-9


def test_dixon_coles_correction_lifts_low_score_draws():
    """The correction exists to stop the model under-predicting 0-0 and 1-1."""
    plain = simulate.score_matrix(1.4, 1.2, 0.0)
    corrected = simulate.score_matrix(1.4, 1.2, -0.10)
    assert corrected[0, 0] > plain[0, 0]
    assert corrected[1, 1] > plain[1, 1]


def test_stronger_team_is_favoured():
    teams = ["strong", "weak"]
    fit = ratings.Fit(teams, np.array([0.5, -0.5]), np.array([0.4, -0.4]),
                      np.log(1.35), 0.2, -0.05)
    h, d, a = simulate.outcome_probs(
        simulate.score_matrix(*fit.lambdas("strong", "weak"), fit.rho))
    assert h > 0.6 > a


def test_home_advantage_is_worth_something():
    teams = ["a", "b"]
    fit = ratings.Fit(teams, np.zeros(2), np.zeros(2), np.log(1.35), 0.25, -0.05)
    h, _, a = simulate.outcome_probs(
        simulate.score_matrix(*fit.lambdas("a", "b"), fit.rho))
    assert h > a


def test_devig_removes_the_margin():
    p = devig({"a": 2.0, "b": 2.0})
    assert abs(sum(p.values()) - 1) < 1e-9
    assert abs(p["a"] - 0.5) < 1e-9


def test_simulation_probabilities_are_coherent(ds):
    """The invariants that catch a broken tabulation: exactly one champion,
    UCL_PLACES qualifiers and RELEGATION_PLACES relegations per season."""
    fit = ratings.fit(
        [m for m in ds.pl if m.season >= "2024-25"],
        ds.teams, dt.date(2026, 8, 21))
    sim = simulate.simulate_season(fit, ds.fixtures, ds.teams,
                                   n_sims=4000, scenarios=20)
    assert abs(sim["title"].sum() - 1) < 0.02
    assert abs(sim["ucl"].sum() - config.UCL_PLACES) < 0.05
    assert abs(sim["relegation"].sum() - config.RELEGATION_PLACES) < 0.05
    assert np.allclose(sim["position"].sum(axis=1), 1, atol=1e-6)
    assert np.allclose(sim["position"].sum(axis=0), 1, atol=1e-6)
