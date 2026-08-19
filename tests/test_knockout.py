"""Tests for the knockout machinery and the cup pipeline.

Two-legged ties are computed exactly, so they can be checked exactly: a tie
between identical clubs is 50/50, home advantage in the second leg is worth
something and not much, and the probabilities of reaching each round have to sum
to the number of places in that round -- 16 in the round of 16, one champion.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import knockout, leagues, ratings, simulate                 # noqa: E402
from model.parse import Match                                          # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fit(n: int = 36, spread: float = 0.35, home: float = 0.28) -> ratings.Fit:
    """A synthetic fit with clubs evenly spaced from strong to weak."""
    teams = [f"c{i:02d}" for i in range(n)]
    att = np.linspace(spread, -spread, n)
    dfn = np.linspace(spread, -spread, n)
    return ratings.Fit(teams, att, dfn, np.log(1.35), home, 0.0,
                       homes={"europe": home}, default_group="europe")


def test_a_tie_between_identical_clubs_on_neutral_ground_is_a_coin_flip():
    f = ratings.Fit(["a", "b"], np.zeros(2), np.zeros(2), np.log(1.35), 0.0, 0.0)
    assert knockout.two_legged(f, "a", "b") == pytest.approx(0.5, abs=1e-12)


def test_second_leg_at_home_is_worth_something_but_not_much():
    """Over ninety minutes a two-legged tie between equals is exactly level:
    each club hosts once. The only edge the higher seed gets is that extra time,
    when it happens, is played at its ground -- worth about a point of
    probability, which is what seeding a bracket is supposed to be worth."""
    f = ratings.Fit(["a", "b"], np.zeros(2), np.zeros(2), np.log(1.35), 0.28, 0.0)
    p = knockout.two_legged(f, "a", "b")
    assert 0.50 < p < 0.52, p
    # Give a a real edge and the tie must follow it.
    f2 = ratings.Fit(["a", "b"], np.array([0.4, -0.4]), np.array([0.4, -0.4]),
                     np.log(1.35), 0.28, 0.0)
    assert knockout.two_legged(f2, "a", "b") > 0.75


def test_extra_time_is_shorter_and_quieter_than_ninety_minutes():
    w, d, l = knockout._extra_time(1.5, 1.2, 0.0)
    assert abs(w + d + l - 1) < 1e-9
    from model.simulate import outcome_probs, score_matrix
    _, d90, _ = outcome_probs(score_matrix(1.5, 1.2, 0.0))
    # Thirty minutes at 85% tempo: many more goalless halves than a full match.
    assert d > d90 + 0.25
    assert knockout.ET_FRACTION * knockout.ET_TEMPO < 0.35


def test_no_away_goals():
    """A 2-0 / 0-2 tie must be settled by extra time, not by where goals fell.
    With two identical clubs the answer is a half, and any away-goals rule would
    break that symmetry."""
    f = ratings.Fit(["a", "b"], np.zeros(2), np.zeros(2), np.log(1.35), 0.0, 0.0)
    assert knockout.two_legged(f, "a", "b") == pytest.approx(0.5, abs=1e-12)


def test_one_off_match_always_produces_a_winner():
    f = ratings.Fit(["a", "b"], np.array([0.2, -0.2]), np.array([0.1, -0.1]),
                    np.log(1.35), 0.28, 0.0)
    p = knockout.one_off(f, "a", "b", neutral=True)
    q = knockout.one_off(f, "b", "a", neutral=True)
    assert p + q == pytest.approx(1.0, abs=1e-9)


def test_bracket_probabilities_sum_to_the_places_in_each_round():
    fit = _fit()
    teams = fit.teams
    rng = np.random.default_rng(7)
    orders = np.stack([rng.permutation(36) for _ in range(2000)]).astype(np.int16)
    br = knockout.simulate_bracket(fit, teams, orders, group="europe", max_sims=2000)
    for key, want in (("r16", 16), ("qf", 8), ("sf", 4), ("final", 2), ("win", 1)):
        assert abs(br[key].sum() - want) < 1e-9, (key, br[key].sum())
    # Monotone: you cannot win the thing without reaching the final.
    for i in range(36):
        assert br["win"][i] <= br["final"][i] <= br["sf"][i] <= br["qf"][i] \
            <= br["r16"][i] + 1e-12


def test_only_the_top_twenty_four_reach_the_round_of_sixteen():
    """Positions 25-36 are eliminated by the league phase and cannot appear."""
    fit = _fit()
    orders = np.tile(np.arange(36, dtype=np.int16), (500, 1))
    br = knockout.simulate_bracket(fit, fit.teams, orders, group="europe",
                                   max_sims=500)
    assert br["r16"][24:].sum() == 0
    # The eight direct qualifiers always play the round of 16.
    assert np.allclose(br["r16"][:8], 1.0)


def test_playoff_seeding_is_strict():
    assert knockout.PLAYOFF_TIES[0] == (9, 24)
    assert knockout.PLAYOFF_TIES[-1] == (16, 17)
    assert len(knockout.PLAYOFF_TIES) == 8
    assert sorted(sum(knockout.PLAYOFF_TIES, ())) == list(range(9, 25))


def test_the_two_top_seeds_can_only_meet_in_the_final():
    """The bracket has to keep first and second apart, or finishing first buys
    nothing."""
    fit = _fit()
    orders = np.tile(np.arange(36, dtype=np.int16), (400, 1))
    # Seed 1 sits in slot 0, seed 2 in slot 1; QF and SF pairings must never
    # bring slot 0 and slot 1 together.
    halves = {}
    for qi, (a, b) in enumerate(knockout.QF_PAIRS):
        halves[a] = halves[b] = qi // 2
    assert halves[0] != halves[1]


def test_cup_leverage_events_use_the_advancement_lines():
    """A cup's three leverage events are top-8 / top-24 / eliminated, not
    title / European place / relegation."""
    lg = leagues.CHAMPIONS_LEAGUE
    assert simulate.CUP_EVENTS == ("top8", "qualify", "out")
    assert lg.advance_direct == 8 and lg.advance_playoff == 16
    assert lg.n_teams - lg.advance_direct - lg.advance_playoff == 12


def test_simulate_season_can_return_finishing_orders():
    fit = _fit(4)
    teams = fit.teams
    tiny = leagues.League(slug="tiny", name="Tiny", country="Nowhere",
                          n_teams=4, n_matches=12, ucl_places=1,
                          releg_places=1, releg_note=None)
    d = dt.date(2026, 9, 8)
    fixtures = [Match(date=d, home=a, away=b)
                for a in teams for b in teams if a != b]
    sim = simulate.simulate_season(fit, fixtures, teams, league=tiny,
                                   n_sims=200, scenarios=10, keep_orders=True)
    orders = sim["orders"]
    assert orders.shape == (sim["n_sims"], 4)
    for row in orders[:20]:
        assert sorted(row) == [0, 1, 2, 3]


def test_staleness_widens_the_interval_and_is_capped():
    ref = dt.date(2026, 8, 19)
    seen = {"fresh": dt.date(2026, 8, 1),
            "year": dt.date(2025, 8, 19),
            "ancient": dt.date(2020, 1, 1)}
    sd = ratings.staleness_sd(["fresh", "year", "ancient", "never"], seen, ref,
                              base=0.1)
    assert sd[0] == pytest.approx(0.1, abs=0.005)
    assert sd[1] == pytest.approx(0.2, abs=0.005)
    assert sd[2] == pytest.approx(0.2)          # capped at 2x
    assert sd[3] == pytest.approx(0.2)          # never seen at all
