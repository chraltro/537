"""Central configuration. Every tunable the model has lives here, on purpose:
a forecast whose dials are scattered through the code is a forecast nobody can audit.

What is *not* here any more: anything that differs between leagues. Team counts,
season lengths, European places, relegation lines and source paths all live on a
`League` object in `model.leagues`, and every computation path takes one. The
constants below are kept as Premier League aliases so a caller with no league in
hand still gets the league this project started with -- `tests/test_leagues.py`
pins them to `leagues.PREMIER_LEAGUE` so the two cannot drift apart.
"""
from __future__ import annotations

from . import leagues

# ---- Season under forecast -------------------------------------------------
SEASON = "2026-27"
SEASON_LABEL = "2026/27"

# ---- Premier League aliases (see module docstring) -------------------------
DEFAULT_LEAGUE = leagues.PREMIER_LEAGUE
N_TEAMS = DEFAULT_LEAGUE.n_teams
N_MATCHES = DEFAULT_LEAGUE.n_matches
UCL_PLACES = DEFAULT_LEAGUE.ucl_places
EUROPA_PLACES = DEFAULT_LEAGUE.europa_places
RELEGATION_PLACES = DEFAULT_LEAGUE.releg_places

# ---- Data sources ----------------------------------------------------------
FD_BASE = leagues.FD_BASE
OF_BASE = leagues.OF_BASE

# Premier League seasons pulled for model fitting (football-datasets CSV codes)
# and the Championship seasons used only to rate promoted clubs.
PL_SEASONS = DEFAULT_LEAGUE.fd_season_codes(SEASON)
CH_SEASONS = DEFAULT_LEAGUE.second_season_labels(SEASON)

# ---- Ratings ---------------------------------------------------------------
# Weight on actual goals vs the shot-derived expectation when measuring how well
# a team played. Shots are the more repeatable signal in small samples, which is
# the core reason a ratings model beats the league table.
#
# Both of these were chosen by grid search on the 2022-23..2024-25 walk-forward
# backtest, not by taste. The honest read of that search: the decay rate matters
# considerably more than the blend. With only on-target/off-target as a stand-in
# for shot quality, the blend is worth roughly 0.005 of log-loss, not the much
# larger gain a real expected-goals model would give.
GOALS_WEIGHT = 0.70
# Exponential time decay, per day: a half-life of about eight months.
TIME_DECAY = 0.0028
# Ridge pull of attack/defence parameters toward zero, stabilises sparse fits.
RIDGE = 0.02
MAX_GOALS = 10             # score matrix truncation; P(>10) is ~1e-6

# ---- Preseason priors ------------------------------------------------------
# Fraction of a team's rating carried into the next season (rest reverts to the
# league mean). Estimated from season-to-season rating autocorrelation.
CARRYOVER = 0.72
# Extra shrink applied to promoted clubs on top of the league-conversion offset.
PROMOTED_SHRINK = 0.60
# Market anchor weight at matchweek 0, decaying linearly to zero by MARKET_DECAY_MW.
MARKET_WEIGHT = 0.65
MARKET_DECAY_MW = 10

# ---- Simulation ------------------------------------------------------------
N_SIMS = 50_000
RATING_SD = 0.085          # posterior sd per team rating, from parametric bootstrap
SEED = 537
