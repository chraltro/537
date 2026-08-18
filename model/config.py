"""Central configuration. Every tunable the model has lives here, on purpose:
a forecast whose dials are scattered through the code is a forecast nobody can audit."""
from __future__ import annotations

# ---- Season under forecast -------------------------------------------------
SEASON = "2026-27"
SEASON_LABEL = "2026/27"
N_TEAMS = 20
N_MATCHES = 380

# European qualification. England earned a fifth Champions League place for
# 2026/27 via the UEFA European Performance Spot, so the UCL line is 5th.
UCL_PLACES = 5
EUROPA_PLACES = 2          # 6th-7th, approximate: domestic cup winners move this line
RELEGATION_PLACES = 3

# ---- Data sources ----------------------------------------------------------
# Both are reachable as raw GitHub content, which keeps the pipeline identical
# locally and on an Actions runner, with no API keys anywhere.
FD_BASE = "https://raw.githubusercontent.com/datasets/football-datasets/main/datasets/premier-league"
OF_BASE = "https://raw.githubusercontent.com/openfootball/england/master"

# Premier League seasons pulled for model fitting (football-datasets CSV codes).
# 2000-01 is where shots-on-target coverage becomes reliable.
PL_SEASONS = [f"{y%100:02d}{(y+1)%100:02d}" for y in range(2000, 2027)]
# Championship seasons, used only to rate promoted clubs.
CH_SEASONS = [f"{y}-{(y+1)%100:02d}" for y in range(2010, 2027)]

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
