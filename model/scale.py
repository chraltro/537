"""Attack and defence as a rating out of 100, instead of as goals per game.

The model thinks in goals: an attack is "1.91 expected against an average
opponent", a defence is "0.55 conceded". Those are the honest units and they are
what every probability on this site is computed from, but they are hard to read
at a glance and, for defence, backwards -- the smaller number is the better
club, which is the opposite of every other column on the page.

So each is also published as a rating out of 100, on the convention every
football fan already has: higher is better, both times, and the numbers live
roughly between 40 and 90.

The transform, in one line:

    rating = 35 + 60 / (1 + exp(-z / 1.3))        z = (log x - log ref) / sd

Three things about it are deliberate.

*Logs.* Goal rates are multiplicative -- a club that scores twice the average is
as far above it as one scoring half is below -- so the distance that matters is
a ratio, and a ratio is a difference in logs.

*A logistic, not a straight line.* A linear map of a z-score has to choose
between compressing the middle, where nearly every club is, and sending the
outliers off the end of the scale. The logistic keeps the middle open and
flattens the tails, so the best attack in Europe lands near 90 rather than at
140, and nothing ever needs clamping.

*Fixed spreads.* `sd` is a constant, measured once (see below) and written down,
not recomputed each build. If it were recomputed, one club's summer would move
every other club's rating, and a season where the title race tightened would
silently inflate everybody. The cost is that a competition whose clubs are
unusually close together shows a narrower band of ratings than one where they
are spread out -- which is not a flaw, it is the most interesting thing on the
page: the Championship's attacks really are closer together than the
Bundesliga's.

Two scales exist, for the same reason SPI has two.

`LEAGUE` centres each competition on its own average, so 65 is an average club
*in this competition*. `EUROPE` centres on an average big-five club, the same
reference the global ranking quotes SPI against, so a rating there is comparable
across borders and most of Europe sits below 50. Neither is convertible into the
other, and the glossary says so.
"""
from __future__ import annotations

import math

#: The band ratings live in, and how sharply the logistic bends. 35/95 with a
#: slope of 1.3 puts an average club at 65 on the league scale, three standard
#: deviations above it at 90, and three below at 40.
LO, HI, SLOPE = 35.0, 95.0, 1.3

#: Spread of log attack and log defence within a competition, averaged over the
#: nine this site forecasts (2026-27 preseason). Attack is the wider of the two:
#: clubs differ more in what they score than in what they concede.
SD_ATT = 0.214
SD_DEF = 0.168

#: Spread of the same two across the pooled European corpus, which runs from
#: Bayern to the Luxembourg National Division and is therefore far wider.
SD_EUROPE = 0.55


def rating(z: float) -> int:
    """A z-score onto the 35-95 band. Whole numbers: a rating is not precise."""
    return round(LO + (HI - LO) / (1.0 + math.exp(-z / SLOPE)))


def attack(off: float, ref: float, sd: float) -> int:
    """Goals scored against an average opponent, as a rating. More is better."""
    if off <= 0 or ref <= 0:
        return round(LO + (HI - LO) / 2)
    return rating((math.log(off) - math.log(ref)) / sd)


def defence(dfn: float, ref: float, sd: float) -> int:
    """Goals conceded, as a rating. Fewer conceded is a *higher* number.

    This is the sign flip that makes the two columns readable side by side. It
    is also the whole reason the column was worth changing: "0.55" being better
    than "0.80" is something a reader has to remember, and "88" beating "71" is
    not.
    """
    if dfn <= 0 or ref <= 0:
        return round(LO + (HI - LO) / 2)
    return rating(-(math.log(dfn) - math.log(ref)) / sd)


def league_reference(offs: list[float], dfns: list[float]) -> tuple[float, float]:
    """The competition's own average attack and defence, in goals.

    The geometric mean, because the scale is a ratio scale: the club halfway
    between one that scores 1.0 and one that scores 4.0 scores 2.0, not 2.5.
    """
    def geo(xs: list[float]) -> float:
        good = [x for x in xs if x > 0]
        if not good:
            return 1.0
        return math.exp(sum(math.log(x) for x in good) / len(good))
    return geo(offs), geo(dfns)
