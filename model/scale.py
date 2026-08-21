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

One scale, and only one. 65 is an average big-five club, the same reference the
global ranking quotes SPI against, so a rating means the same thing whichever
league its club plays in and most of Europe sits below 50.

There were two for a while, one centred on each competition and one on Europe,
and the site published both without saying which was which: the league table
gave Arsenal's defence as 89 and the club page gave 81. Worse, the comparison
page drew a single radar out of two axes on one scale and five on the other. A
rating that is only meaningful next to its own neighbours is a rank with extra
steps, so the league scale is gone rather than labelled.
"""
from __future__ import annotations

import math

#: The band ratings live in, and how sharply the logistic bends. 35/95 with a
#: slope of 1.3 puts an average club at 65 on the league scale, three standard
#: deviations above it at 90, and three below at 40.
LO, HI, SLOPE = 35.0, 95.0, 1.3

#: Spread of log attack and log defence across the pooled European corpus, which
#: runs from Bayern to the Luxembourg National Division and is therefore far
#: wider than any one division. Measured once and frozen; `tools/measure_scale.py`
#: reports the drift.
SD_EUROPE = 0.55

# --------------------------------------------------------------------------
# The other six
# --------------------------------------------------------------------------
#: Every rating on this site is the same transform over a different measurable,
#: and each spread below was measured across all nine competitions (five
#: seasons, 2021-22 to 2025-26) and then frozen -- see the module docstring for
#: why they are constants rather than recomputed.
#:
#: True where a bigger measurement is a better club. Where it is False the
#: z-score is negated before the logistic, so that every rating on the site
#: reads the same way round: bigger is better, always, without exception.
DIMENSIONS: dict[str, bool] = {
    # Universal -- goals and dates only, so every competition has them.
    #
    # Standard deviation of goal difference, match to match, inverted. High
    # means predictable, and predictable is not the same as good: a side that
    # loses narrowly every week scores well here. It is a description, not a
    # compliment, and the page says so.
    "consistency": False,
    # Big five only -- these need a shot or a card, and only the results mirror
    # has one.
    #
    # Shots on target per match, logged. Deliberately separate from attack: a
    # club that creates plenty and finishes badly is the interesting case, and
    # one number cannot show it.
    "creation": True,
    # Goals per shot on target, logged.
    "finishing": True,
    # Yellow + three per red + a sixth of a foul, per match, inverted -- so a
    # high rating is a clean side.
    "discipline": False,
}

#: What fraction of each measure's spread across clubs is real, as opposed to
#: the luck of the matches we happened to see.
#:
#: For a mean over n matches the observed variance between clubs is the true
#: variance plus the sampling variance, and the second is estimable from the
#: matches themselves. `tools/measure_scale.py` prints the split.
#:
#: This is why two dimensions this site used to publish no longer exist.
#: *Home advantage* came out at 0.07 and *big games* at 0.04: ninety-three and
#: ninety-six per cent of what looked like clubs differing was one club's luck
#: over seventy matches, or over the eighteen hard ones. Both are well-known
#: results -- club-specific home advantage is tiny, and beating expectation
#: against strong sides does not persist -- and neither was survivable as a
#: number out of 100 next to five that mean something.
#:
#: The four that remain are shrunk toward 65 by their own reliability, which is
#: Kelley's estimate of a true score: a measurement two standard deviations out
#: on a measure that resolves 60% of what it sees is best read as 1.2 out.
RELIABILITY: dict[str, float] = {
    "consistency": 0.65,      # 90 matches of goal difference
    "creation": 0.94,         # ~1,000 shots on target per club
    "finishing": 0.59,        # a conversion rate is thin even over five seasons
    "discipline": 0.92,
}


#: The spread of each, across every club that has it: 879 for the one that needs
#: only goals and dates, 88 for the three that need a shot or a card, whose whole
#: population is the big five.
#:
#: The within-league spreads these replaced are quoted beside each, because the
#: difference is the interesting part. Consistency widens a long way -- a corpus
#: running from Bayern to Luxembourg spreads out more than any single division
#: does -- while the three shot-based ones barely move, since their population
#: was always the big five and the only change is using one average for all five
#: rather than five averages.
#:
#: Measured once, at the 2026-27 preseason build, and frozen -- see the module
#: docstring for why these are constants. `tools/measure_scale.py` regenerates
#: them and prints what changed.
EUROPE_SD: dict[str, float] = {
    "consistency": 0.2681,    # within a league: 0.14
    "creation": 0.1841,       # within a league: 0.176
    "finishing": 0.1017,      # within a league: 0.098
    "discipline": 0.5634,     # within a league: 0.48
}


def dimension(name: str, value: float, ref: float, *, log: bool = False,
              europe: bool = True) -> int | None:
    """One measurable against an average club, as a rating out of 100.

    `europe` is kept as a parameter and ignored: it is always true now, and
    every caller passes it, so removing it would be a diff across four files to
    say the same thing. Nothing here can produce a league-relative rating.

    `log` for quantities that are ratios rather than differences -- a shot count
    or a conversion rate, where twice the average is as far above it as half is
    below. Points-per-game differences are already differences and are not
    logged.
    """
    higher = DIMENSIONS.get(name)
    if higher is None or value is None or ref is None:
        return None
    sd = EUROPE_SD.get(name)
    if sd is None:
        return None
    if log:
        if value <= 0 or ref <= 0:
            return None
        z = (math.log(value) - math.log(ref)) / sd
    else:
        z = (value - ref) / sd
    # Shrunk toward the middle by how much of this measure is real. Without it
    # a rating states more than the matches behind it support, which is the
    # failure that cost this site its home-advantage and big-game axes.
    z *= RELIABILITY.get(name, 1.0)
    return rating(z if higher else -z)


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
