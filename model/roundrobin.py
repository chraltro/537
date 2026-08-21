"""The fixtures a double round-robin has left, worked out from the ones it has played.

Twenty-one of the leagues this project rates have no fixture list anywhere: no
GitHub feed carries them, and the publishers that do want a key and a bill. What
they do have is a results grid on Wikipedia -- one cell per ordered pair of
clubs -- and for a competition where every club plays every other exactly twice,
once at home and once away, that grid is enough to say what is left.

It is enough because the remaining fixtures are not a guess. A sixteen-club
double round-robin is 240 matches, one per ordered pair; the grid says which of
those 240 have a score in them; the rest are the ones still to play. That is the
competition's format, published in its own rules, not an inference about a
schedule. Which is also the limit of it: the *order* is a schedule and the grid
does not carry one, so nothing here invents a date, a matchweek or a kick-off
time. A league read this way knows exactly who still has to play whom, and
nothing whatever about when.

The shape is checked before it is trusted. A file that is not a plain double
round-robin -- a league that splits into championship and relegation rounds
after thirty-three games, or plays a third round, or runs a play-off -- produces
a pair that appears twice or a club with too many home games, and `remaining`
refuses rather than returning a fixture list for a competition that does not
work the way it was assumed to.
"""
from __future__ import annotations


class ShapeError(RuntimeError):
    """The played matches are not a partial plain double round-robin.

    Raised rather than worked around. The whole reason a remaining-fixture list
    can be derived at all is that the format fixes it, so a format that does not
    fit is not a fixture list with a few odd rows in it -- it is a different
    competition, and guessing at one is exactly what this project does not do.
    """


def all_pairs(clubs: list[str]) -> list[tuple[str, str]]:
    """Every ordered pair, which for this format is every fixture of the season.

    Sorted, so two runs of the same season produce the same list in the same
    order and a rebuild does not churn the output files.
    """
    names = sorted(clubs)
    return [(h, a) for h in names for a in names if h != a]


def check(clubs: list[str], played: list[tuple[str, str]]) -> None:
    """Refuse anything that is not a partial plain double round-robin."""
    names = set(clubs)
    if len(names) != len(clubs):
        raise ShapeError("the club list contains a duplicate")
    if len(names) < 4:
        raise ShapeError(f"{len(names)} clubs is not a league")

    # Every club plays every other once at each ground, so a club can appear at
    # most once per ordered pair and the two checks below are the whole format:
    # a pair that repeats, and a club nobody listed. There is deliberately no
    # third check on how many home games a club has, because the arithmetic
    # makes it unreachable -- exceeding it needs either a repeated pair or an
    # opponent outside the list, and both are already refused.
    seen: set[tuple[str, str]] = set()
    for h, a in played:
        if h not in names or a not in names:
            missing = h if h not in names else a
            raise ShapeError(f"{missing!r} played a match but is not in the club list")
        if h == a:
            raise ShapeError(f"{h!r} is recorded as playing itself")
        if (h, a) in seen:
            raise ShapeError(
                f"{h!r} v {a!r} appears twice, so this is not a competition where "
                "each pair meets once at each ground")
        seen.add((h, a))


def remaining(clubs: list[str],
              played: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """The ordered pairs that have not met yet, in a stable order.

    `played` is every pair with a score against it. The result plus `played` is
    the whole season, which is what `model.data.Dataset.validate` then checks
    against the league's own declared size -- so a grid that is missing a club,
    or carrying one too many, fails there as well as here.
    """
    check(clubs, played)
    done = set(played)
    return [pair for pair in all_pairs(clubs) if pair not in done]


def season_size(n_clubs: int) -> int:
    """How many matches a double round-robin of this size is."""
    return n_clubs * (n_clubs - 1)
