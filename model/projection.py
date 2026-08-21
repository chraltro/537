"""A projected final table for a league whose fixture list nobody publishes.

Most of Europe is not forecastable the way the Premier League is. There is no
fixture file for the Ekstraklasa or the Eliteserien on any feed this project can
read, so there is no season to play out: no matchweeks, no dates, no next round.
What there is, once `model.wikifootball` has read a season's results grid, is
the set of matches still to play -- because in a plain double round-robin the
pairs without a score against them are exactly the fixtures left, which is the
competition's format and not a guess about anyone's schedule.

That is enough for the question people actually ask, which is who wins it. The
simulation does not care what order the remaining matches come in: it plays each
of them once per simulated season and sorts the table. So a projection is a real
forecast of the final table, produced by the same code, off the same pooled
ratings as every other number on this site.

What a projection deliberately does not say
-------------------------------------------
**When.** No dates, no matchweeks, no calendar, no "next fixture". The grid
carries none of it and none of it is invented here.

**What a position is worth.** This project has no machine-readable source for
these competitions' qualification and relegation rules, and the rules move: a
European place follows a coefficient, a relegation line follows a licensing
decision. So the table is projected and the lines are not drawn. Positions are
simulated; what a position wins is left to the reader, who can look it up.

Everything published from here is therefore a probability over finishing
positions, plus the points that go with them, and nothing else.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from . import leagues, roundrobin, simulate
from .parse import Match

#: A stand-in competition for the simulator, which reads a league's European
#: and relegation lines to work out where its table's thresholds fall. A
#: projection publishes none of that -- see the module docstring -- and the two
#: numbers below are never read back out. They are 1 and 1 rather than 0 and 0
#: because `simulate_season` slices the position matrix with them, and a slice
#: of zero silently returns every column instead of none.
def _placeholder(name: str, n_teams: int) -> leagues.League:
    return leagues.League(
        slug="projection", name=name, country="", n_teams=n_teams,
        n_matches=roundrobin.season_size(n_teams),
        ucl_places=1, releg_places=1, releg_note=None, kind="projection")


@dataclass
class Projection:
    """One league's season, as far as a results grid can describe it."""

    slug: str
    name: str
    country: str
    season: str
    source: str                       # 'wikipedia', for the note on the page
    clubs: list[str]                  # club ids, the season's entrants
    played: list[Match] = field(default_factory=list)
    remaining: list[tuple[str, str]] = field(default_factory=list)

    @property
    def n_teams(self) -> int:
        return len(self.clubs)

    @property
    def n_matches(self) -> int:
        return roundrobin.season_size(self.n_teams)


def fixtures(clubs: list[str], played: list[Match],
             *, when: dt.date) -> list[Match]:
    """The whole season as `Match` objects: the played ones, then the derived rest.

    The derived ones carry `date_approx` and the date of the newest result in
    the league, which is not a claim about when they will be played -- nothing
    knows that -- but the only ordering key available. Every reader of these
    fixtures is told, by the flag, not to print it.
    """
    done = [(m.home, m.away) for m in played]
    out = list(played)
    for home, away in roundrobin.remaining(clubs, done):
        out.append(Match(date=when, home=home, away=away, played=False,
                         season=played[0].season if played else "",
                         extra={"date_approx": True, "date_unknown": True}))
    return out


def table(clubs: list[str], played: list[Match]) -> dict[str, dict]:
    """The league table as it stands, from the grid's own results."""
    rows = {c: {"team": c, "pld": 0, "w": 0, "d": 0, "l": 0,
                "gf": 0, "ga": 0, "pts": 0} for c in clubs}
    for m in played:
        if not m.played or m.hg is None or m.ag is None:
            continue
        for t, gf, ga in ((m.home, m.hg, m.ag), (m.away, m.ag, m.hg)):
            r = rows[t]
            r["pld"] += 1
            r["gf"] += gf
            r["ga"] += ga
            if gf > ga:
                r["w"] += 1
                r["pts"] += 3
            elif gf == ga:
                r["d"] += 1
                r["pts"] += 1
            else:
                r["l"] += 1
    for r in rows.values():
        r["gd"] = r["gf"] - r["ga"]
    return rows


def run(proj: Projection, fit, *, n_sims: int = 20000, seed: int = 537,
        rating_sd=None) -> dict:
    """Play the rest of the season and turn it into the payload the site reads.

    Only the keys a projection is entitled to. `simulate_season` also returns
    European and relegation probabilities and the points thresholds that go
    with them, computed against the placeholder above, and every one of those is
    dropped here rather than published against rules this project does not hold.
    """
    when = max((m.date for m in proj.played), default=dt.date.today())
    fx = fixtures(proj.clubs, proj.played, when=when)
    teams = sorted(proj.clubs)
    lg = _placeholder(proj.name, len(teams))
    kw = {} if rating_sd is None else {"rating_sd": rating_sd}
    sim = simulate.simulate_season(fit, fx, teams, league=lg, n_sims=n_sims,
                                   seed=seed, curves=False, leverage=False, **kw)

    standing = table(proj.clubs, proj.played)
    rows = []
    for i, t in enumerate(teams):
        r = standing[t]
        rows.append({
            "id": t,
            "pld": r["pld"], "w": r["w"], "d": r["d"], "l": r["l"],
            "gf": r["gf"], "ga": r["ga"], "gd": r["gd"], "now": r["pts"],
            "pts": round(float(sim["points_mean"][i]), 1),
            "pts_lo": round(float(sim["points_p10"][i]), 1),
            "pts_hi": round(float(sim["points_p90"][i]), 1),
            "title": round(float(sim["title"][i]), 4),
            "last": round(float(sim["position"][i, -1]), 4),
            "position": [round(float(x), 4) for x in sim["position"][i]],
        })
    rows.sort(key=lambda r: (-r["pts"], -r["title"]))
    return {
        "slug": proj.slug, "name": proj.name, "country": proj.country,
        "season": proj.season, "kind": "projection",
        "n_teams": proj.n_teams, "n_matches": proj.n_matches,
        "matches_played": sum(1 for m in proj.played if m.played),
        "matches_total": proj.n_matches,
        "n_sims": int(sim["n_sims"]),
        "source": proj.source,
        "fixtures_known": False,
        "teams": rows,
    }
