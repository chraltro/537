"""The one payload the browser needs to re-run the season for itself.

The 'what if' page lets a reader fix results and watch the table move, which
only works if the client can build the same score distributions the pipeline
uses. What ships is the fitted expectation for every unplayed fixture rather
than the ratings themselves: the browser then samples from exactly the grid
`model.simulate` would have sampled from, and cannot quietly drift away from
the published forecast by reimplementing the model in JavaScript.

One thing is deliberately left out. The headline forecast redraws team ratings
between scenarios, so it carries 'we are not certain how good these teams are'
as well as 'football is random'. A conditional re-run holds the ratings fixed,
because the question being asked is what these results would do, not what a
different set of ratings would do. The page says so.
"""
from __future__ import annotations

import datetime as dt
import json

from . import leagues, simulate


def _standings(fixtures, teams: list[str]) -> dict[str, dict]:
    """Points, goals and matches played from the results already on the board."""
    rows = {t: {"pts": 0, "gf": 0, "ga": 0, "played": 0} for t in teams}
    for f in fixtures:
        if not f.played or f.hg is None or f.ag is None:
            continue
        for t, gf, ga in ((f.home, f.hg, f.ag), (f.away, f.ag, f.hg)):
            r = rows.get(t)
            if r is None:
                continue
            r["played"] += 1
            r["gf"] += int(gf)
            r["ga"] += int(ga)
            r["pts"] += 3 if gf > ga else (1 if gf == ga else 0)
    return rows


def write_sim_input(fit, fixtures, teams, adj, meta, path, *,
                    league: leagues.League | None = None) -> dict:
    """Write sim_input.json: the standing table plus every fixture's expectation.

    Unplayed fixtures carry the two Poisson means; played ones carry the score.
    Nothing else, because this file is fetched before the page can draw anything
    -- except the three league shape numbers, so the worker can draw the European
    and relegation lines without a second request or a hardcoded constant.
    """
    lg = league or leagues.DEFAULT
    table = _standings(fixtures, teams)
    out_teams = []
    for t in teams:
        m = meta.get(t, {})
        s = table[t]
        out_teams.append({
            "id": t,
            "name": m.get("name", t),
            "short": m.get("short", t[:3].upper()),
            "primary": m.get("primary", "#7A8290"),
            "pts": s["pts"], "gf": s["gf"], "ga": s["ga"], "played": s["played"],
        })

    out_fixtures = []
    for f in sorted(fixtures, key=lambda x: (x.matchday or 0, x.date, x.home)):
        row = {"h": f.home, "a": f.away, "md": f.matchday,
               "date": f.date.isoformat(), "played": bool(f.played)}
        if f.played:
            row["hg"] = int(f.hg)
            row["ag"] = int(f.ag)
        else:
            lh, la = simulate._lambdas(fit, f.home, f.away, adj)
            row["lh"] = round(lh, 4)
            row["la"] = round(la, 4)
        out_fixtures.append(row)

    payload = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "rho": round(float(fit.rho), 6),
        "ucl_places": lg.ucl_places,
        "releg_places": lg.releg_places,
        "n_teams": lg.n_teams,
        "teams": out_teams,
        "fixtures": out_fixtures,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    return payload
