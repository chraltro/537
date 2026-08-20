"""Assembles the parsed sources into the match sets the model consumes.

A `Dataset` is always a dataset *of one league*: its top-flight history, its own
second tier (needed only to rate promoted clubs) and its fixture list. Nothing
in here knows which league it is beyond the `League` object it was handed.
"""
from __future__ import annotations

from datetime import date

from . import config, fetch, leagues
from .parse import Match, TeamRegistry, parse_football_data_csv, parse_openfootball


def _season_label(code: str) -> str:
    start = int(code[:2])
    century = 2000 if start < 90 else 1900
    return f"{century + start}-{code[2:]}"


class Dataset:
    def __init__(self, league: leagues.League | None = None,
                 season: str = config.SEASON) -> None:
        self.league = league or leagues.DEFAULT
        self.season = season
        self.reg = TeamRegistry()
        self.top: list[Match] = []           # this division's history
        self.second: list[Match] = []        # the tier below, to rate clubs coming up
        self.above: list[Match] = []         # the tier above, to rate clubs coming down
        self.fixtures: list[Match] = []      # the season ahead, in full
        #: Where this season's results actually came from, and how fresh they
        #: are. The build stamp says when the pipeline ran; this says what the
        #: pipeline could see, which is routinely weeks older -- the results
        #: mirror does not even create a season's CSV until months in (it added
        #: 2025-26 on 2026-02-17), so early-season builds run entirely off the
        #: openfootball fallback. A site that hides that is claiming currency it
        #: does not have.
        self.sources: dict = {}

    # Legacy names from the single-league pipeline. They alias the same lists,
    # so `ds.pl.append(...)` and `ds.top.append(...)` are the same call.
    @property
    def pl(self) -> list[Match]:
        return self.top

    @property
    def ch(self) -> list[Match]:
        return self.second

    # -- loading ------------------------------------------------------------
    def load(self) -> "Dataset":
        lg = self.league
        print(f"Loading {lg.name} history…")
        self.top = self._division_history(lg)
        print(f"  · {len(self.top)} matches over "
              f"{len({m.season for m in self.top})} seasons "
              f"(source: {lg.source})")

        if lg.above_slug:
            above = leagues.get(lg.above_slug)
            print(f"Loading {above.name} history (the division above)…")
            self.above = self._division_history(above)
            print(f"  · {len(self.above)} matches over "
                  f"{len({m.season for m in self.above})} seasons")

        print("Loading second-tier history…")
        for label in lg.second_season_labels(self.season):
            text = fetch.fixtures_text(lg, label, "second", required=False)
            if not text:
                continue
            self.second.extend(
                m for m in parse_openfootball(text, label, self.reg)
                if m.played and m.stage != "playoff")
        print(f"  · {len(self.second)} second-tier matches")

        print(f"Loading {self.season} fixtures…")
        text = fetch.fixtures_text(lg, self.season, "top")
        # The play-off is not part of the league season. The Championship's file
        # carries five knockout matches beside the 552 league ones, and treating
        # them as fixtures would have every club playing 47 games, a table that
        # counts a Wembley final as league points, and a fit that learns from a
        # neutral-ground tie as if it were a Tuesday at Ashton Gate. The
        # promotion play-off is simulated separately, from the league table the
        # simulation produces -- `knockout.promotion_playoff`.
        self.fixtures = [m for m in parse_openfootball(text, self.season, self.reg)
                         if m.stage != "playoff"]
        self._merge_current_results()
        self.validate()
        return self

    def _division_history(self, lg: leagues.League) -> list[Match]:
        """One division's played matches, from whichever feed carries it.

        Used for this league and, for a second tier, for the division above it.
        The two share `self.reg`, so a club is the same id in both and the fit
        can bridge them the way it already bridges a top flight and its own
        second tier.
        """
        out: list[Match] = []
        if lg.source == "openfootball":
            # No football-datasets directory exists for this competition -- the
            # mirror carries five leagues and no more -- so history comes from
            # the same plain-text feed as the fixtures. Goals only: no shots, no
            # cards, no half-time score, no referee.
            for label in lg.top_season_labels(self.season):
                if lg is self.league and label == self.season:
                    continue          # the fixture file below is that season
                text = fetch.fixtures_text(lg, label, "top", required=False)
                if not text:
                    continue
                out.extend(m for m in parse_openfootball(text, label, self.reg)
                           if m.played and m.stage != "playoff")
        else:
            for code in lg.fd_season_codes(self.season):
                label = _season_label(code)
                current = label == self.season
                required = (lg is self.league) and not current
                text = fetch.results_csv(lg, code, required=required)
                if not text:
                    if current and lg is self.league:
                        print(f"  · {label}: no results yet (season not started)")
                    continue
                out.extend(m for m in parse_football_data_csv(text, label, self.reg)
                           if m.played)
        return out

    def before(self, cutoff: date) -> list[Match]:
        """Every match the rating fit may see as of `cutoff`.

        One place decides what is in the corpus, because there were four and
        they had to agree: the live fit, the season-by-season rating history,
        the prior calibration and the walk-forward backtest. For a top flight
        this is its own record plus the tier below; for a second tier it is
        also the tier above, which is the only evidence about the three clubs
        that arrive from there every season.
        """
        return [m for m in self.top + self.second + self.above if m.date < cutoff]

    @property
    def kickoff(self) -> date:
        """The season's first scheduled fixture, used as the rating reference."""
        return min(f.date for f in self.fixtures)

    def _merge_current_results(self) -> None:
        """Fold any results already played into the fixture list.

        The football-data mirror is the primary in-season source; openfootball's
        own file is the fallback, so one source going quiet does not freeze the
        forecast.
        """
        played: dict[tuple[str, str], Match] = {}
        for m in self.top:
            if m.season == self.season:
                played[(m.home, m.away)] = m
        n_mirror = len(played)
        for m in self.fixtures:
            if m.played and (m.home, m.away) not in played:
                played[(m.home, m.away)] = m       # openfootball fallback
        n_fallback = len(played) - n_mirror

        for f in self.fixtures:
            src = played.get((f.home, f.away))
            if src is not None and src is not f:
                f.hg, f.ag, f.played = src.hg, src.ag, True
                f.hs, f.as_, f.hst, f.ast = src.hs, src.as_, src.hst, src.ast
                f.hthg, f.htag, f.referee = src.hthg, src.htag, src.referee
                f.hc, f.ac, f.hf, f.af = src.hc, src.ac, src.hf, src.af
                f.hy, f.ay, f.hr, f.ar = src.hy, src.ay, src.hr, src.ar
        # Make sure in-season results reach the rating fit exactly once.
        have = {(m.home, m.away) for m in self.top if m.season == self.season}
        for f in self.fixtures:
            if f.played and (f.home, f.away) not in have:
                self.top.append(f)
        done = sum(1 for f in self.fixtures if f.played)
        results = [f for f in self.fixtures if f.played]
        upcoming = [f for f in self.fixtures if not f.played]
        latest = max((f.date for f in results), default=None)
        nxt = min((f.date for f in upcoming), default=None)
        shots = sum(1 for f in results if f.hst is not None)
        lg = self.league
        mirror = (lg.source == "mirror")
        self.sources = {
            "kind": lg.source,
            "mirror": {
                "name": "football-datasets (football-data.co.uk mirror)",
                "url": (lg.fd_csv_url(lg.fd_season_codes(self.season)[-1])
                        if mirror else None),
                "played": n_mirror,
                "available": mirror,
            },
            "fixtures": {
                "name": "openfootball",
                "url": lg.of_url(self.season, "top"),
                "played": n_fallback,
                "available": True,
            },
            "results_to": latest.isoformat() if latest else None,
            "next_fixture": nxt.isoformat() if nxt else None,
            "played": done,
            "with_shots": shots,
        }
        print(f"  · {len(self.fixtures)} fixtures, {done} already played "
              f"({n_mirror} from the results mirror, {n_fallback} from openfootball)")

    # -- integrity ----------------------------------------------------------
    def validate(self) -> None:
        lg = self.league
        n = len(self.fixtures)
        if n != lg.n_matches:
            raise ValueError(
                f"{lg.slug}: expected {lg.n_matches} fixtures, parsed {n}")
        teams = {m.home for m in self.fixtures} | {m.away for m in self.fixtures}
        if len(teams) != lg.n_teams:
            raise ValueError(
                f"{lg.slug}: expected {lg.n_teams} teams, parsed {len(teams)}")
        for t in teams:
            h = sum(1 for m in self.fixtures if m.home == t)
            a = sum(1 for m in self.fixtures if m.away == t)
            if h != lg.n_teams - 1 or a != lg.n_teams - 1:
                raise ValueError(f"{lg.slug}: {t} has {h} home / {a} away fixtures")
        auto = [t for t in teams if self.reg.meta.get(t, {}).get("auto")]
        if auto:
            raise ValueError(f"{lg.slug}: unmapped club names in fixtures: {auto}")

    # -- views --------------------------------------------------------------
    @property
    def teams(self) -> list[str]:
        return sorted({m.home for m in self.fixtures})

    def pl_before(self, cutoff: date) -> list[Match]:
        return [m for m in self.top if m.date < cutoff]

    def season_table(self, season: str) -> dict[str, dict]:
        """Final (or current) league table for one season."""
        rows: dict[str, dict] = {}
        for m in (x for x in self.top if x.season == season and x.played):
            for t, gf, ga in ((m.home, m.hg, m.ag), (m.away, m.ag, m.hg)):
                r = rows.setdefault(t, {"team": t, "pld": 0, "w": 0, "d": 0, "l": 0,
                                        "gf": 0, "ga": 0, "pts": 0})
                r["pld"] += 1
                r["gf"] += gf
                r["ga"] += ga
                r["w"] += gf > ga
                r["d"] += gf == ga
                r["l"] += gf < ga
                r["pts"] += 3 if gf > ga else (1 if gf == ga else 0)
        return rows
