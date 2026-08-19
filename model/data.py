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
        self.top: list[Match] = []           # top-flight history, with shots
        self.second: list[Match] = []        # second-tier history, goals only
        self.fixtures: list[Match] = []      # the season ahead, in full

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
        for code in lg.fd_season_codes(self.season):
            label = _season_label(code)
            current = label == self.season
            text = fetch.results_csv(lg, code, required=not current)
            if not text:
                if current:
                    print(f"  · {label}: no results yet (season not started)")
                continue
            got = parse_football_data_csv(text, label, self.reg)
            self.top.extend(m for m in got if m.played)
        print(f"  · {len(self.top)} matches over "
              f"{len({m.season for m in self.top})} seasons")

        print("Loading second-tier history…")
        for label in lg.second_season_labels(self.season):
            text = fetch.fixtures_text(lg, label, "second", required=False)
            if not text:
                continue
            self.second.extend(
                m for m in parse_openfootball(text, label, self.reg) if m.played)
        print(f"  · {len(self.second)} second-tier matches")

        print(f"Loading {self.season} fixtures…")
        text = fetch.fixtures_text(lg, self.season, "top")
        self.fixtures = parse_openfootball(text, self.season, self.reg)
        self._merge_current_results()
        self.validate()
        return self

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
        for m in self.fixtures:
            if m.played and (m.home, m.away) not in played:
                played[(m.home, m.away)] = m       # openfootball fallback

        for f in self.fixtures:
            src = played.get((f.home, f.away))
            if src is not None and src is not f:
                f.hg, f.ag, f.played = src.hg, src.ag, True
                f.hs, f.as_, f.hst, f.ast = src.hs, src.as_, src.hst, src.ast
        # Make sure in-season results reach the rating fit exactly once.
        have = {(m.home, m.away) for m in self.top if m.season == self.season}
        for f in self.fixtures:
            if f.played and (f.home, f.away) not in have:
                self.top.append(f)
        done = sum(1 for f in self.fixtures if f.played)
        print(f"  · {len(self.fixtures)} fixtures, {done} already played")

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
