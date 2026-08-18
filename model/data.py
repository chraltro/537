"""Assembles the parsed sources into the match sets the model consumes."""
from __future__ import annotations

from datetime import date

from . import config, fetch
from .parse import Match, TeamRegistry, parse_football_data_csv, parse_openfootball


def _season_label(code: str) -> str:
    start = int(code[:2])
    century = 2000 if start < 90 else 1900
    return f"{century + start}-{code[2:]}"


class Dataset:
    def __init__(self) -> None:
        self.reg = TeamRegistry()
        self.pl: list[Match] = []            # Premier League history, with shots
        self.ch: list[Match] = []            # Championship history, goals only
        self.fixtures: list[Match] = []      # the 380 matches of the season ahead

    # -- loading ------------------------------------------------------------
    def load(self) -> "Dataset":
        print("Loading Premier League history…")
        for code in config.PL_SEASONS:
            label = _season_label(code)
            current = label == config.SEASON
            text = fetch.premier_league_csv(code, required=not current)
            if not text:
                if current:
                    print(f"  · {label}: no results yet (season not started)")
                continue
            got = parse_football_data_csv(text, label, self.reg)
            self.pl.extend(m for m in got if m.played)
            print(f"  · {label}: {sum(m.played for m in got)} matches")

        print("Loading Championship history…")
        for label in config.CH_SEASONS:
            text = fetch.openfootball(label, "2-championship", required=False)
            if not text:
                continue
            got = [m for m in parse_openfootball(text, label, self.reg) if m.played]
            self.ch.extend(got)
        print(f"  · {len(self.ch)} Championship matches")

        print(f"Loading {config.SEASON} fixtures…")
        text = fetch.openfootball(config.SEASON, "1-premierleague")
        self.fixtures = parse_openfootball(text, config.SEASON, self.reg)
        self._merge_current_results()
        self.validate()
        return self

    def _merge_current_results(self) -> None:
        """Fold any 2026/27 results already played into the fixture list.

        The football-data mirror is the primary in-season source; openfootball's
        own file is the fallback, so one source going quiet does not freeze the
        forecast.
        """
        played: dict[tuple[str, str], Match] = {}
        for m in self.pl:
            if m.season == config.SEASON:
                played[(m.home, m.away)] = m
        for m in self.fixtures:
            if m.played and (m.home, m.away) not in played:
                played[(m.home, m.away)] = m       # openfootball fallback

        merged = 0
        for f in self.fixtures:
            src = played.get((f.home, f.away))
            if src is not None and src is not f:
                f.hg, f.ag, f.played = src.hg, src.ag, True
                f.hs, f.as_, f.hst, f.ast = src.hs, src.as_, src.hst, src.ast
                merged += 1
        # Make sure in-season results reach the rating fit exactly once.
        have = {(m.home, m.away) for m in self.pl if m.season == config.SEASON}
        for f in self.fixtures:
            if f.played and (f.home, f.away) not in have:
                self.pl.append(f)
        done = sum(1 for f in self.fixtures if f.played)
        print(f"  · {len(self.fixtures)} fixtures, {done} already played")

    # -- integrity ----------------------------------------------------------
    def validate(self) -> None:
        n = len(self.fixtures)
        if n != config.N_MATCHES:
            raise ValueError(f"expected {config.N_MATCHES} fixtures, parsed {n}")
        teams = {m.home for m in self.fixtures} | {m.away for m in self.fixtures}
        if len(teams) != config.N_TEAMS:
            raise ValueError(f"expected {config.N_TEAMS} teams, parsed {len(teams)}")
        for t in teams:
            h = sum(1 for m in self.fixtures if m.home == t)
            a = sum(1 for m in self.fixtures if m.away == t)
            if h != config.N_TEAMS - 1 or a != config.N_TEAMS - 1:
                raise ValueError(f"{t} has {h} home / {a} away fixtures")
        auto = [t for t in teams if self.reg.meta.get(t, {}).get("auto")]
        if auto:
            raise ValueError(f"unmapped club names in fixtures: {auto}")

    # -- views --------------------------------------------------------------
    @property
    def teams(self) -> list[str]:
        return sorted({m.home for m in self.fixtures})

    def pl_before(self, cutoff: date) -> list[Match]:
        return [m for m in self.pl if m.date < cutoff]

    def season_table(self, season: str) -> dict[str, dict]:
        """Final (or current) league table for one season."""
        rows: dict[str, dict] = {}
        for m in (x for x in self.pl if x.season == season and x.played):
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
