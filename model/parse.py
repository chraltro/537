"""Source readers and the one thing that silently breaks football data pipelines:
club-name normalisation across sources that spell the same club three ways."""
from __future__ import annotations

import csv
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_PATH = os.path.join(HERE, "data", "team_meta.json")

_DROP_TOKENS = {"fc", "afc"}
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def normalise(name: str) -> str:
    """Fold a club name to a comparable key.

    'AFC Bournemouth', 'Bournemouth' and 'Bournemouth FC' all land on the same
    key; 'Man City' and 'Man United' deliberately do not.

    Accents are folded first, because the two feeds disagree about them within a
    single club: the results mirror writes 'Malaga' and 'M'gladbach' where
    openfootball writes 'Málaga CF' and 'Borussia Mönchengladbach'. Decomposing
    to NFKD and dropping the combining marks makes those the same key without a
    per-club alias for every accented name in four countries.
    """
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # Ligatures and the Nordic/German letters NFKD leaves alone.
    s = s.lower().replace("ß", "ss").replace("ø", "o").replace("æ", "ae")
    s = s.replace("đ", "d").replace("ł", "l").replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    toks = [t for t in s.split() if t not in _DROP_TOKENS]
    return " ".join(toks)


class TeamRegistry:
    """Maps every spelling seen in any source onto a stable club id.

    Clubs outside the curated set (lower-league sides that appear in Championship
    history) are auto-registered rather than dropped: they still carry real
    signal about how strong a promoted club's opponents were.
    """

    def __init__(self) -> None:
        self.meta: dict[str, dict] = json.load(open(META_PATH))
        self._by_key: dict[str, str] = {}
        for tid, m in self.meta.items():
            for key in [normalise(m["name"]), normalise(tid.replace("-", " ")), tid]:
                self._by_key[key] = tid
            for alias in m.get("aliases", []):
                self._by_key[normalise(alias)] = tid

    def resolve(self, name: str) -> str:
        key = normalise(name)
        if key in self._by_key:
            return self._by_key[key]
        tid = re.sub(r"[^a-z0-9]+", "-", key).strip("-")
        self._by_key[key] = tid
        self.meta.setdefault(tid, {
            "id": tid, "name": name.strip(), "short": tid[:3].upper(),
            "primary": "#7A8290", "secondary": "#FFFFFF", "aliases": [], "auto": True,
        })
        return tid

    def display(self, tid: str) -> str:
        return self.meta.get(tid, {}).get("name", tid)


@dataclass
class Match:
    date: date
    home: str
    away: str
    hg: int | None = None          # full-time goals
    ag: int | None = None
    hs: int | None = None          # shots
    as_: int | None = None
    hst: int | None = None         # shots on target
    ast: int | None = None
    matchday: int | None = None
    time: str | None = None
    season: str = ""
    played: bool = False
    extra: dict = field(default_factory=dict)


def _to_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def parse_football_data_csv(text: str, season: str, reg: TeamRegistry) -> list[Match]:
    """football-data.co.uk schema as mirrored by datasets/football-datasets."""
    out: list[Match] = []
    for row in csv.DictReader(text.splitlines()):
        if not row.get("HomeTeam") or not row.get("AwayTeam"):
            continue
        raw = (row.get("Date") or "").strip()
        d = None
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"):
            try:
                d = datetime.strptime(raw, fmt).date()
                break
            except ValueError:
                continue
        if d is None:
            continue
        hg, ag = _to_int(row.get("FTHG")), _to_int(row.get("FTAG"))
        out.append(Match(
            date=d, home=reg.resolve(row["HomeTeam"]), away=reg.resolve(row["AwayTeam"]),
            hg=hg, ag=ag,
            hs=_to_int(row.get("HS")), as_=_to_int(row.get("AS")),
            hst=_to_int(row.get("HST")), ast=_to_int(row.get("AST")),
            season=season, played=hg is not None and ag is not None,
        ))
    return out

_MD_RE = re.compile(r"^\s*(?:▪)?\s*(?:Matchday|Regular Season -|Round)\s*(\d+)", re.I)
_DATE_RE = re.compile(
    r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+([A-Za-z]{3})\w*\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
# A trailing '2-1 (1-0)' / '2-1' result, however the rest of the line is laid out.
_EXTRA = r"(?:\s*(?:a\.?e\.?t\.?|pen\.?s?\.?|aet))?"
_TRAIL_SCORE = re.compile(r"\s{2,}(\d+)\s*-\s*(\d+)" + _EXTRA + r"\s*(?:\([^)]*\))?" + _EXTRA + r"\s*$", re.I)
_TIME = re.compile(r"^\s*(\d{1,2}:\d{2})\s+")
_NOTE = re.compile(r"\s*\[[^\]]*\]\s*")
# Club names outside England are full of digits -- 'Como 1907', '1. FC Köln',
# 'Bayer 04 Leverkusen', 'Stade Rennais FC 1901'. So a side cannot be rejected
# for containing a digit; it is rejected for not containing a word.
_WORD = re.compile(r"[^\W\d_]{2,}", re.UNICODE)
_NOT_A_CLUB = re.compile(r"\d\s*[-:]\s*\d")


def _is_club(s: str) -> bool:
    return bool(s) and bool(_WORD.search(s)) and not _NOT_A_CLUB.search(s)


def parse_openfootball(text: str, season: str, reg: TeamRegistry) -> list[Match]:
    """openfootball plain-text league files.

    Dates are section headers that carry forward, the year is stated only when it
    changes, and the repository uses three different line layouts across seasons:
    'Home v Away', 'Home v Away  2-1 (1-0)', and 'Home  2-1 (1-0)  Away'. All
    three appear in files this pipeline reads, so all three are handled here.
    """
    start_year = int(season.split("-")[0])
    out: list[Match] = []
    cur_date: date | None = None
    matchday: int | None = None
    for raw_line in text.splitlines():
        line = _NOTE.sub("  ", raw_line.rstrip())
        if not line.strip() or line.lstrip().startswith("#") or line.startswith("="):
            continue
        m = _MD_RE.match(line)
        if m:
            matchday = int(m.group(1))
            continue
        m = _DATE_RE.match(line)
        if m:
            mon = _MONTHS.get(m.group(1).lower())
            if mon:
                year = int(m.group(3)) if m.group(3) else (
                    start_year if mon >= 7 else start_year + 1)
                cur_date = date(year, mon, int(m.group(2)))
            continue
        if re.match(r"^\s*\(", line) or cur_date is None:
            continue          # scorer continuation, or content before any date header

        tm = _TIME.match(line)
        time = tm.group(1) if tm else None
        body = line[tm.end():] if tm else line

        hg = ag = None
        sm = _TRAIL_SCORE.search(body)
        if sm:
            hg, ag = int(sm.group(1)), int(sm.group(2))
            body = body[:sm.start()]

        if " v " in body:
            home, _, away = body.partition(" v ")
        else:
            sm2 = re.search(r"\s{2,}(\d+)\s*-\s*(\d+)" + _EXTRA + r"\s*(?:\([^)]*\))?" + _EXTRA + r"\s{2,}", body, re.I)
            if not sm2:
                continue
            hg, ag = int(sm2.group(1)), int(sm2.group(2))
            home, away = body[:sm2.start()], body[sm2.end():]

        home, away = home.strip(), away.strip()
        if not _is_club(home) or not _is_club(away):
            continue
        out.append(Match(date=cur_date, home=reg.resolve(home), away=reg.resolve(away),
                         hg=hg, ag=ag, matchday=matchday, time=time, season=season,
                         played=hg is not None and ag is not None))
    return out
