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

    def known(self, name: str) -> str | None:
        """`resolve` without the auto-registration, for a source we do not trust yet.

        `resolve` mints an id for a spelling it has never seen, which is right
        for a Championship opponent from 2004 and wrong for a second feed of a
        league we already carry: football-data.co.uk writes "Lech Poznan" where
        openfootball writes "KKS Lech Poznań", and auto-registering that would
        put twenty duplicate Polish clubs in the ranking, each with half a
        season and no history, rather than updating the twenty already there.

        So a new source resolves through this, and a league whose clubs do not
        all land on ids we already hold does not enter the corpus at all.
        """
        return self._by_key.get(normalise(name))

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
    # -- half time and discipline -------------------------------------------
    # Present in every football-data mirror CSV and, until now, read by nothing.
    # Half-time goals are complete from 2000-01 in all five leagues; cards,
    # corners and fouls likewise. `referee` is, in practice, a Premier League
    # column: 26 seasons there (2000-01..2025-26) against two in the Bundesliga
    # files, two in Serie A's and none at all in La Liga's or Ligue 1's.
    hthg: int | None = None        # half-time goals
    htag: int | None = None
    hc: int | None = None          # corners
    ac: int | None = None
    hf: int | None = None          # fouls conceded
    af: int | None = None
    hy: int | None = None          # yellow cards
    ay: int | None = None
    hr: int | None = None          # red cards
    ar: int | None = None
    referee: str | None = None
    matchday: int | None = None
    time: str | None = None
    season: str = ""
    played: bool = False
    extra: dict = field(default_factory=dict)

    # -- European competition metadata --------------------------------------
    # Only populated by the openfootball reader when the source says so, so a
    # domestic Match is byte-identical to what it was before these existed.
    #: normalised stage: 'league' | 'playoff' | 'r16' | 'qf' | 'sf' | 'final'
    #: | 'q1'..'q4' | None
    stage: str | None = None
    #: leg number inside a two-legged tie (1 or 2), when the source states it
    leg: int | None = None
    #: three-letter association codes carried by European club names, kept
    #: because the draw constraints in the plan's 3.5 need them. The club id
    #: itself is resolved WITHOUT the suffix so 'Arsenal FC (ENG)' and
    #: 'Arsenal' are one club.
    home_assoc: str | None = None
    away_assoc: str | None = None
    #: 'cl' | 'el' | 'conf' | 'clq' | ... for European files, '' for domestic
    comp: str = ""
    #: True when the ingested score is the 90-minute score of a tie that went
    #: to extra time (the a.e.t./penalty result is deliberately discarded).
    aet: bool = False


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
            hthg=_to_int(row.get("HTHG")), htag=_to_int(row.get("HTAG")),
            hc=_to_int(row.get("HC")), ac=_to_int(row.get("AC")),
            hf=_to_int(row.get("HF")), af=_to_int(row.get("AF")),
            hy=_to_int(row.get("HY")), ay=_to_int(row.get("AY")),
            hr=_to_int(row.get("HR")), ar=_to_int(row.get("AR")),
            referee=(row.get("Referee") or "").strip() or None,
            season=season, played=hg is not None and ag is not None,
        ))
    return out

#: A round heading in an openfootball domestic file, in every spelling the
#: corpus actually contains.
#:
#: This used to accept `Matchday 3`, `Regular Season - 3` and `Round 3` only,
#: which is three of the seven forms in use -- and it failed on every heading in
#: Belgium's file, which writes `▪ 1. Round`. The result was 306 fixtures with
#: no matchweek at all: the What-if simulator's week selector was empty and its
#: heading read "NaN", the front page could not group the next round, and
#: nothing failed loudly enough for anybody to notice. The European reader two
#: hundred lines below had handled `1. Round` since the day it was written.
#:
#: Handled, in the order they appear here:
#:   ▪ Matchday 3            ▪▪ Matchday 3        (a doubled marker)
#:   ▪ Regular Season - 3    ▪ Regular, Matchday 3
#:   ▪ Championship, Matchday 3 / ▪ Relegation, Matchday 3   (a split phase)
#:   ▪ Round 3               ▪ 3. Round           ▪ 3. Round (datum TBC)
#:
#: Knockout headings -- `▪ Final`, `▪ Semifinals`, `▪ Round of 16` -- carry no
#: bare number after the keyword and so still match nothing, which is correct:
#: a domestic file's knockout rounds are not matchweeks.
_MD_RE = re.compile(
    r"""^\s*▪*\s*                      # zero or more round markers
        (?:[^,\d]*,\s*)?               # an optional phase prefix, e.g. 'Championship, '
        (?:
            (?:Matchday|Regular\s+Season\s*-|Round|Spieltag|Speeldag|Jornada)
            \s*(\d+)
          | (\d+)\s*\.\s*Round        # '3. Round', the form Belgium uses
        )""",
    re.I | re.X)
_DATE_RE = re.compile(
    r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+([A-Za-z]{3})\w*\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
#: A kick-off time, or the placeholder a feed uses when the time is not yet
#: fixed. Belgium's 2026-27 file writes `--:--` for one match whose slot the
#: league had not announced; without this the placeholder is read as part of
#: the home club's name and the fixture is silently dropped.
_TIME = re.compile(r"^\s*(?:(\d{1,2}:\d{2})|-{1,2}:-{1,2}|\?{1,2}:\?{1,2})\s+")
_NOTE = re.compile(r"\s*\[[^\]]*\]\s*")
# Club names outside England are full of digits -- 'Como 1907', '1. FC Köln',
# 'Bayer 04 Leverkusen', 'Stade Rennais FC 1901'. So a side cannot be rejected
# for containing a digit; it is rejected for not containing a word.
_WORD = re.compile(r"[^\W\d_]{2,}", re.UNICODE)
_NOT_A_CLUB = re.compile(r"\d\s*[-:]\s*\d")
#: The three-letter association code European files append to every club name.
#: Stripped before `resolve()` so 'Arsenal FC (ENG)' is the same club as the one
#: the Premier League feed calls 'Arsenal', and kept on the Match because the
#: draw constraints need to know who is Spanish.
_ASSOC = re.compile(r"\s*\(([A-Z]{3})\)\s*$")

# The result block at the end (or, in the old three-column layout, the middle)
# of a fixture line. Five shapes appear in the corpus and all five are here:
#     2-1                       full time, no half-time given
#     2-1 (1-0)                 full time (half time)
#     2-1 a.e.t. (1-1, 0-1)     after extra time (90 minutes, half time)
#     2-0 a.e.t. (0-0)          after extra time (90 minutes)
#     4-3 pen. 1-1 a.e.t. (1-1, 0-1)   shootout, a.e.t., (90 minutes, half time)
# The model wants the 90-MINUTE score every time: a shootout is a coin flip and
# extra time is a different game with a different scoring rate, so folding either
# into a Dixon-Coles fit teaches it that knockout ties are high-scoring.
_RESULT = r"""
    (?:(?P<pen_h>\d+)\s*-\s*(?P<pen_a>\d+)\s*pen\.?s?\.?\s+)?
    (?P<h>\d+)\s*-\s*(?P<a>\d+)
    (?P<aet>\s*(?:a\.?e\.?t\.?|aet))?
    (?:\s*\(\s*(?P<p1h>\d+)\s*-\s*(?P<p1a>\d+)
       (?:\s*,\s*(?P<p2h>\d+)\s*-\s*(?P<p2a>\d+))?\s*\))?
"""
# One space, not two, before a trailing score: the corpus is mostly aligned
# in columns but not always -- Belgium's 2026-27 file writes
# `Club Brugge v KV Kortrijk 3-0` for one fixture, and requiring two spaces
# left the score glued to the club name, `_is_club` rejected it for holding
# a digit-hyphen-digit, and the match vanished. A trailing score is anchored
# to the end of the line, so relaxing this cannot swallow a club name.
_TRAIL_SCORE = re.compile(r"\s+" + _RESULT + r"\s*$", re.I | re.X)
_MID_SCORE = re.compile(r"\s{2,}" + _RESULT + r"\s{2,}", re.I | re.X)


def _score_from(m: "re.Match") -> tuple[int, int, bool, tuple[int, int] | None]:
    """The 90-minute score, whether the tie went past it, and the half-time score.

    When `a.e.t.` is present the leading pair is the score after 120 minutes and
    the FIRST pair inside the parentheses is the score after 90 -- verified
    against the corpus, e.g. `Juventus v Galatasaray 3-2 a.e.t. (3-0, 1-0)`,
    which was 3-0 at full time (levelling a 2-5 first leg) before extra time.
    Without `a.e.t.` the leading pair is full time and the parenthesis is the
    half-time score, which is the plain domestic layout.
    """
    aet = bool(m.group("aet"))
    p1 = ((int(m.group("p1h")), int(m.group("p1a")))
          if m.group("p1h") is not None else None)
    p2 = ((int(m.group("p2h")), int(m.group("p2a")))
          if m.group("p2h") is not None else None)
    if aet and p1 is not None:
        # `2-1 a.e.t. (1-1, 0-1)`: the parentheses hold 90 minutes then half time.
        return p1[0], p1[1], True, p2
    # `2-1 (1-0)`: the parenthesis is the half-time score, which the mirror
    # feed gives us as its own column and this feed has been throwing away.
    return int(m.group("h")), int(m.group("a")), aet, p1


def _is_club(s: str) -> bool:
    return bool(s) and bool(_WORD.search(s)) and not _NOT_A_CLUB.search(s)


#: Stage headers, normalised. openfootball spells the same stage four ways
#: across fifteen seasons -- '▪ Group A', '▪ League phase', '▪ League, Matchday 3'
#: and '▪ Group, Matchday 3' are all the group/league phase -- so the reader maps
#: them onto one vocabulary rather than making every consumer know the history.
_STAGE_HEAD = re.compile(r"^\s*▪+\s*(.+?)\s*$")
_MD_IN = re.compile(r"^Matchday\s+(\d+)$", re.I)
_ROUND_IN = re.compile(r"^(?:Round\s+(\d+)|(\d+)\.\s*Round)$", re.I)
_FINAL_STAGES = {
    "round of 32": "r32", "round of 16": "r16",
    "quarterfinals": "qf", "quarter-finals": "qf",
    "semifinals": "sf", "semi-finals": "sf", "final": "final",
    # 2020-21 el.txt switches to German mid-file, as does '▪ Gruppe H'.
    "sechzehntelfinale": "r32", "achtelfinale": "r16", "viertelfinale": "qf",
    "halbfinale": "sf", "finale": "final",
}


def _stage_header(head: str) -> tuple[str | None, int | None]:
    """Map one '▪ ...' header onto (stage, number).

    `number` is a matchday for the league phase and a leg for a two-legged
    round; None where the file does not say.
    """
    parts = [p.strip() for p in head.split(",")]
    first = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    num = None
    for p in parts:
        mm = _MD_IN.match(p)
        if mm:
            num = int(mm.group(1))
    low = first.lower()

    if low in _FINAL_STAGES:
        return _FINAL_STAGES[low], num
    if low in ("finals", "knockout"):
        return _FINAL_STAGES.get(rest.lower(), "final"), num
    if (low in ("league", "group", "regular", "league phase", "group phase")
            or low.startswith(("group ", "gruppe", "regular season",
                               "championship round", "relegation round"))):
        return "league", num
    if low.startswith(("playoff", "play-off", "playout", "qualifying")):
        return "playoff", num
    rm = _ROUND_IN.match(first)
    if rm:
        return "q" + (rm.group(1) or rm.group(2)), num
    return None, num


#: Domestic files carry stage headers too, and until now nothing read them.
#: The Championship's play-off is five knockout matches sitting in the same file
#: as the 552 league ones -- two semi-final legs each way and a Wembley final --
#: and they were being fitted as league fixtures and counted in the league
#: table. Belgium's regular season is followed by split rounds that *are* league
#: matches carrying league points, plus a separate tie for a Conference League
#: place that is not.
#:
#: So the rule is not "a named group means knockout": '▪ Championship, Matchday
#: 3' and '▪ Relegation, Matchday 2' are Belgium's split and count, while
#: '▪ Relegation, Abstieg' is a German relegation tie and does not. What decides
#: it is whether the header names a knockout round.
_KNOCKOUT_WORDS = (
    "final", "finals", "semifinal", "semifinals", "semi-final", "semi-finals",
    "quarterfinal", "quarterfinals", "quarter-final", "quarter-finals",
    "playoff", "playoffs", "play-off", "play-offs", "playout", "playouts",
    # A German file's relegation tie, and the placement bracket some leagues
    # play after the split.
    "abstieg", "platzierung",
    # 'Match for 3rd place'.
    "place",
    # A domestic file that names a UEFA competition is describing the tie for a
    # place in it, not a round of its own league.
    "champions league", "europa league", "conference league", "ecl",
)


def domestic_stage(head: str) -> str:
    """'league' or 'playoff' for one domestic '▪ ...' header.

    An unrecognised header is league, deliberately. Every competition read here
    is a league first, and this classifier also runs over the forty-six smaller
    top flights that only feed the pooled European fit, where a header nobody
    has seen is far more likely to be a spelling of "matchday" than a knockout
    round. Dropping real matches on a guess is the expensive mistake.
    """
    for part in (p.strip().lower() for p in head.split(",")):
        for word in _KNOCKOUT_WORDS:
            if part == word or part.startswith(word + " ") or part.endswith(" " + word):
                return "playoff"
        if part.startswith("round of"):
            return "playoff"
    return "league"


def parse_openfootball(text: str, season: str, reg: TeamRegistry,
                       *, comp: str = "", euro: bool = False) -> list[Match]:
    """openfootball plain-text league files.

    Dates are section headers that carry forward, the year is stated only when it
    changes, and the repository uses three different line layouts across seasons:
    'Home v Away', 'Home v Away  2-1 (1-0)', and 'Home  2-1 (1-0)  Away'. All
    three appear in files this pipeline reads, so all three are handled here.

    `euro=True` additionally reads the stage headers and the '(ENG)' association
    suffixes that only European competition files carry. It is a flag rather
    than a separate reader because the line layout is identical; what differs is
    which extra columns exist.
    """
    start_year = int(season.split("-")[0])
    out: list[Match] = []
    cur_date: date | None = None
    matchday: int | None = None
    stage: str | None = None
    leg: int | None = None
    for raw_line in text.splitlines():
        line = _NOTE.sub("  ", raw_line.rstrip())
        if not line.strip() or line.lstrip().startswith("#") or line.startswith("="):
            continue
        hm = _STAGE_HEAD.match(line)
        if hm:
            if euro:
                stage, num = _stage_header(hm.group(1))
                # A matchday only means something in the league phase; the
                # '▪ Playoffs, Matchday 2' of a two-legged tie is a leg, and
                # letting it overwrite `matchday` is exactly what would make a
                # knockout second leg look like league-phase matchday 2.
                matchday = num if stage == "league" else None
                leg = num if stage not in (None, "league") else None
                continue
            # A domestic header says two things: whether what follows is still
            # the league, and sometimes which round it is. `_MD_RE` below only
            # reads a matchday that starts the header, so '▪ Regular, Matchday
            # 46' was losing its number; take it from the header's own segments
            # instead. A knockout round has no matchday by definition.
            stage = domestic_stage(hm.group(1))
            num = None
            for part in (q.strip() for q in hm.group(1).split(",")):
                mm = _MD_IN.match(part)
                if mm:
                    num = int(mm.group(1))
            matchday = num if stage == "league" else None
            if num is not None or stage != "league":
                continue
        m = _MD_RE.match(line)
        if m:
            # Two alternatives, one number: 'Matchday 3' fills the first group,
            # '3. Round' the second.
            matchday = int(m.group(1) or m.group(2))
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
        aet = False
        ht = None
        sm = _TRAIL_SCORE.search(body)
        if sm:
            hg, ag, aet, ht = _score_from(sm)
            body = body[:sm.start()]

        if " v " in body:
            home, _, away = body.partition(" v ")
        else:
            sm2 = _MID_SCORE.search(body)
            if not sm2:
                continue
            hg, ag, aet, ht = _score_from(sm2)
            home, away = body[:sm2.start()], body[sm2.end():]

        home, away = home.strip(), away.strip()
        h_assoc = a_assoc = None
        if euro:
            am = _ASSOC.search(home)
            if am:
                h_assoc, home = am.group(1), home[:am.start()].strip()
            am = _ASSOC.search(away)
            if am:
                a_assoc, away = am.group(1), away[:am.start()].strip()
        if not _is_club(home) or not _is_club(away):
            continue
        out.append(Match(date=cur_date, home=reg.resolve(home), away=reg.resolve(away),
                         hg=hg, ag=ag,
                         hthg=ht[0] if ht else None, htag=ht[1] if ht else None,
                         matchday=matchday, time=time, season=season,
                         played=hg is not None and ag is not None,
                         stage=stage, leg=leg, home_assoc=h_assoc, away_assoc=a_assoc,
                         comp=comp, aet=aet))
    return out


def parse_openfootball_euro(text: str, season: str, reg: TeamRegistry,
                            comp: str = "cl") -> list[Match]:
    """European competition file: stage headers and '(ENG)' suffixes included."""
    return parse_openfootball(text, season, reg, comp=comp, euro=True)
