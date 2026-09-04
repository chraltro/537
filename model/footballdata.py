"""football-data.co.uk, for the three stale leagues a bulk feed still reaches.

The site already reads this publisher's data: `datasets/football-datasets` is a
GitHub mirror of its big-five files, which is where the shots, cards, corners
and half-time columns come from. What that mirror does not carry, and what no
GitHub repository carries, is the publisher's *extra* set: a second group of
files covering leagues outside the main run, of which Poland, Romania and
Switzerland are ones this project has been rating from 2024-25 data.

The extra files are a different schema from the main ones, and thinner:

    Country,League,Season,Date,Time,Home,Away,HG,AG,Res,PH,PD,PA,...

Goals and closing odds, no shots and no cards. So these three leagues gain a
current season and keep the same feature set they had; they do not join the
five that carry a shot count. The odds columns are read by nothing here. This
project does not publish betting value and has no use for a bookmaker's price.

Everything below is written from the published format and has never been
fetched by the process that wrote it: the development sandbox denies every host
except GitHub. `model.external.probe` is what turns that into a fact rather
than a hope, on a runner that can actually reach the file.
"""
from __future__ import annotations

import csv
import datetime as dt
import io

from . import fetch
from .parse import Match, TeamRegistry

BASE = "https://www.football-data.co.uk/new"

#: The file each association lives in. Named by the publisher, not by ISO, which
#: is why Switzerland is SWZ here and SUI everywhere else in this project.
FILES: dict[str, str] = {"POL": "POL", "ROU": "ROU", "SUI": "SWZ"}

#: Column spellings this reader accepts, most-likely first. The extra files use
#: the short forms; the main files use the long ones. Accepting both costs
#: nothing and means the reader survives the publisher unifying its schemas.
_HOME = ("Home", "HomeTeam")
_AWAY = ("Away", "AwayTeam")
_HG = ("HG", "FTHG")
_AG = ("AG", "FTAG")


#: The spellings this publisher uses for clubs the GitHub feed spells another
#: way, keyed by association so a fix for Poland cannot reach into Romania.
#:
#: These are not decoration. `model.external.probe` refuses a league outright
#: when a club in the overlap season does not resolve, because that season is
#: one we already hold in full: a name that fails there is a second spelling of
#: a club already in the ranking, and minting an id for it would put the same
#: club in twice with half a record each. On 2026-08-21 the runner refused all
#: three leagues for exactly this reason -- seven names in Poland, seven in
#: Romania, three in Switzerland -- and every one of them is below.
#:
#: Accents are not the problem; `normalise` folds those, which is why "Zurich"
#: finds "FC Zürich" and "Slask Wroclaw" finds "Śląsk Wrocław" with nothing
#: written here. The problem is the publisher dropping the town: "Rakow" for
#: "Raków Częstochowa", "Legia" for "Legia Warszawa", "U. Cluj" for
#: "Universitatea Cluj". Only names that are unambiguous in their own league are
#: written down -- "Lausanne" is Lausanne-Sport and nothing else, while a bare
#: "Zaglebie" could be Lubin or Sosnowiec and is therefore left to fail loudly.
ALIASES: dict[str, dict[str, str]] = {
    "POL": {
        "Cracovia": "KS Cracovia",
        "Jagiellonia": "Jagiellonia Białystok",
        "Legia": "Legia Warszawa",
        "Puszcza": "Puszcza Niepołomice",
        "Rakow": "Raków Częstochowa",
        "Stal Mielec": "FKS Stal Mielec",
        "Lech": "Lech Poznan",
        "Lechia": "Lechia Gdańsk",
        "Widzew": "Widzew Łódź",
        "Piast": "Piast Gliwice",
        "Korona": "Korona Kielce",
        "Radomiak": "Radomiak Radom",
        "Slask": "Śląsk Wrocław",
        # Left out on the first pass as a guess -- Poland has a Zagłębie Lubin
        # and a Zagłębie Sosnowiec -- and then written down when the runner
        # named it as the last thing blocking the league. It is not a guess now:
        # the overlap season is 2024-25, we hold that season in full, and the
        # only Zagłębie in it was Lubin. The collapse check in
        # `model.external.probe` is what stops this quietly misfiling Sosnowiec
        # if the publisher ever uses the bare name for the other one.
        "Zaglebie": "Zagłębie Lubin",
        "Motor": "Motor Lublin",
    },
    "ROU": {
        "Din. Bucuresti": "Dinamo Bucureşti",
        "Otelul": "Oţelul Galaţi",
        "Petrolul": "Petrolul Ploieşti",
        "Poli Iasi": "FC Politehnica Iași",
        "Sepsi Sf. Gheorghe": "Sepsi OSK",
        "U. Cluj": "Universitatea Cluj",
        "Univ. Craiova": "CS Universitatea Craiova",
        "Sepsi": "Sepsi OSK",
        "Farul": "Farul Constanța",
        "Rapid": "Rapid Bucureşti",
    },
    "SUI": {
        "Grasshoppers": "Grasshopper Club Zürich",
        "Lausanne": "FC Lausanne-Sport",
        "Yverdon": "Yverdon Sport FC",
    },
}


#: Spellings for clubs that arrived after the GitHub feed stopped publishing,
#: which is why they are here and not in ALIASES: the test that checks an alias
#: target against the openfootball season it came from cannot check these,
#: because the club was never in one.
#:
#: They matter where two verified feeds cover one league and disagree about a
#: new club's name. Poland is the case: the results come from this publisher,
#: which writes "Wisla" for the club promoted this summer, and the season's
#: entrant list comes from the Wikipedia grid, which writes "Wisła Kraków". Two
#: ids for one club, and the projection refused the league rather than run a
#: season with a club in it twice and another missing.
#:
#: A wrong entry here is not silent. It either makes a club play itself, which
#: the probe refuses, or leaves the league a club short against last season's
#: count, which the projection refuses and prints both lists for.
CROSS_FEED: dict[str, dict[str, str]] = {
    "POL": {
        # Established by the runner on 2026-08-21: the grid's eighteen entrants
        # matched this publisher's on every club but this one.
        "Wisla": "Wisła Kraków",
        "Termalica B-B.": "Bruk-Bet Termalica Nieciecza",
    },
}


class FormatError(RuntimeError):
    """The file came back but is not the file we were promised. Raised rather
    than worked around: a reader that guesses at an unexpected header is how a
    feed silently starts producing nonsense."""


def url(assoc: str) -> str:
    return f"{BASE}/{FILES[assoc]}.csv"


def _pick(row: dict, names: tuple[str, ...]) -> str | None:
    for n in names:
        if row.get(n) not in (None, ""):
            return row[n]
    return None


def _date(raw: str) -> dt.date | None:
    """The publisher writes dd/mm/yyyy, and has written dd/mm/yy in older files.
    Both are read; anything else is left for the caller to count as a skip."""
    raw = (raw or "").strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _season(raw: str) -> str:
    """'2025/2026' and '2025/26' both become '2025-26'; a summer league's bare
    '2025' stays '2025', which is the spelling openfootball uses for it too."""
    raw = (raw or "").strip().replace(" ", "")
    if "/" in raw:
        a, b = raw.split("/", 1)
        return f"{a}-{b[-2:]}"
    return raw


def load(assoc: str, reg: TeamRegistry,
         since: str = "2018-19") -> list[tuple[Match, str, str]]:
    """Played matches for one association, newest schema or oldest.

    Returns `(match, home_name, away_name)` with the raw spellings still
    attached, because `model.external` resolves and gates them rather than
    letting a reader mint club ids of its own.
    """
    # One attempt. This is a second feed: a host that is blocked or gone
    # answers immediately and identically, and the retry backoff spends
    # three seconds per URL to hear it twice. The GitHub sources keep the
    # full four tries, whose failures really are transient.
    text = fetch.get(url(assoc), required=False, tries=1)
    if text is None:
        raise FormatError(f"unreachable: {url(assoc)}")

    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise FormatError(f"{url(assoc)} parsed to zero rows")
    head = rows[0]
    for label, names in (("home", _HOME), ("away", _AWAY),
                         ("home goals", _HG), ("away goals", _AG)):
        if not any(n in head for n in names):
            raise FormatError(
                f"{url(assoc)} has no {label} column; saw {sorted(head)[:12]}")

    alias = {**ALIASES.get(assoc, {}), **CROSS_FEED.get(assoc, {})}
    out: list[tuple[Match, str, str]] = []
    for row in rows:
        home, away = _pick(row, _HOME), _pick(row, _AWAY)
        home, away = alias.get(home, home), alias.get(away, away)
        hg, ag = _pick(row, _HG), _pick(row, _AG)
        when = _date(row.get("Date", ""))
        if not (home and away and when) or hg is None or ag is None:
            continue                      # a fixture not yet played, or a blank row
        season = _season(row.get("Season", ""))
        if season and season < since:
            continue
        try:
            m = Match(date=when, home=home, away=away,
                      hg=int(float(hg)), ag=int(float(ag)),
                      season=season, played=True,
                      time=(row.get("Time") or "").strip() or None)
        except ValueError:
            continue                      # a goal column that is not a number
        out.append((m, home, away))
    if not out:
        raise FormatError(f"{url(assoc)} has {len(rows)} rows and no played matches")
    return out


def source(assoc: str, league: str, group: str):
    """An `ExternalSource` for one association, ready to be probed."""
    from .external import ExternalSource
    return ExternalSource(
        source="football-data.co.uk", assoc=assoc, league=league, group=group,
        load=lambda reg, a=assoc: load(a, reg),
        note="Goals only; this publisher's extra files carry no shot or card columns.")
