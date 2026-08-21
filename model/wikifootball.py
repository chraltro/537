"""Wikipedia, for the twenty-one leagues nothing else publishes.

Gibraltar, San Marino, Andorra, Kosovo, Montenegro, Moldova, Armenia, Malta,
Luxembourg, North Macedonia, Bosnia, Albania, Wales, Northern Ireland, Serbia,
Ukraine, Croatia, Bulgaria, Hungary, Slovakia and Slovenia have no free bulk
results feed anywhere: not on GitHub, not from football-data.co.uk, not from any
publisher that does not want a key and a monthly bill. They do have a Wikipedia
season article, and that article carries a full results grid.

What the grid gives, and what it does not
-----------------------------------------
It gives every score: one cell per ordered pair of clubs, so a 12-club league is
132 results and a 20-club league is 380. That is the whole competition.

It does not give dates. A results matrix says Lincoln beat Europa 2-0; it does
not say when. Every match from this source is therefore dated at the midpoint of
its season and flagged `date_approx`, which is exactly as much as the source
knows. The rating fit weights a match by `exp(-decay * days_old)`, so a season
read this way is one epoch with one weight rather than a run of days -- honest,
and a great deal better than the alternative, which is rating the champions of
Kosovo from a season that ended in 2025.

It also does not give a canonical club name. The grid writes whatever the
article's editor wrote, which is why nothing here is armed by default.

Why nothing is armed
--------------------
`ARMED` is empty. Each league joins it only after a probe on a runner shows that
the article exists, the grid parses, and every club in it resolves to a club
this project already carries. That last condition is the whole difficulty: the
registry knows "Lincoln Red Imps FC" from openfootball's Gibraltar file, and the
article may write "Lincoln Red Imps", "Lincoln" or "Lincoln Red Imps F.C.".
`normalise` folds most of that and an alias fixes the rest, but an alias written
blind is a guess, and a guess here mints a duplicate club.

So the sequence is: probe, read the unresolved names the probe prints, add
aliases for them, probe again, and arm the league when the list is empty. One
league at a time, each one a change that can be pointed at.

Licence: Wikipedia text is CC BY-SA. Scores are facts and not copyrightable, but
the site credits the source on its method page regardless, which is the right
thing to do and costs a sentence.
"""
from __future__ import annotations

import datetime as dt
import re
import urllib.parse

from . import fetch
from .parse import Match, TeamRegistry

API = "https://en.wikipedia.org/w/index.php"

#: Season-article titles, one template per association. The en dash is
#: Wikipedia's, not a typo: '2025–26 Ekstraklasa' is the real title and
#: '2025-26 Ekstraklasa' is a redirect at best.
TITLES: dict[str, str] = {
    "POL": "{a}–{bb} Ekstraklasa",
    "ROU": "{a}–{bb} Liga I",
    "SUI": "{a}–{bb} Swiss Super League",
    "SRB": "{a}–{bb} Serbian SuperLiga",
    "UKR": "{a}–{bb} Ukrainian Premier League",
    "CRO": "{a}–{bb} First Football League (Croatia)",
    "BUL": "{a}–{bb} First Professional Football League (Bulgaria)",
    "SVK": "{a}–{bb} Slovak First Football League",
    "SVN": "{a}–{bb} Slovenian PrvaLiga",
    "HUN": "{a}–{bb} Nemzeti Bajnokság I",
    "BIH": "{a}–{bb} Premier League of Bosnia and Herzegovina",
    "ALB": "{a}–{bb} Kategoria Superiore",
    "ARM": "{a}–{bb} Armenian Premier League",
    "MKD": "{a}–{bb} Macedonian First Football League",
    "MDA": "{a}–{bb} Moldovan Super Liga",
    "KOS": "{a}–{bb} Football Superleague of Kosovo",
    "MNE": "{a}–{bb} Montenegrin First League",
    "MLT": "{a}–{bb} Maltese Premier League",
    "LUX": "{a}–{bb} Luxembourg National Division",
    # Two summer leagues, whose articles are titled by the bare year. The
    # template still takes {bb}; `title` passes an empty string and strips it.
    "NOR": "{a} Eliteserien",
    "BLR": "{a} Belarusian Premier League",
    "GIB": "{a}–{bb} Gibraltar Football League",
    "AND": "{a}–{bb} Primera Divisió",
    "SMR": "{a}–{bb} Campionato Sammarinese di Calcio",
    "WAL": "{a}–{bb} Cymru Premier",
    "NIR": "{a}–{bb} NIFL Premiership",
}

#: Leagues whose grid has been probed on a runner, whose clubs all resolve, and
#: which therefore contribute matches. Empty by design; see the module docstring.
#: Adding a code here without a green probe behind it is the one thing this
#: whole arrangement exists to prevent.
ARMED: frozenset[str] = frozenset()

#: Leagues being watched: probed on every run, reported beside the armed ones,
#: and contributing nothing at all until they move up into ARMED.
#:
#: These five are the plain double round-robins -- every club plays every other
#: exactly twice, no championship split, no third round, no play-off inside the
#: table -- which is the one shape whose remaining fixtures follow from its
#: played ones. That is what makes them forecastable from a results grid with no
#: schedule anywhere: see `model/roundrobin.py`. The other leagues in TITLES can
#: still gain a current season this way, and a rating with it, but not a
#: projected final table.
#:
#: Measured, not assumed: each one's last complete openfootball season has
#: exactly n(n-1) matches and no ordered pair twice -- Norway 16/240, Belarus
#: 16/240, Luxembourg 16/240, Ukraine 16/240, Poland 18/306.
CANDIDATES: frozenset[str] = frozenset({"NOR", "BLR", "LUX", "UKR", "POL"})

#: Extra spellings this source uses for clubs the registry already holds, keyed
#: by association so a fix for Malta cannot collide with one for Moldova. Filled
#: in from what a probe reports, never from guesswork.
ALIASES: dict[str, dict[str, str]] = {}


class GridError(RuntimeError):
    """The article came back and does not contain a readable results grid."""


def title(assoc: str, season: str) -> str:
    """'2025-26' becomes the article's own '2025–26'; a summer league's '2025'
    is passed through, since its articles are titled with the bare year."""
    if "-" in season:
        a, b = season.split("-", 1)
        return TITLES[assoc].format(a=a, bb=b[-2:])
    return TITLES[assoc].format(a=season, bb="").rstrip("– ")


def url(assoc: str, season: str) -> str:
    q = urllib.parse.urlencode({"title": title(assoc, season), "action": "raw"})
    return f"{API}?{q}"


def season_midpoint(season: str) -> dt.date:
    """One date for a whole season, because the grid carries no other.

    A winter season labelled 2025-26 turns the year at its middle, so 1 January
    of the later year is both the arithmetic midpoint and the obvious one. A
    summer season labelled 2025 runs roughly April to October, so 1 July.
    """
    if "-" in season:
        return dt.date(int(season.split("-")[0]) + 1, 1, 1)
    return dt.date(int(season), 7, 1)


_MATCH = re.compile(r"\|\s*match_([A-Za-z0-9_]+)_([A-Za-z0-9_]+)\s*=\s*"
                    r"(\d+)\s*[–—-]\s*(\d+)")
#: A club's display name. The value is a wiki link about half the time --
#: `|name_BRE=[[FC Dynamo Brest|Dynamo Brest]]` -- and stopping at the first
#: pipe reads that as "[[FC Dynamo Brest", which resolves to nothing and blocked
#: five leagues on the first probe. So a link is matched whole and taken apart
#: below; anything else stops at the pipe as before.
_NAME = re.compile(r"\|\s*name_([A-Za-z0-9_]+)\s*=\s*(\[\[[^\]\n]*\]\]|[^\n|}]+)")
_TEAM = re.compile(r"\|\s*team\d+\s*=\s*([A-Za-z0-9_]+)")


def name_variants(raw: str) -> tuple[str, ...]:
    """The spellings one `name_X=` value offers, most readable first.

    `[[FC Dynamo Brest|Dynamo Brest]]` is two names for one club: the article's
    title and what the table prints. Both are worth having, because which one
    the registry knows depends on what openfootball wrote, and the two feeds
    disagree in both directions -- Wikipedia's title is the formal one for
    Norwegian clubs ("Vålerenga Fotball") and the short one for Ukrainian
    ("Obolon Kyiv"). Trying each in turn is not a guess: the link says they are
    the same club.
    """
    v = " ".join(raw.split()).strip()
    m = re.match(r"^\[\[([^\]|]+)(?:\|([^\]]*))?\]\]$", v)
    if not m:
        return (v,) if v else ()
    target = m.group(1).strip()
    shown = (m.group(2) or "").strip()
    return tuple(dict.fromkeys(x for x in (shown, target) if x))


def grid_names(text: str) -> dict[str, tuple[str, ...]]:
    """Every club code in the grid, with the spellings its name line offers."""
    out: dict[str, tuple[str, ...]] = {}
    for code, val in _NAME.findall(text):
        got = name_variants(val)
        if got:
            out[code] = got
    return out


def parse_grid(text: str) -> list[tuple[str, str, int, int]]:
    """Every `match_HOME_AWAY=H–A` cell, with codes expanded to club names.

    `Module:Sports results` writes a short code per club and a `name_CODE=`
    line giving its display name. Some articles omit the names and use the
    club's short name as the code itself, which is why a missing name falls
    back to the code rather than dropping the row.
    """
    variants = grid_names(text)
    names = {code: v[0] for code, v in variants.items()}
    codes = set(_TEAM.findall(text)) | set(names)
    cells = _MATCH.findall(text)
    if not cells:
        raise GridError("no match_X_Y cells found")

    out: list[tuple[str, str, int, int]] = []
    for home, away, hg, ag in cells:
        if home == away:
            continue                       # a diagonal cell, occasionally filled
        if codes and (home not in codes or away not in codes):
            continue                       # a stray cell from another grid
        out.append((names.get(home, home), names.get(away, away), int(hg), int(ag)))
    if not out:
        raise GridError(f"{len(cells)} cells found, none with two known clubs")
    return out


def grid_clubs(text: str) -> list[str]:
    """Every club in the grid, including the ones that have not played yet.

    `parse_grid` can only see a club that appears in a filled cell, which in the
    first week of a season is not all of them and in the first day is none of
    them. The template lists its entrants separately -- `team1=`, `team2=` --
    and that list is what a remaining-fixture derivation has to start from,
    because a club missing from it loses every one of its fixtures silently.
    """
    names = {code: v[0] for code, v in grid_names(text).items()}
    order = _TEAM.findall(text)
    seen, out = set(), []
    for code in order or sorted(names):
        if code in seen:
            continue
        seen.add(code)
        out.append(names.get(code, code))
    return out


def load(assoc: str, reg: TeamRegistry,
         seasons: tuple[str, ...] = ("2025-26",)) -> list[tuple[Match, str, str]]:
    """Played matches for one association, as `(match, home_name, away_name)`.

    Aliases are applied to the raw names here rather than in the registry, so an
    alias added for one league's article cannot change how any other source
    reads the same string.
    """
    alias = ALIASES.get(assoc, {})
    out: list[tuple[Match, str, str]] = []
    reached = 0
    for season in seasons:
        text = fetch.get(url(assoc, season), required=False, tries=2)
        if not text or "match_" not in text:
            continue
        reached += 1
        when = season_midpoint(season)
        # Which of a club's spellings to hand over: an alias if one is written
        # for it, otherwise whichever the registry already knows, otherwise the
        # one the table prints -- which is then what the probe reports as
        # unresolved, and what an alias gets written for.
        best: dict[str, str] = {}
        for names in grid_names(text).values():
            for name in names:
                if name in alias:
                    best[names[0]] = alias[name]
                    break
                if reg.known(name):
                    best[names[0]] = name
                    break
        for home, away, hg, ag in parse_grid(text):
            h = best.get(home, alias.get(home, home))
            a = best.get(away, alias.get(away, away))
            m = Match(date=when, home=h, away=a, hg=hg, ag=ag,
                      season=season, played=True,
                      extra={"date_approx": True, "source": "wikipedia"})
            out.append((m, h, a))
    if not reached:
        raise GridError(f"no article reachable for {assoc} in {list(seasons)}")
    return out


def seasons_for(have: tuple[str, ...],
                through: str = "2026-27") -> tuple[str, ...]:
    """The seasons to ask Wikipedia for: the newest one the GitHub feed carries,
    then every season after it up to `through`.

    That first season is not wanted for its results, which we already have. It
    is wanted because `model.external.probe` refuses a source it cannot line up
    against a season it already knows, and this is the season it lines it up on.
    """
    if not have:
        return ()
    newest = max(have)
    if "-" not in newest:                          # a summer league, labelled by year
        start, stop = int(newest), int(through.split("-")[0])
        return tuple(str(y) for y in range(start, stop + 1))
    start, stop = int(newest.split("-")[0]), int(through.split("-")[0])
    return tuple(f"{y}-{str(y + 1)[-2:]}" for y in range(start, stop + 1))


def source(assoc: str, league: str, group: str,
           seasons: tuple[str, ...] = ("2025-26",), *, contributes: bool = True):
    """An `ExternalSource` for one association, ready to be probed."""
    from .external import ExternalSource
    return ExternalSource(
        source="wikipedia", assoc=assoc, league=league, group=group,
        load=lambda reg, a=assoc, s=seasons: load(a, reg, s),
        contributes=contributes,
        note="Results grid only; every match is dated at its season's midpoint.")
