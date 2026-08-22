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
TITLES: dict[str, tuple[str, ...]] = {
    "POL": ("{a}–{bb} Ekstraklasa",),
    "ROU": ("{a}–{bb} Liga I",),
    "SUI": ("{a}–{bb} Swiss Super League",),
    "SRB": ("{a}–{bb} Serbian SuperLiga",),
    "UKR": ("{a}–{bb} Ukrainian Premier League",),
    "CRO": ("{a}–{bb} First Football League (Croatia)",),
    "BUL": ("{a}–{bb} First Professional Football League (Bulgaria)",),
    "SVK": ("{a}–{bb} Slovak First Football League",),
    "SVN": ("{a}–{bb} Slovenian PrvaLiga",),
    "HUN": ("{a}–{bb} Nemzeti Bajnokság I",),
    "BIH": ("{a}–{bb} Premier League of Bosnia and Herzegovina",),
    "ALB": ("{a}–{bb} Kategoria Superiore",),
    "ARM": ("{a}–{bb} Armenian Premier League",),
    "MKD": ("{a}–{bb} Macedonian First Football League",),
    "MDA": ("{a}–{bb} Moldovan Super Liga",),
    "KOS": ("{a}–{bb} Football Superleague of Kosovo",),
    "MNE": ("{a}–{bb} Montenegrin First League",),
    "MLT": ("{a}–{bb} Maltese Premier League",),
    "LUX": ("{a}–{bb} Luxembourg National Division",),
    # Two summer leagues, whose articles are titled by the bare year. The
    # template still takes {bb}; `title` passes an empty string and strips it.
    "NOR": ("{a} Eliteserien",),
    "BLR": ("{a} Belarusian Premier League",),
    "GIB": ("{a}–{bb} Gibraltar Football League",),
    "AND": ("{a}–{bb} Primera Divisió",),
    "SMR": ("{a}–{bb} Campionato Sammarinese di Calcio",),
    "WAL": ("{a}–{bb} Cymru Premier",),
    "NIR": ("{a}–{bb} NIFL Premiership",),
    # The rest of the associations whose GitHub feed has no current season.
    # Winter leagues first, then the summer ones, whose articles are titled
    # with the bare year. Where a league has been renamed or its article
    # disambiguated, every spelling worth trying is here and the first that
    # comes back with a grid wins; a title that is simply wrong 404s and the
    # league reports as unreachable, which costs nothing but a round trip.
    "CYP": ("{a}–{bb} Cypriot First Division",),
    "CZE": ("{a}–{bb} Czech First League",),
    "DEN": ("{a}–{bb} Danish Superliga",),
    "AZE": ("{a}–{bb} Azerbaijan Premier League",),
    "SCO": ("{a}–{bb} Scottish Premiership",),
    "GRE": ("{a}–{bb} Super League Greece",),
    "TUR": ("{a}–{bb} Süper Lig",),
    "AUT": ("{a}–{bb} Austrian Football Bundesliga",),
    # Lithuania's A Lyga became the TOPLYGA in 2026, so the overlap season and
    # the season being read are under different names.
    "LTU": ("{a} TOPLYGA", "{a} A Lyga"),
    "EST": ("{a} Meistriliiga",),
    "GEO": ("{a} Erovnuli Liga",),
    "SWE": ("{a} Allsvenskan",),
    "FIN": ("{a} Veikkausliiga",),
    "ISL": ("{a} Besta deild karla", "{a} Úrvalsdeild karla"),
    "IRL": ("{a} League of Ireland Premier Division",),
    "LVA": ("{a} Latvian Higher League", "{a} Virsliga"),
    "FRO": ("{a} Betri deildin", "{a} Faroe Islands Premier League"),
}

#: Leagues whose grid has been probed on a runner, whose clubs all resolve, and
#: which therefore contribute matches. Adding a code here without a green probe
#: behind it is the one thing this whole arrangement exists to prevent, so each
#: one carries the run that put it here:
#:
#:   NOR  238/238 of 2024,     608 matches, 2 new clubs
#:   BLR  240/240 of 2024,     464 matches, 3 new clubs
#:   LUX  240/240 of 2024-25,  504 matches, 7 new clubs
#:   UKR  237/238 of 2024-25,  498 matches, 5 new clubs
#:
#: Ukraine is the only one not unanimous, and one disagreement in 238 is what
#: the 95% floor is there to absorb: feeds differ over awarded results, and a
#: forfeit recorded as 3-0 by one and 0-0 by the other is not two different
#: competitions.
ARMED: frozenset[str] = frozenset({"NOR", "LUX", "BLR", "UKR"})

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
#: Everything with an article title that is not already in service. A candidate
#: is fetched, parsed and judged exactly like an armed feed and then contributes
#: nothing, so the cost of listing one that turns out not to work is a line in
#: the build log. The cost of not listing it is a league whose ratings stay
#: frozen at the last season its GitHub feed published, which for most of these
#: was May 2025.
#:
#: Poland stays on the list rather than being armed: its results come from
#: football-data.co.uk, which carries dates, and the grid is read only for the
#: season's entrant list.
CANDIDATES: frozenset[str] = frozenset(TITLES) - frozenset({"ROU", "SUI"})

#: Names the twin check flags as looking like a club we already hold, which are
#: genuinely their own club.
#:
#: The check is deliberately blunt, because minting a second id for a club that
#: is already in the ranking is the one mistake here that cannot be seen from
#: the outside: the club appears twice, each copy holding half a record and
#: neither one right. So it stops on a resemblance and asks for a decision, and
#: this is where the decision is written down, with what settled it.
#:
#: UKR/Poltava, settled 2026-08-21. Ukraine has two clubs in the same city and
#: the names alone cannot separate them. Vorskla Poltava went down at the end of
#: 2024-25 after eighteen seasons up, and SC Poltava is a different club, which
#: finished sixteenth in 2025-26 and went down in its turn. So the 'Poltava' in
#: the grid is not a short form of the club we already hold; the two are never
#: in the league in the same season, which is why the same-season rule below
#: could not settle it either.
#:
#: This one is not from a page that could be fetched and checked -- the sandbox
#: reaches GitHub and nothing else, and the runner does not carry the question
#: back -- so it rests on two independent searches agreeing on the specifics.
#: Weaker evidence than the rest of this file runs on, and recorded as such.
#: What limits the damage if it is wrong is that it is one club in one league:
#: the ranking would carry an SC Poltava that is really Vorskla, and the
#: projected table would show a club with a season's results under a name that
#: belongs to another. Nothing else moves.
DISTINCT: dict[str, frozenset[str]] = {"UKR": frozenset({"Poltava"})}

#: Leagues whose grid drives a projected final table. A separate set from ARMED
#: because the two answer different questions. ARMED is "does this feed's
#: results go into the pooled corpus", and only one feed per league may say yes
#: or the same matches are counted twice. PROJECTED is "is this grid good enough
#: to take the season's entrant list and the matches left from", which is a
#: thing a grid can be even when its results are not the ones being used.
#:
#: Poland is exactly that case: football-data.co.uk carries its results with
#: dates on them and is the feed in service, while the Wikipedia article is
#: where the list of who is in the league this season comes from.
PROJECTED: frozenset[str] = frozenset({"NOR", "LUX", "BLR", "POL", "UKR"})

#: Extra spellings this source uses for clubs the registry already holds, keyed
#: by association so a fix for Malta cannot collide with one for Moldova. Filled
#: in from what a probe reports, never from guesswork.
#:
#: Every entry below is a name a runner refused, matched against the club list
#: openfootball published for the same season, and every target is checked by a
#: test against that feed. Two kinds of difference turn up and neither is one
#: `normalise` can fold. The article drops a prefix the feed keeps -- Wikipedia
#: writes "Gomel" where openfootball writes "FK Gomel", and FK is not in the
#: dropped-token list because dropping it would merge clubs elsewhere. Or the
#: two transliterate the same Cyrillic differently: Dynamo and Dinamo, Zorya
#: Luhansk and Zorya Lugansk, Chornomorets Odesa and Chernomorets Odessa.
ALIASES: dict[str, dict[str, str]] = {
    "NOR": {
        "KFUM": "KFUM Oslo",
        "KFUM-Kameratene Oslo": "KFUM Oslo",
        "Sarpsborg": "Sarpsborg 08",
        "Sarpsborg 08 FF": "Sarpsborg 08",
        "Strømsgodset": "Strømsgodset IF",
        "Strømsgodset Toppfotball": "Strømsgodset IF",
    },
    "BLR": {
        "Dynamo Brest": "Dinamo Brest",
        "FC Dynamo Brest": "Dinamo Brest",
        "Dynamo Minsk": "Dinamo Minsk",
        "Gomel": "FK Gomel",
        "Isloch Minsk Raion": "FK Isloch Minsk",
        "Minsk": "FK Minsk",
        "Slavia Mozyr": "FK Slaviya Mozyr",
        "Slutsk": "FK Slutsk",
        "Shakhtyor Soligorsk": "Shakhter Soligorsk",
        "Smorgon": "FK Smorgon",
        "Vitebsk": "FK Vitebsk",
    },
    "LUX": {
        "Mondorf-les-Bains": "US Mondorf",
        "US Mondorf-les-Bains": "US Mondorf",
        "Progrès Niederkorn": "Progrès Niedercorn",
        "FC Progrès Niederkorn": "Progrès Niedercorn",
        "Racing Union": "RFCU Luxemburg",
        "Racing FC Union Luxembourg": "RFCU Luxemburg",
    },
    "UKR": {
        # The 2025-26 and 2026-27 articles drop the city the 2024-25 one keeps,
        # so each of these is here twice.
        "Chornomorets": "Chernomorets Odessa",
        "Obolon": "FK Obolon",
        "Veres": "NK Veres",
        "Zorya": "Zorya Lugansk",
        "Chornomorets Odesa": "Chernomorets Odessa",
        "Livyi Bereh Kyiv": "Livyi Bereh",
        "Obolon Kyiv": "FK Obolon",
        "Oleksandriya": "FK Oleksandriya",
        "Veres Rivne": "NK Veres",
        "Zorya Luhansk": "Zorya Lugansk",
    },
    "POL": {
        "Stal Mielec": "FKS Stal Mielec",
    },
}


class GridError(RuntimeError):
    """The article came back and does not contain a readable results grid."""


def titles(assoc: str, season: str) -> list[str]:
    """Every article title this league's season might be under, best first.

    '2025-26' becomes the article's own '2025–26'; a summer league's '2025' is
    passed through, since its articles are titled with the bare year.

    A list and not a name, because leagues get renamed and the article follows
    them: Lithuania's A Lyga became the TOPLYGA in 2026, so the season this site
    needs for its overlap check is under one name and the season it wants the
    results from is under another. Sponsors move these names about constantly
    and no single spelling survives them. Each is tried in turn and the first
    that comes back with a grid in it wins.
    """
    out = []
    for tpl in TITLES[assoc]:
        if "-" in season:
            a, b = season.split("-", 1)
            out.append(tpl.format(a=a, bb=b[-2:]))
        else:
            out.append(tpl.format(a=season, bb="").rstrip("– "))
    return out


def title(assoc: str, season: str) -> str:
    """The likeliest title, for a message or a link."""
    return titles(assoc, season)[0]


def url(assoc: str, season: str, which: int = 0) -> str:
    q = urllib.parse.urlencode({"title": titles(assoc, season)[which],
                                "action": "raw"})
    return f"{API}?{q}"


def season_midpoint(season: str, today: dt.date | None = None) -> dt.date:
    """One date for a whole season, because the grid carries no other.

    A winter season labelled 2025-26 turns the year at its middle, so 1 January
    of the later year is both the arithmetic midpoint and the obvious one. A
    summer season labelled 2025 runs roughly April to October, so 1 July.

    Never later than yesterday, which matters for the season being played right
    now: in August the midpoint of 2026-27 is next January, and a match dated
    next January is a match the rating fit cannot see, so a club promoted this
    summer had no matches at all, no rating, and no place in its own league's
    projection. Yesterday and not today because the fit's cutoff is exclusive --
    `corpus.before(ref)` keeps what happened strictly before it, and with the
    reference date being today, a match dated today is still invisible.
    """
    mid = (dt.date(int(season.split("-")[0]) + 1, 1, 1) if "-" in season
           else dt.date(int(season), 7, 1))
    return min(mid, (today or dt.date.today()) - dt.timedelta(days=1))


#: A club's short code, as the template writes it. Not `[A-Za-z0-9_]`: the codes
#: are the article editor's own abbreviations and they are not all ASCII, so
#: that class silently dropped a club from Norway's 2025 grid and three from
#: Poland's 2026-27 one. A code with an accented letter in it was matched by
#: neither the entrant list nor the name lines, its cells were then filtered out
#: as belonging to some other grid, and the league quietly ran a club short.
_CODE = r"[^\s=|}\]]+"

_MATCH = re.compile(r"\|\s*match_(" + _CODE + r")_(" + _CODE + r")\s*=\s*"
                    r"(\d+)\s*[–—-]\s*(\d+)")
#: A club's display name. The value is a wiki link about half the time --
#: `|name_BRE=[[FC Dynamo Brest|Dynamo Brest]]` -- and stopping at the first
#: pipe reads that as "[[FC Dynamo Brest", which resolves to nothing and blocked
#: five leagues on the first probe. So a link is matched whole and taken apart
#: below; anything else stops at the pipe as before.
_NAME = re.compile(r"\|\s*name_(" + _CODE + r")\s*=\s*"
                   r"(\[\[[^\]\n]*\]\]|[^\n|}]+)")
_TEAM = re.compile(r"\|\s*team\d+\s*=\s*(" + _CODE + r")")


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
    order = [c for c in _TEAM.findall(text) if c in names]
    # A `teamN=` code with no `name_` line beside it is not a club this article
    # is naming, it is a code from another template on the same page: Poland's
    # season article carries a league table whose entrant list uses its own
    # short codes, and reading those as clubs produced two entrants called "G"
    # and "WP". Only a code the grid names is an entrant.
    seen, out = set(), []
    for code in order or sorted(names):
        if code in seen:
            continue
        seen.add(code)
        out.append(names[code])
    return out


def _chosen(text: str, reg: TeamRegistry, alias: dict[str, str]) -> dict[str, str]:
    """Which of each club's spellings to hand over.

    An alias if one is written for it, otherwise whichever spelling the registry
    already knows, otherwise the one the table prints -- which is then what the
    probe reports as unresolved, and what an alias gets written for. Keyed by
    that printed name, since that is what `parse_grid` returns.
    """
    out: dict[str, str] = {}
    for names in grid_names(text).values():
        for name in names:
            if name in alias:
                out[names[0]] = alias[name]
                break
            if reg.known(name):
                out[names[0]] = name
                break
    return out


def read(assoc: str, reg: TeamRegistry, season: str):
    """One season's article: its entrants and its results, spellings resolved.

    Returns `(clubs, [(home, away, hg, ag)])`, or None if the article is not
    there. The club list is the template's own `team1=`..`teamN=` and not the
    clubs that happen to appear in a filled cell, because a season two weeks old
    has clubs with no result yet and they still have a full fixture list.
    """
    text = None
    for i in range(len(TITLES[assoc])):
        got = fetch.get(url(assoc, season, i), required=False, tries=2)
        if got and "match_" in got:
            text = got
            break
    if text is None:
        return None
    alias = ALIASES.get(assoc, {})
    best = _chosen(text, reg, alias)
    pick = lambda n: best.get(n, alias.get(n, n))          # noqa: E731
    clubs = [pick(n) for n in grid_clubs(text)]
    rows = [(pick(h), pick(a), hg, ag) for h, a, hg, ag in parse_grid(text)]
    return clubs, rows


def load(assoc: str, reg: TeamRegistry,
         seasons: tuple[str, ...] = ("2025-26",)) -> list[tuple[Match, str, str]]:
    """Played matches for one association, as `(match, home_name, away_name)`.

    Aliases are applied to the raw names here rather than in the registry, so an
    alias added for one league's article cannot change how any other source
    reads the same string.
    """
    out: list[tuple[Match, str, str]] = []
    reached = 0
    for season in seasons:
        got = read(assoc, reg, season)
        if got is None:
            continue
        reached += 1
        when = season_midpoint(season)
        for home, away, hg, ag in got[1]:
            m = Match(date=when, home=home, away=away, hg=hg, ag=ag,
                      season=season, played=True,
                      extra={"date_approx": True, "source": "wikipedia"})
            out.append((m, home, away))
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
        contributes=contributes, distinct=DISTINCT.get(assoc, frozenset()),
        note="Results grid only; every match is dated at its season's midpoint.")
