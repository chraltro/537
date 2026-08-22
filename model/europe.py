"""The European corpus: UEFA competition results, and the domestic leagues that
feed them.

Two jobs, both of them supply problems rather than modelling ones.

*The bridge.* `openfootball/champions-league` carries every season of UEFA
club competition, and those matches are the only edges in the graph that connect
one domestic league to another. Every rating comparison across borders -- is
Bodø/Glimt closer to Brentford or to Burnley -- rests on them, so they are loaded
in full rather than sampled.

*The base.* Twenty-nine of the participating associations have no domestic feed
newer than May 2025 (docs/european-competitions-plan.md 1.5). They are loaded
anyway, because a stale league is still the only evidence that separates the
champions of Croatia from the eighth-placed club, and the staleness is handled
where it belongs -- as widened uncertainty in the simulation, not as a silently
confident rating.

Nothing here writes files. `fetch.get` caches every URL on disk exactly as the
domestic pipeline does, so a warm run costs no network at all.
"""
from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass

from . import config, fetch, leagues
from .parse import Match, TeamRegistry, parse_openfootball, parse_openfootball_euro

OF = "https://raw.githubusercontent.com/openfootball"

# --------------------------------------------------------------------------
# UEFA competitions
# --------------------------------------------------------------------------
#: Which competition files each season of `openfootball/champions-league`
#: actually contains, listed rather than probed: a 404 costs a round trip and
#: the set has been stable for years. A season missing from this table is still
#: reachable via `load_competitions(seasons=[...])`, which probes.
#:
#: The gaps are the story of the feed, not of the competitions: the 2025-26
#: Europa and Conference League phases were played in full and never committed.
EURO_FILES: dict[str, tuple[str, ...]] = {
    "2011-12": ("cl",), "2012-13": ("cl",), "2013-14": ("cl",),
    "2014-15": ("cl",), "2015-16": ("cl",), "2016-17": ("cl",),
    "2017-18": ("cl",), "2018-19": ("cl",), "2019-20": ("cl",),
    "2020-21": ("cl", "el"),
    "2021-22": ("cl", "el", "conf"),
    "2022-23": ("cl", "el", "conf"),
    "2023-24": ("cl", "el", "conf"),
    "2024-25": ("cl", "el", "conf", "clq", "elq", "confq"),
    "2025-26": ("cl", "clq", "elq", "confq"),
}

#: The competitions whose matches are European rather than domestic. Used for
#: the per-competition home-advantage term: European home advantage is measurably
#: about half again the Premier League's, which is too big to average away.
EURO_COMPS = frozenset({"cl", "el", "conf", "clq", "elq", "confq"})

#: The one group id every European match shares.
EUROPE = "europe"


def euro_url(season: str, comp: str) -> str:
    return f"{OF}/champions-league/master/{season}/{comp}.txt"


def load_competitions(reg: TeamRegistry, seasons: list[str] | None = None,
                      comps: tuple[str, ...] | None = None,
                      *, quiet: bool = False) -> list[Match]:
    """Every played UEFA match we can reach, as `Match` objects.

    Club ids come out of the same `TeamRegistry` the domestic feeds use, so
    'Arsenal FC (ENG)' here and 'Arsenal' in the Premier League CSV are one club
    and the bridge actually connects.
    """
    out: list[Match] = []
    todo = seasons or list(EURO_FILES)
    for season in todo:
        want = comps or EURO_FILES.get(season) or ("cl", "el", "conf",
                                                   "clq", "elq", "confq")
        for comp in want:
            text = fetch.get(euro_url(season, comp), required=False)
            if not text:
                continue
            got = parse_openfootball_euro(text, season, reg, comp)
            out.extend(m for m in got if m.played)
    if not quiet:
        print(f"  · {len(out)} European matches over "
              f"{len({m.season for m in out})} seasons")
    return out


def league_phase(matches: list[Match], season: str, comp: str = "cl") -> list[Match]:
    """The 36-team single-table part of one season, in matchday order."""
    return sorted((m for m in matches
                   if m.season == season and m.comp == comp and m.stage == "league"),
                  key=lambda m: (m.matchday or 0, m.date, m.home))


# --------------------------------------------------------------------------
# Domestic leagues outside the big five
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DomesticSource:
    """One non-big-five top flight and the seasons upstream actually has.

    `seasons` is listed rather than derived because the repositories disagree
    about which years they carry, and probing 48 countries for absent seasons
    costs 200 round trips to learn what a table already knows.
    """

    assoc: str                  # UEFA three-letter code, as European files spell it
    name: str
    url_tpl: str                # formatted with {season}
    seasons: tuple[str, ...]
    #: The country in words. The ranking page prints this in a column beside
    #: England, Spain and Germany, and it read "POL", "TUR", "SUI" for
    #: forty-three of the sixty leagues: half a table in words and half in codes.
    country: str = ""

    @property
    def group(self) -> str:
        """Competition-group id for the home-advantage term and the ridge centre."""
        return f"dom-{self.assoc.lower()}"

    def url(self, season: str) -> str:
        return self.url_tpl.format(season=season)


#: Where title-casing the repository's directory name does not give the country.
#: One entry, which is why this is a dict and not a table of forty-six.
_COUNTRY_SPELLING = {"bosnia-herzegovina": "Bosnia and Herzegovina"}


def _europe_repo(assoc: str, name: str, directory: str, code: str,
                 seasons: tuple[str, ...]) -> DomesticSource:
    """A country inside the 50-country `openfootball/europe` collection.

    The country name comes from the directory, which is already the country in
    lower case: `north-macedonia`, `faroe-islands`, `san-marino`. Deriving it
    beats a second hand-kept list of forty-six names that could disagree with
    the first.
    """
    country = _COUNTRY_SPELLING.get(directory, directory.replace("-", " ").title())
    return DomesticSource(assoc, name,
                          f"{OF}/europe/master/{directory}/{{season}}_{code}1.txt",
                          seasons, country)


_RECENT = ("2023-24", "2024-25")
_SUMMER = ("2023", "2024", "2025")

#: Every association that sends clubs into the competitions we model and has a
#: reachable domestic feed. Ordered by how much the rating base needs it: the
#: countries whose clubs reach the league phase most often come first.
#:
#: Missing on purpose: ISR and KAZ have no source at all anywhere on GitHub
#: (plan 1.5), so their clubs are rated from European matches only. MCO is not
#: missing -- Monaco plays in Ligue 1 and the France feed covers it.
DOMESTIC: tuple[DomesticSource, ...] = (
    _europe_repo("NED", "Eredivisie", "netherlands", "nl",
                 ("2018-19", "2019-20", "2020-21", "2021-22", "2022-23",
                  "2023-24", "2024-25", "2025-26", "2026-27")),
    _europe_repo("POR", "Primeira Liga", "portugal", "pt",
                 ("2018-19", "2019-20", "2020-21", "2021-22", "2022-23",
                  "2023-24", "2024-25", "2025-26", "2026-27")),
    DomesticSource("BEL", "Belgian Pro League",
                   f"{OF}/belgium/master/{{season}}/be1.txt",
                   ("2018-19", "2019-20", "2021-22", "2023-24", "2024-25",
                    "2025-26", "2026-27"), "Belgium"),
    DomesticSource("AUT", "Austrian Bundesliga",
                   f"{OF}/austria/master/{{season}}/1-bundesliga.txt",
                   ("2018-19", "2019-20", "2020-21", "2021-22", "2022-23",
                    "2023-24", "2024-25", "2025-26"), "Austria"),
    _europe_repo("SCO", "Scottish Premiership", "scotland", "sco",
                 ("2018-19", "2019-20", "2020-21", "2023-24", "2024-25", "2025-26")),
    _europe_repo("GRE", "Super League Greece", "greece", "gr",
                 ("2018-19", "2019-20", "2020-21", "2023-24", "2024-25", "2025-26")),
    _europe_repo("TUR", "Süper Lig", "turkey", "tr",
                 ("2018-19", "2019-20", "2020-21", "2023-24", "2024-25", "2025-26")),
    _europe_repo("SUI", "Swiss Super League", "switzerland", "ch",
                 ("2018-19", "2019-20", "2020-21", "2023-24", "2024-25")),
    _europe_repo("CZE", "Czech First League", "czech-republic", "cz",
                 ("2018-19", "2020-21", "2023-24", "2024-25")),
    _europe_repo("HUN", "Nemzeti Bajnokság I", "hungary", "hu",
                 ("2018-19", "2020-21", "2023-24", "2024-25")),
    _europe_repo("NOR", "Eliteserien", "norway", "no", _SUMMER),
    _europe_repo("SWE", "Allsvenskan", "sweden", "se", _SUMMER),
    _europe_repo("DEN", "Superliga", "denmark", "dk", _RECENT),
    _europe_repo("POL", "Ekstraklasa", "poland", "pl", _RECENT),
    _europe_repo("CRO", "HNL", "croatia", "hr", _RECENT),
    _europe_repo("SRB", "Serbian SuperLiga", "serbia", "rs", _RECENT),
    _europe_repo("UKR", "Ukrainian Premier League", "ukraine", "ua", _RECENT),
    _europe_repo("CYP", "Cypriot First Division", "cyprus", "cy", _RECENT),
    _europe_repo("SVN", "Slovenian PrvaLiga", "slovenia", "si", _RECENT),
    _europe_repo("ROU", "Liga I", "romania", "ro", _RECENT),
    _europe_repo("BUL", "First Professional League", "bulgaria", "bg", _RECENT),
    _europe_repo("SVK", "Slovak Super Liga", "slovakia", "sk", _RECENT),
    _europe_repo("BIH", "Premier League of BiH", "bosnia-herzegovina", "ba", _RECENT),
    _europe_repo("AZE", "Azerbaijan Premier League", "azerbaijan", "az", _RECENT),
    _europe_repo("ARM", "Armenian Premier League", "armenia", "am", _RECENT),
    _europe_repo("MDA", "Moldovan Super Liga", "moldova", "md", _RECENT),
    _europe_repo("MKD", "Macedonian First League", "north-macedonia", "mk", _RECENT),
    _europe_repo("ALB", "Kategoria Superiore", "albania", "al", _RECENT),
    _europe_repo("KOS", "Football Superleague of Kosovo", "kosovo", "kos", _RECENT),
    _europe_repo("MNE", "Montenegrin First League", "montenegro", "me", _RECENT),
    _europe_repo("WAL", "Cymru Premier", "wales", "wal", _RECENT),
    _europe_repo("NIR", "NIFL Premiership", "northern-ireland", "nir", _RECENT),
    _europe_repo("IRL", "League of Ireland", "ireland", "ie", _SUMMER),
    _europe_repo("FIN", "Veikkausliiga", "finland", "fi", _SUMMER),
    _europe_repo("ISL", "Besta deild", "iceland", "is", _SUMMER),
    _europe_repo("LVA", "Virsliga", "latvia", "lv", _SUMMER),
    _europe_repo("LTU", "A Lyga", "lithuania", "lt", _SUMMER),
    _europe_repo("EST", "Meistriliiga", "estonia", "ee", _SUMMER),
    _europe_repo("BLR", "Belarusian Premier League", "belarus", "by", _SUMMER),
    _europe_repo("GEO", "Erovnuli Liga", "georgia", "ge", _SUMMER),
    _europe_repo("FRO", "Betri deildin", "faroe-islands", "fo", _SUMMER),
    _europe_repo("LUX", "Luxembourg National Division", "luxembourg", "lu", _RECENT),
    _europe_repo("MLT", "Maltese Premier League", "malta", "mt", _RECENT),
    _europe_repo("GIB", "Gibraltar Football League", "gibraltar", "gi", _RECENT),
    _europe_repo("AND", "Primera Divisió", "andorra", "ad", _RECENT),
    _europe_repo("SMR", "Campionato Sammarinese", "san-marino", "sm", _RECENT),
)

BY_ASSOC: dict[str, DomesticSource] = {s.assoc: s for s in DOMESTIC}


def load_domestic(reg: TeamRegistry, assocs: list[str] | None = None,
                  *, quiet: bool = False) -> list[Match]:
    """Played matches from every reachable non-big-five top flight.

    Each match is tagged with its league's group id, which is what lets the
    pooled fit give Norway its own home-advantage term and its own ridge centre
    instead of pretending Eliteserien and the Premier League are one population.
    """
    out: list[Match] = []
    srcs = [BY_ASSOC[a] for a in assocs] if assocs else list(DOMESTIC)
    for src in srcs:
        n0 = len(out)
        for season in src.seasons:
            text = fetch.get(src.url(season), required=False)
            if not text:
                continue
            got = parse_openfootball(text, season, reg, comp=src.group)
            out.extend(m for m in got if m.played)
        if not quiet and len(out) == n0:
            print(f"  ! {src.assoc} {src.name}: no reachable seasons")
    if not quiet:
        print(f"  · {len(out)} matches from {len(srcs)} non-big-five leagues")
    return out



# --------------------------------------------------------------------------
# Second feeds, off GitHub
# --------------------------------------------------------------------------
#: The associations whose openfootball file stopped and which a second source
#: can carry. Nothing here is trusted on sight: `model.external.probe` fetches
#: each one, parses it and resolves every club name against the registry before
#: a single match is allowed into the corpus. See `model/external.py` for why
#: that gate exists and why it cannot be satisfied where this code is written.
#:
#: The group ids match the `DomesticSource` entries above exactly, which is what
#: lets a second feed extend a league rather than found a parallel one.
#: Set once the probe has printed, so rebuilding the corpus per competition does
#: not reprint the same nine lines nine times.
_ANNOUNCED: list[bool] = []

#: The most recent probe's verdicts, for whoever writes `sources.json`. A
#: module-level handoff because the corpus is built deep inside a per-league
#: call and the file is written once for the whole site.
LAST_VERDICTS: list["object"] = []


def seasons_held_in_full(domestic: list[Match] | None, group: str,
                         listed: tuple[str, ...]) -> tuple[str, ...]:
    """The seasons of one league the GitHub feed carries enough of to check against.

    Not the same question as which seasons it lists. openfootball opens a
    season's file when the season starts and then, for the leagues that went
    quiet, stops: Norway's 2025 file holds 44 of its 240 matches and Belarus's
    holds 8 of 240. Anchoring the overlap on a file like that fails the probe on
    thinness -- 44 fixtures is below the sixty it needs -- and would have read
    as "Wikipedia disagrees with us" when the truth is that we stopped looking.

    So the anchor is the newest season with a real season's worth of matches in
    it, and every season after that one, thin file or no file, is asked for.
    """
    if domestic is None:
        return tuple(listed)
    n: dict[str, int] = {}
    for m in domestic:
        if m.comp == group and m.played:
            n[m.season] = n.get(m.season, 0) + 1
    if not n:
        return tuple(listed)
    # Against this league's own biggest season rather than a fixed number,
    # because "a season's worth" is 56 matches in Moldova and 380 in England.
    # A fixed floor gets one of them wrong whichever number is chosen: at sixty
    # it refused Moldova's complete season for being small, and at fifty it
    # accepted the fifty-odd matches of Sweden's abandoned 2025 file as though
    # they were a season, anchored the overlap there, and lost a league that had
    # been arming cleanly on 2024.
    most = max(n.values())
    full = tuple(sorted(s for s, c in n.items() if c >= 0.8 * most))
    return full or tuple(listed)


def external_sources(domestic: list[Match] | None = None) -> list["object"]:
    from . import footballdata, wikifootball
    from .external import ExternalSource

    out: list[ExternalSource] = []
    for assoc in ("POL", "ROU", "SUI"):
        src = BY_ASSOC[assoc]
        out.append(footballdata.source(assoc, src.name, src.group))
    for assoc in sorted(wikifootball.ARMED | wikifootball.CANDIDATES):
        src = BY_ASSOC[assoc]
        # The last season the GitHub feed carries in full, plus everything after
        # it. The first of those is not wanted for its data -- we already have
        # it -- but as the season the probe lines the two feeds up on.
        have = seasons_held_in_full(domestic, src.group, src.seasons)
        out.append(wikifootball.source(
            assoc, src.name, src.group, wikifootball.seasons_for(have),
            contributes=assoc in wikifootball.ARMED))
    return out


def load_external(reg: TeamRegistry, domestic: list[Match],
                  *, quiet: bool = False) -> tuple[list[Match], list["object"]]:
    """Matches from every second feed that passes its probe, and the verdicts.

    `domestic` is what the GitHub feeds just produced. Each source is checked
    against its own league's slice of it, which is what makes the probe a
    comparison rather than a download.

    The verdicts are returned rather than swallowed because they are the only
    account of what happened: on a runner they are printed, and in
    `coverage.json` they are what lets the site say which competitions are
    running on a second feed and which are still stopped at 2024-25.
    """
    from . import external

    srcs = external_sources(domestic)
    if not srcs:
        return [], []
    # The corpus is rebuilt once per competition, so a probe that printed every
    # time would print nine times and say the same thing nine times. The fetch
    # cache makes the repeats free; this makes them quiet.
    say = not quiet and not _ANNOUNCED
    if say:
        print(f"Probing {len(srcs)} second feed(s) outside GitHub…")
    by_group: dict[str, list[Match]] = {}
    for m in domestic:
        if m.comp:
            by_group.setdefault(m.comp, []).append(m)

    out: list[Match] = []
    verdicts = []
    fed: set[str] = set()
    #: (group, season) pairs where the GitHub feed opened a file and abandoned
    #: it, and a second feed carries the whole season. The caller drops these
    #: from the domestic corpus before adding what came back.
    stale: set[tuple[str, str]] = set()
    for src in srcs:
        mine = by_group.get(src.group, [])
        v = external.probe(src, reg, mine)
        verdicts.append(v)
        if say:
            print(v.line())
        if not src.armed:
            continue
        if src.group in fed:
            # Two armed feeds for one league would each add the seasons the
            # GitHub feed lacks, and `external.matches` only knows about the
            # GitHub ones -- so every match of those seasons would go into the
            # fit twice. Poland is the live case: it has both a football-data
            # file and a Wikipedia grid.
            raise ValueError(
                f"{src.source}/{src.assoc} is a second armed feed for "
                f"{src.group}; arm one of them, not both")
        fed.add(src.group)
        out.extend(external.matches(src, reg, mine))
        for season in src.superseded:
            stale.add((src.group, season))
    if say:
        armed = sum(1 for v in verdicts if v.ok and not v.watching)
        watch = sum(1 for v in verdicts if v.watching)
        print(f"  · {armed}/{len(srcs) - watch} armed, {len(out)} matches added"
              + (f", {watch} watched" if watch else ""))
        _ANNOUNCED.append(True)
    LAST_VERDICTS[:] = verdicts
    if say and stale:
        for group, season in sorted(stale):
            print(f"  · {group} {season}: the GitHub file is a stub, "
                  "so the second feed's season replaces it")
    return out, verdicts, stale


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------
def group_key(m: Match, default: str = "") -> str:
    """The competition group a match belongs to, for the home term and the ridge.

    Domestic matches loaded by `model.data` carry no `comp`, so the caller
    passes the league slug as `default`; everything loaded here labels itself.
    """
    if m.comp in EURO_COMPS:
        return EUROPE
    return m.comp or default


#: How far back "which league does this club play in" is allowed to look. A club
#: plays in one division at a time and the answer that matters is the current
#: one, so recent matches decide it and older ones only break ties.
CURRENT_WINDOW_DAYS = 550


def club_leagues(matches: list[Match], default: str = "other") -> dict[str, str]:
    """Which league each club belongs to, for the hierarchical ridge centre.

    A club plays for exactly one league at a time, so what is wanted is the
    division it is in *now* -- not the one it has spent the most matches in.
    Counting the whole record would file Brentford under the Championship, where
    it played for twenty seasons, rather than the Premier League, where it plays
    today; the ridge would then shrink its rating toward the wrong mean and the
    global ranking would print the wrong league beside its name.

    So: the most common domestic group inside the last eighteen months, with the
    whole record as a tie-break for a club that has not played recently. Clubs
    seen only in European matches -- Israeli and Kazakh sides, for which no
    domestic feed exists anywhere -- fall back to a shared 'other' pool: with no
    league-mates to borrow from, shrinking them toward the global mean is the
    honest default and the one the old zero-centred ridge always used.
    """
    latest = max((m.date for m in matches), default=None)
    cutoff = (latest - dt.timedelta(days=CURRENT_WINDOW_DAYS)) if latest else None

    recent: dict[str, dict[str, int]] = {}
    ever: dict[str, dict[str, int]] = {}
    for m in matches:
        g = group_key(m)
        if g in ("", EUROPE):
            continue
        for t in (m.home, m.away):
            ever.setdefault(t, {})[g] = ever.setdefault(t, {}).get(g, 0) + 1
            if cutoff is not None and m.date >= cutoff:
                recent.setdefault(t, {})[g] = recent.setdefault(t, {}).get(g, 0) + 1
    out: dict[str, str] = {}
    for t, counts in ever.items():
        pick = recent.get(t) or counts
        out[t] = max(pick.items(), key=lambda kv: kv[1])[0]
    for m in matches:
        for t in (m.home, m.away):
            out.setdefault(t, default)
    return out


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "europe")


def our_fixture_path(season: str) -> str:
    return os.path.join(DATA, f"fixtures-{season}.txt")


def participants_path(season: str) -> str:
    return os.path.join(DATA, f"participants-{season}.json")


class AwaitingDraw(RuntimeError):
    """No pairings exist yet. Not an error in the source -- an error in the
    calendar. The 2026-27 draw is 27 August 2026; before it there is nothing to
    forecast and the league must stay `ready: false` rather than ship a guess."""


def load_cup_fixtures(reg: TeamRegistry, season: str, comp: str = "cl",
                      *, quiet: bool = False) -> tuple[list[Match], dict]:
    """The league-phase fixture list, ours first.

    Plan Risk 1, and it is the difference between shipping and not: openfootball
    published the last three league-phase files +3, +68 and +208 days after the
    draw, and never published two of them at all. So `data/europe/fixtures-{season}.txt`
    is the PRIMARY source, and the upstream file is consulted only as a results
    override -- applied when, and only when, it has strictly more played matches
    than we do.
    """
    ours: list[Match] = []
    path = our_fixture_path(season)
    if os.path.exists(path):
        text = open(path, encoding="utf-8").read()
        ours = [m for m in parse_openfootball_euro(text, season, reg, comp)
                if m.stage in (None, "league")]

    text = fetch.get(euro_url(season, comp), required=False)
    theirs: list[Match] = []
    if text:
        theirs = [m for m in parse_openfootball_euro(text, season, reg, comp)
                  if m.stage == "league"]

    n_ours = sum(1 for m in ours if m.played)
    n_theirs = sum(1 for m in theirs if m.played)
    meta = {"season": season, "ours": len(ours), "ours_played": n_ours,
            "openfootball": len(theirs), "openfootball_played": n_theirs}

    if not ours and not theirs:
        raise AwaitingDraw(
            f"no {comp} fixtures for {season}: {path} is empty and openfootball "
            f"has no {season}/{comp}.txt")
    if not ours:
        meta["source"] = "openfootball"
        return theirs, meta
    if theirs and n_theirs > n_ours:
        meta["source"] = "ours+openfootball-results"
        by_pair = {(m.home, m.away): m for m in theirs if m.played}
        for f in ours:
            src = by_pair.get((f.home, f.away))
            if src is not None and not f.played:
                f.hg, f.ag, f.played, f.aet = src.hg, src.ag, True, src.aet
        return ours, meta
    meta["source"] = "ours"
    return ours, meta


def load_replay_fixtures(reg: TeamRegistry, season: str, comp: str = "cl"
                         ) -> tuple[list[Match], dict]:
    """A finished season's league phase, replayed as if it were the fixture list.

    Staging data. The site needs 36 rows, eight matchdays and a bracket to build
    against, and the 2026-27 draw does not exist until 27 August; a real finished
    season is a far better fixture than anything synthetic, and it is stamped as
    a replay so it can never be mistaken for the live forecast.
    """
    text = fetch.get(euro_url(season, comp), required=True)
    got = [m for m in parse_openfootball_euro(text, season, reg, comp)
           if m.stage == "league"]
    return got, {"season": season, "source": "openfootball-replay",
                 "ours": 0, "ours_played": 0,
                 "openfootball": len(got),
                 "openfootball_played": sum(1 for m in got if m.played)}


def load_big_five(reg: TeamRegistry, *, from_year: int = 2015,
                  season: str | None = None, quiet: bool = False) -> list[Match]:
    """Big-five top-flight history, tagged with each league's group id.

    The same football-data mirror the domestic pipeline reads, loaded here only
    to give the pooled fit its other end of the bridge. Shot columns are ignored:
    a pooled fit is goals-only by necessity, since no non-big-five feed has them.
    """
    from . import config, leagues as _lg
    from .data import _season_label
    from .parse import parse_football_data_csv

    out: list[Match] = []
    season = season or config.SEASON
    for lg in _lg.BIG_FIVE:
        for code in lg.fd_season_codes(season):
            label = _season_label(code)
            if int(label.split("-")[0]) < from_year:
                continue
            text = fetch.results_csv(lg, code, required=False)
            if not text:
                continue
            for m in parse_football_data_csv(text, label, reg):
                if m.played:
                    m.comp = lg.slug
                    out.append(m)
    if not quiet:
        print(f"  · {len(out)} big-five matches from {from_year}")
    return out


def load_second_tiers(reg: TeamRegistry, *, from_year: int = 2015,
                      quiet: bool = False) -> list[Match]:
    """The division below each competition this site forecasts.

    Not for its own sake. Fourteen clubs with a forecast page had no rating at
    all -- no SPI, no attack, no defence -- because they are projected up from a
    division the pooled corpus could not see, and their only top-flight record
    was too old for the fit to use. A league table with three blank rows in its
    strength column is a worse answer than the one this fixes.

    It also closes the gaps in a club's rating trajectory. A season spent one
    division down used to be a hole in the line, because the corpus had nothing
    from it; now the line runs through it, which is what a trajectory is for.

    Tagged `<slug>-2`, its own competition group, so the pooled fit gives the
    division its own home-advantage term and its own ridge centre rather than
    pretending the Championship and the Premier League are one population. The
    scale set is unchanged: it is defined as the big-five *top* flights, and a
    second-tier group is not one of them, so no published number moves because
    of what a second tier does.
    """
    # A division this site forecasts in its own right is loaded by the ranking
    # build under its own slug. England's second tier is the Championship, which
    # is one of the nine, so reading it here as well would put every Championship
    # match into the fit twice and quietly double its weight. Compared by URL
    # rather than by a hand-kept list of exceptions, so a second tier promoted to
    # a forecast competition later cannot reintroduce this.
    forecast_top = {lg.of_url(config.SEASON, "top") for lg in leagues.LEAGUES}
    out: list[Match] = []
    for lg in leagues.LEAGUES:
        if not lg.of_second:
            continue
        if lg.of_url(config.SEASON, "second") in forecast_top:
            if not quiet:
                print(f"  · {lg.slug}: second tier is a forecast competition, "
                      "loaded there instead")
            continue
        n0 = len(out)
        for label in lg.second_season_labels(config.SEASON):
            if int(label.split("-")[0]) < from_year:
                continue
            text = fetch.get(lg.of_url(label, "second"), required=False)
            if not text:
                continue
            got = parse_openfootball(text, label, reg, comp=f"{lg.slug}-2")
            out.extend(m for m in got if m.played)
        if not quiet and len(out) == n0:
            print(f"  ! {lg.slug}: no reachable second-tier seasons")
    if not quiet:
        print(f"  · {len(out)} matches from the divisions below")
    return out


class Corpus:
    """Every match the pooled fit sees, with its competition group attached.

    Assembling this is the whole of the 'data is the hard half' problem: five
    domestic feeds in one shape, forty-six in another, six UEFA competitions in
    a third, and one club-id namespace across all of them.
    """

    def __init__(self, reg: TeamRegistry | None = None) -> None:
        self.reg = reg or TeamRegistry()
        self.matches: list[Match] = []
        self.euro: list[Match] = []
        #: One `external.Verdict` per second feed, filled in by `load`.
        self.verdicts: list[object] = []

    def add(self, matches: list[Match], group: str) -> "Corpus":
        """Fold in matches from a source that does not label itself."""
        for m in matches:
            if m.played:
                if not m.comp:
                    m.comp = group
                self.matches.append(m)
        return self

    #: Where the corpus starts. 2011, because that is where the UEFA files start
    #: and those matches are the only edges joining one league to another: a
    #: season with domestic results and no European ties is a set of leagues
    #: with no exchange rate between them. It used to be 2015, which left the
    #: 2012-13 and 2013-14 fits with nothing but European ties in them, and a
    #: club rated off twelve of those came out wherever those twelve fell.
    def load(self, *, competitions: bool = True, domestic: bool = True,
             big_five: bool = True, second: bool = True, from_year: int = 2011,
             seasons: list[str] | None = None, quiet: bool = False) -> "Corpus":
        if competitions:
            self.euro = load_competitions(self.reg, seasons, quiet=quiet)
            self.matches.extend(self.euro)
        if domestic:
            dom = load_domestic(self.reg, quiet=quiet)
            self.matches.extend(dom)
            ext, self.verdicts, stale = load_external(self.reg, dom, quiet=quiet)
            if stale:
                keep = [m for m in self.matches
                        if (m.comp, m.season) not in stale]
                self.matches[:] = keep
            self.matches.extend(ext)
        if big_five:
            self.matches.extend(load_big_five(self.reg, from_year=from_year,
                                              quiet=quiet))
        if second:
            self.matches.extend(load_second_tiers(self.reg, from_year=from_year,
                                                  quiet=quiet))
        return self

    # -- views the fit needs ------------------------------------------------
    @property
    def teams(self) -> list[str]:
        return sorted({m.home for m in self.matches} | {m.away for m in self.matches})

    def group_of(self, m: Match) -> str:
        return group_key(m)

    def club_leagues(self) -> dict[str, str]:
        return club_leagues(self.matches)

    def last_seen(self) -> dict[str, object]:
        return last_seen(self.matches)

    def before(self, cutoff) -> list[Match]:
        return [m for m in self.matches if m.date < cutoff]


def last_seen(matches: list[Match]) -> dict[str, "object"]:
    """Date of each club's most recent match in the corpus.

    Feeds the staleness inflation of `RATING_SD`: a club whose newest result is
    fifteen months old is not badly rated, it is uncertainly rated.
    """
    out: dict[str, object] = {}
    for m in matches:
        for t in (m.home, m.away):
            if t not in out or m.date > out[t]:
                out[t] = m.date
    return out
