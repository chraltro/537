"""The European corpus: UEFA competition results, and the domestic leagues that
feed them.

Two jobs, both of them supply problems rather than modelling ones.

*The bridge.* `openfootball/champions-league` carries fifteen seasons of UEFA
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

from . import fetch
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

    @property
    def group(self) -> str:
        """Competition-group id for the home-advantage term and the ridge centre."""
        return f"dom-{self.assoc.lower()}"

    def url(self, season: str) -> str:
        return self.url_tpl.format(season=season)


def _europe_repo(assoc: str, name: str, directory: str, code: str,
                 seasons: tuple[str, ...]) -> DomesticSource:
    """A country inside the 50-country `openfootball/europe` collection."""
    return DomesticSource(assoc, name,
                          f"{OF}/europe/master/{directory}/{{season}}_{code}1.txt",
                          seasons)


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
                    "2025-26", "2026-27")),
    DomesticSource("AUT", "Austrian Bundesliga",
                   f"{OF}/austria/master/{{season}}/1-bundesliga.txt",
                   ("2018-19", "2019-20", "2020-21", "2021-22", "2022-23",
                    "2023-24", "2024-25", "2025-26")),
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

    def add(self, matches: list[Match], group: str) -> "Corpus":
        """Fold in matches from a source that does not label itself."""
        for m in matches:
            if m.played:
                if not m.comp:
                    m.comp = group
                self.matches.append(m)
        return self

    def load(self, *, competitions: bool = True, domestic: bool = True,
             big_five: bool = True, from_year: int = 2015,
             seasons: list[str] | None = None, quiet: bool = False) -> "Corpus":
        if competitions:
            self.euro = load_competitions(self.reg, seasons, quiet=quiet)
            self.matches.extend(self.euro)
        if domestic:
            self.matches.extend(load_domestic(self.reg, quiet=quiet))
        if big_five:
            self.matches.extend(load_big_five(self.reg, from_year=from_year,
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
