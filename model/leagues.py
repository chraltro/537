"""The five leagues the pipeline forecasts, and everything that differs between them.

One `League` object carries every fact the rest of the model needs: how big the
season is, where its results and fixtures come from, how many Champions League
places are on the line and how many clubs go down. Nothing downstream reads a
league constant from anywhere else, which is what makes `--league la-liga` a
parameter change rather than a fork of the code.

Two kinds of fact live here and they are worth telling apart:

* Source shapes -- mirror directories and file paths. Verified reachable
  2026-08-18 against the URLs in docs/multi-league.md.
* Competition rules -- the Champions League line and the relegation line for
  2026-27. These change every year and are sourced individually below.
"""
from __future__ import annotations

from dataclasses import dataclass

# Both feeds are raw GitHub content, so the pipeline runs identically locally
# and on an Actions runner with no API keys anywhere.
FD_BASE = "https://raw.githubusercontent.com/datasets/football-datasets/main/datasets"
OF_BASE = "https://raw.githubusercontent.com/openfootball"

#: How a season is written inside an openfootball path. All five repositories
#: happen to agree on 'YYYY-YY' (2026-27); what actually differs between them is
#: the shape of the path around it, which is why the templates below are stored
#: whole rather than assembled from parts.
OF_SEASON_STYLE = "YYYY-YY"


@dataclass(frozen=True)
class League:
    """One competition, its sources, and its 2026-27 qualification rules."""

    slug: str
    name: str
    country: str
    n_teams: int
    n_matches: int


    # -- competition rules, 2026-27 -----------------------------------------
    ucl_places: int            # league positions that reach the UCL league phase
    releg_places: int          # DIRECT relegation spots only
    releg_note: str | None     # play-off wording shown alongside, or None
    europa_places: int = 2     # positions below the UCL line, approximate:
                               # domestic cup winners move this line every year

    #: "league" = a domestic double round robin; "cup" = a UEFA-style
    #: competition whose league phase feeds a knockout; "promotion" = a second
    #: tier, where the top line is automatic promotion and a band beneath it
    #: plays off for one more place. Cups and promotion leagues both read their
    #: table against advancement lines instead of a Champions League line.
    kind: str = "league"
    advance_direct: int | None = None    # league-phase positions straight to the R16
    advance_playoff: int | None = None   # further positions into the knockout play-off

    #: Shown by the site's league switcher while ready=false, e.g. the date the
    #: competition's draw makes a real forecast possible.
    ready_note: str | None = None

    # -- sources -------------------------------------------------------------
    fd_dir: str = ""           # football-datasets mirror directory (history + shots)
    of_top: str = ""           # openfootball path template, top flight
    of_second: str = ""        # openfootball path template, second tier
    fd_from: int = 2000        # first season year of usable mirror CSVs
    second_from: int = 2010    # first season year the second tier exists upstream

    #: Where this competition's own history comes from. "mirror" is the
    #: football-datasets CSV, which carries shots, cards, half-time scores and
    #: the referee; "openfootball" is the plain-text fixture file, which carries
    #: goals and nothing else. Only the big five are in the mirror -- it has
    #: exactly five league directories -- so every competition added beyond them
    #: is goals-only, the shot blend degrades gracefully via its coverage
    #: weighting, and the method page says so per league.
    source: str = "mirror"
    #: First season of openfootball top-flight history, used when source is
    #: "openfootball". Listed rather than probed: 404s cost a round trip each.
    of_from: int = 2018

    #: The division above this one, by slug, for a league that is not a top
    #: flight. A top flight loads the tier BELOW so it can rate the clubs coming
    #: up; a second tier needs the mirror image, because three of its clubs
    #: every season arrive from above and have no recent record in it at all.
    #: Without this a club relegated from the Premier League is rated on
    #: whatever it did in the Championship years ago -- for West Ham, 2011-12,
    #: which an eight-month half-life discounts to nothing -- and then shrunk
    #: again by a correction measured on clubs promoted from below.
    above_slug: str = ""

    #: How clubs level on points are separated. "gd" is points, goal difference,
    #: goals scored -- England, Germany and France. "h2h" puts the mini-table
    #: among the level clubs first, which is what Spain and Italy actually do and
    #: what this simulation ignored until now.
    tiebreak: str = "gd"

    #: The finishing position that plays a two-legged tie against a second-tier
    #: club instead of going down automatically. Germany and France both send
    #: their third-from-bottom; everywhere else this is None and the drop is
    #: settled by the table alone.
    releg_playoff_pos: int | None = None

    #: Backtest window. The Premier League keeps the full record because it is
    #: the league the model was tuned on and the one whose scores are published
    #: as the headline; the others start later purely to keep the five-league
    #: run inside the ~12 minute budget in docs/multi-league.md.
    backtest_from: str = "2015-16"

    # -- derived -------------------------------------------------------------
    @property
    def out_dir(self) -> str:
        return self.slug

    @property
    def market_file(self) -> str:
        """Per-league market anchor. Absent file => the anchor gets weight 0."""
        return f"{self.slug}.json"

    def fd_csv_url(self, code: str) -> str:
        """`code` is a football-datasets season code such as '2526'."""
        return f"{FD_BASE}/{self.fd_dir}/season-{code}.csv"

    def of_url(self, season: str, tier: str) -> str:
        """`tier` is 'top' or 'second'; `season` is a 'YYYY-YY' label."""
        tpl = self.of_top if tier == "top" else self.of_second
        return f"{OF_BASE}/{tpl.format(season=season)}.txt"

    def fd_season_codes(self, season: str) -> list[str]:
        """Mirror season codes from `fd_from` up to and including `season`."""
        end = int(season.split("-")[0])
        return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(self.fd_from, end + 1)]

    def second_season_labels(self, season: str) -> list[str]:
        end = int(season.split("-")[0])
        return [f"{y}-{(y + 1) % 100:02d}" for y in range(self.second_from, end + 1)]

    def top_season_labels(self, season: str) -> list[str]:
        """Top-flight season labels for a competition read from openfootball."""
        end = int(season.split("-")[0])
        return [f"{y}-{(y + 1) % 100:02d}" for y in range(self.of_from, end + 1)]

    def manifest_entry(self, ready: bool = False) -> dict:
        """The row this league contributes to site/data/leagues.json."""
        row = {"slug": self.slug, "name": self.name, "country": self.country,
               "ready": ready, "kind": self.kind, "n_teams": self.n_teams,
               "ucl_places": self.ucl_places, "releg_places": self.releg_places,
               "releg_note": self.releg_note}
        if self.kind in ("cup", "promotion"):
            row["advance_direct"] = self.advance_direct
            row["advance_playoff"] = self.advance_playoff
        # Only while the league is *not* ready. It is the sentence the switcher
        # shows in place of a forecast ("League-phase draw: 27 August"), and a
        # ready league shipping it left a note about a draw that has already
        # happened in published JSON that the feed and the sitemap builder read.
        if self.ready_note and not ready:
            row["ready_note"] = self.ready_note
        return row

    def public(self) -> dict:
        """The 'league' block embedded in forecast.json."""
        row = {"slug": self.slug, "name": self.name, "country": self.country,
               "ucl_places": self.ucl_places, "releg_places": self.releg_places,
               "releg_note": self.releg_note, "n_teams": self.n_teams}
        if self.kind in ("cup", "promotion"):
            # A cup's table is read against advancement lines, not against a
            # European place and a drop; a second tier's top line is automatic
            # promotion with a play-off band under it. Both take their wording
            # from here rather than from anything hardcoded in the front end.
            row["kind"] = self.kind
            row["advance_direct"] = self.advance_direct
            row["advance_playoff"] = self.advance_playoff
        return row


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------
# Champions League places, 2026-27. UEFA's 36-club league phase gives the top
# four associations four places each and the fifth-ranked association three,
# plus two "European Performance Spots" awarded to the two associations whose
# clubs performed best across 2025-26 European competition.
#
#   England and Spain took both performance spots, so each has five.
#     https://www.uefa.com/uefachampionsleague/news/02a2-1fdbe9a25733-8d37ff5f9226-1000--2026-27-uefa-champions-league-england-and-spain-secure-next-/
#     https://www.premierleague.com/en/news/4573748
#   Italy and Germany (associations 3 and 4) have four each.
#   France is the fifth-ranked association: three clubs go straight into the
#   league phase and the fourth enters the third qualifying round, so the line
#   the simulation reports as "Champions League" is third, not fourth.
#     https://sports.yahoo.com/articles/qualified-next-seasons-champions-league-131551517.html
#
# Relegation, 2026-27. England, Spain and Italy relegate the bottom three with
# no play-off. Germany and France relegate the bottom two directly and send the
# third-bottom club into a two-legged play-off against a second-tier side.
PREMIER_LEAGUE = League(
    slug="premier-league", name="Premier League", country="England",
    n_teams=20, n_matches=380,
    ucl_places=5, releg_places=3, releg_note=None,
    fd_dir="premier-league",
    of_top="england/master/{season}/1-premierleague",
    of_second="england/master/{season}/2-championship",
    # 2000-01 is where the mirror's shots-on-target coverage becomes reliable.
    fd_from=2000, second_from=2010,
    backtest_from="2015-16",
)

LA_LIGA = League(
    slug="la-liga", name="La Liga", country="Spain",
    n_teams=20, n_matches=380,
    ucl_places=5, releg_places=3, releg_note=None,
    tiebreak="h2h",
    fd_dir="la-liga",
    of_top="espana/master/{season}/1-liga",
    of_second="espana/master/{season}/2-liga2",
    fd_from=2000, second_from=2012,
    backtest_from="2015-16",   # common window across leagues for comparability
)

SERIE_A = League(
    slug="serie-a", name="Serie A", country="Italy",
    n_teams=20, n_matches=380,
    ucl_places=4, releg_places=3, releg_note=None,
    tiebreak="h2h",
    fd_dir="serie-a",
    of_top="italy/master/{season}/1-seriea",
    of_second="italy/master/{season}/2-serieb",
    fd_from=2000, second_from=2013,
    backtest_from="2015-16",   # common window across leagues for comparability
)

BUNDESLIGA = League(
    slug="bundesliga", name="Bundesliga", country="Germany",
    n_teams=18, n_matches=306,
    ucl_places=4, releg_places=2,
    releg_note="16th plays a two-legged relegation play-off against the "
               "third-placed 2. Bundesliga club",
    releg_playoff_pos=16,
    fd_dir="bundesliga",
    of_top="deutschland/master/{season}/1-bundesliga",
    of_second="deutschland/master/{season}/2-bundesliga2",
    fd_from=2000, second_from=2012,
    backtest_from="2015-16",   # common window across leagues for comparability
)

LIGUE_1 = League(
    slug="ligue-1", name="Ligue 1", country="France",
    n_teams=18, n_matches=306,
    # France is UEFA's fifth association: three direct league-phase places, with
    # 4th entering the Champions League third qualifying round.
    ucl_places=3, releg_places=2,
    releg_note="16th plays a two-legged relegation play-off against a Ligue 2 "
               "club; 4th enters Champions League qualifying",
    releg_playoff_pos=16,
    fd_dir="ligue-1",
    of_top="france/master/france/{season}_fr1",
    of_second="france/master/france/{season}_fr2",
    fd_from=2000, second_from=2014,
    backtest_from="2015-16",   # common window across leagues for comparability
)

# ---------------------------------------------------------------------------
# Beyond the big five. These three have no football-datasets directory -- the
# mirror carries exactly five league folders -- so they are read from
# openfootball and run on goals alone. `ratings.fit`'s coverage-weighted blend
# already handles a club with no shot data (it is what rates promoted clubs),
# and the method page states the difference per league rather than hiding it.
#
# The single-source dependency is the real risk and it is why the freshness
# panel exists: openfootball's in-season commits are roughly weekly with
# multi-week gaps (measured on 2025-26: 2025-11-11 -> 2025-12-31 -> 2026-02-10),
# and unlike the big five there is no second feed to fall back to.
EREDIVISIE = League(
    slug="eredivisie", name="Eredivisie", country="Netherlands",
    n_teams=18, n_matches=306,
    # The Netherlands sends its champions and its runners-up to the league
    # phase -- this repository's own data/europe/participants-2026-27.json
    # lists PSV (champions) and Feyenoord (2nd) among the 36, and that file is
    # the sourced artefact the Champions League forecast already stands on.
    ucl_places=2,
    releg_places=1,
    releg_note="16th and 17th enter the promotion/relegation play-offs",
    europa_places=2,
    of_top="europe/master/netherlands/{season}_nl1",
    of_second="europe/master/netherlands/{season}_nl2",
    source="openfootball", of_from=2018, second_from=2020,
    backtest_from="2021-22",
)

PRIMEIRA_LIGA = League(
    slug="primeira-liga", name="Primeira Liga", country="Portugal",
    n_teams=18, n_matches=306,
    # Same evidence: participants-2026-27.json carries Porto (champions) and
    # Sporting CP (2nd) in the league phase.
    ucl_places=2,
    releg_places=2, releg_note=None,
    europa_places=2,
    of_top="europe/master/portugal/{season}_pt1",
    of_second="europe/master/portugal/{season}_pt2",
    source="openfootball", of_from=2018, second_from=2020,
    backtest_from="2021-22",
)

#: The second tier read as its own competition: the top line is automatic
#: promotion, the band beneath it plays off for one more place, and the bottom
#: three go down. `kind="promotion"` is what makes the site say "promotion"
#: where a top flight says "Champions League".
CHAMPIONSHIP = League(
    slug="championship", name="Championship", country="England",
    n_teams=24, n_matches=552,
    kind="promotion", advance_direct=2, advance_playoff=4,
    # Reusing the two line fields keeps every downstream computation identical:
    # `ucl_places` is the automatic-promotion line, `releg_places` the drop.
    ucl_places=2, releg_places=3,
    releg_note="3rd to 6th play off for the third promotion place",
    europa_places=4,
    of_top="england/master/{season}/2-championship",
    of_second="england/master/{season}/3-league1",
    above_slug="premier-league",
    source="openfootball", of_from=2004, second_from=2004,
    backtest_from="2021-22",
)


#: Belgium's 2026-27 season is a straight eighteen-club double round robin --
#: the file's own header says "nieuw format: 1 reguliere competitie, 34
#: speeldagen, GEEN play-offs", replacing the championship/relegation split that
#: made earlier seasons impossible to model as a single table. That is what
#: makes this addable now and not before.
JUPILER_PRO_LEAGUE = League(
    slug="pro-league", name="Pro League", country="Belgium",
    n_teams=18, n_matches=306,
    # Belgium sends its champions to the league phase; participants-2026-27.json
    # carries Club Brugge on exactly that basis.
    ucl_places=1,
    releg_places=1,
    releg_note="17th plays a relegation play-off against a Challenger Pro League club",
    europa_places=3,
    of_top="belgium/master/{season}/be1",
    # No second tier upstream: openfootball has no `be2` at any season. The
    # promoted-club correction therefore falls back to the measured Premier
    # League constants, which `priors.regress` records in its output.
    of_second="",
    source="openfootball", of_from=2018, second_from=2026,
    backtest_from="2021-22",
)


# ---------------------------------------------------------------------------
# European competitions. The league phase is literally a 36-team league whose
# clubs each play 8 of the other 35, so the same machinery carries it; the
# knockout stages live in model/knockout.py. Fixture source is OUR OWN
# committed file (data/europe/fixtures-{season}.txt) with openfootball as a
# results override only -- see docs/european-competitions-plan.md, Risk 1.
CHAMPIONS_LEAGUE = League(
    slug="champions-league",
    name="Champions League",
    country="Europe",
    n_teams=36, n_matches=144,
    kind="cup", advance_direct=8, advance_playoff=16,
    ready_note="League-phase draw: 27 August",
    # UCL/relegation lines are meaningless for a cup; kept harmless.
    ucl_places=8, releg_places=12,
    releg_note="9th-24th enter a two-legged knockout play-off",
    # Kept for completeness; the cup pipeline reads its fixtures through
    # `model.europe`, which puts OUR committed file first (Risk 1).
    of_top="champions-league/master/{season}/cl",
    of_second="",
    backtest_from="2024-25",     # the two Swiss-format seasons are the honest holdout
)

#: Registry order is the order the site shows them in. Domestic leagues build
#: today; the Champions League appears in the manifest (ready:false until its
#: pipeline lands) so the site can already show the sixth entry.
LEAGUES: tuple[League, ...] = (PREMIER_LEAGUE, LA_LIGA, SERIE_A, BUNDESLIGA,
                               LIGUE_1, EREDIVISIE, PRIMEIRA_LIGA,
                               JUPILER_PRO_LEAGUE, CHAMPIONSHIP)

#: The five the model was calibrated on and whose backtest is quoted as the
#: headline. Kept separate from `LEAGUES` so anything that means "the big five"
#: says so rather than meaning "whatever the registry happens to hold".
BIG_FIVE: tuple[League, ...] = (PREMIER_LEAGUE, LA_LIGA, SERIE_A, BUNDESLIGA,
                                LIGUE_1)
EUROPEAN: tuple[League, ...] = (CHAMPIONS_LEAGUE,)
BY_SLUG: dict[str, League] = {lg.slug: lg for lg in LEAGUES + EUROPEAN}
DEFAULT = PREMIER_LEAGUE


def get(slug: str) -> League:
    try:
        return BY_SLUG[slug]
    except KeyError:
        raise KeyError(
            f"unknown league {slug!r}; known: {', '.join(BY_SLUG)}") from None
