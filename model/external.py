"""Sources that do not live on GitHub, and the gate that decides whether to use one.

Until now every byte this project read came from `raw.githubusercontent.com`.
That rule bought something real -- one host, one failure mode, no keys, and a
sandbox that can verify any source before it ships -- and it cost 343 clubs
across 24 competitions, whose ratings stopped at 2024-25 because openfootball
stopped publishing them and no other GitHub feed picked them up.

Two sources outside GitHub cover the gap, and they cover different parts of it:

* **football-data.co.uk** publishes Poland, Romania and Switzerland as CSV, free,
  no key, updated through the season. Three of the 24.
* **Wikipedia** carries a results grid for all 24, including the twenty-one --
  Gibraltar, San Marino, Andorra, Kosovo, Montenegro and the rest -- that no
  bulk feed anywhere reaches.

The problem this module exists to solve is that the machine which writes the
reader cannot reach either host. The development sandbox allows GitHub and
denies everything else, so the project's own rule -- *verify a source by
fetching it* -- cannot be satisfied where the code is written. It can be
satisfied on the Actions runner, which has open egress.

So a source is armed by evidence rather than by assertion, and the evidence is
stronger than "it downloaded".

The overlap test
----------------
Every second feed here covers a league this project already carries; what it
adds is the seasons after the GitHub feed stopped. That leaves an overlap: at
least one season both feeds describe. The overlap is the whole verification.

1. **Is it the same competition?** Line the two feeds up on the overlap season
   and compare scores fixture by fixture. Two independent sources agreeing on
   three hundred results is not a coincidence, and it is the only check that
   catches the genuinely dangerous failure: a correctly-parsed file of the wrong
   league, or of the right league's second division.

2. **Is it the same set of clubs?** A club name in the overlap that does not
   resolve is a second spelling of a club already held -- "Rakow" for "Raków
   Częstochowa" -- because we know from the GitHub feed exactly who played that
   season. Those block the league until an alias is written, since minting an id
   for one puts a duplicate club in the global ranking with half a record.

3. **Which clubs are genuinely new?** A name that appears only in seasons past
   the overlap is a promoted club that the GitHub feed shut down too early to
   see. Minting an id for those is correct, so they are allowed and listed.

A source that cannot be lined up against a season we already know does not arm.
That is why the Wikipedia reader fetches one season more than it needs: the
extra season is not data, it is the proof.
"""
from __future__ import annotations

import collections
import datetime as dt
import difflib
from dataclasses import dataclass, field

from .parse import Match, TeamRegistry, normalise

#: How much of the overlap season the two feeds must agree on. Not 1.0: feeds
#: disagree about awarded forfeits and about the occasional abandoned match
#: replayed behind closed doors, and one such row should not veto a season.
#: Well below this and the two files are not describing the same competition.
MIN_AGREEMENT = 0.95

#: And enough fixtures to make that ratio mean something. A 12-club league plays
#: 132, a 20-club league 380, so sixty is a comfortable floor that still refuses
#: a season the source only half carries.
MIN_OVERLAP_MATCHES = 60

#: A feed that parses but carries nothing recent is a working parser pointed at
#: an abandoned file. Loose enough for a summer league between seasons, tight
#: enough to catch a URL that quietly became a redirect to last year.
FRESH_DAYS = 400


@dataclass
class Verdict:
    """What a probe learned. Everything here is printable, because on a runner
    this is the only account anyone gets of why a league did or did not arm."""

    source: str
    assoc: str                            # UEFA code, for the build log
    league: str                           # the competition's name, for the site
    ok: bool
    reason: str = ""                      # plain English; this one is published
    detail: str = ""                      # the exception, for whoever is debugging
    matches: int = 0
    latest: str | None = None
    clubs: int = 0
    overlap_season: str | None = None
    compared: int = 0
    agreed: int = 0
    unresolved: tuple[str, ...] = ()      # blocks: a second spelling of a club we hold
    new_clubs: tuple[str, ...] = ()       # allowed: promoted since the old feed stopped

    @property
    def agreement(self) -> float:
        return self.agreed / self.compared if self.compared else 0.0

    #: A source that was probed to find out what it would do, and is not used.
    #: Carried on the verdict so the log and the site say so rather than showing
    #: a green tick beside a league nothing reads.
    watching: bool = False

    def _new(self) -> str:
        """Clubs this feed adds to the ranking, by name. Printed rather than
        counted: minting a club id is the one irreversible thing a second feed
        does, and 'five new clubs' is not something anyone can check."""
        return (f", {len(self.new_clubs)} new club(s): "
                + ", ".join(self.new_clubs)) if self.new_clubs else ""

    def line(self) -> str:
        if self.ok:
            mark = "·" if self.watching else "✓"
            what = "would arm -- " if self.watching else ""
            return (f"  {mark} {self.source}/{self.assoc}: {what}{self.matches} "
                    f"matches to {self.latest}, agrees on {self.agreed}/"
                    f"{self.compared} of {self.overlap_season}{self._new()}")
        bits = [self.reason]
        if self.unresolved:
            # Every one of them, not the first few. This line is the only thing
            # a runner tells anyone about why a league did not arm, and the
            # answer is an alias per name: a list that stops at six sends the
            # next build round-tripping for the seventh.
            bits.append(", ".join(self.unresolved))
        if self.detail:
            bits.append(self.detail)
        return f"  ✗ {self.source}/{self.assoc}: " + " -- ".join(bits)

    def as_json(self) -> dict:
        return {"source": self.source, "assoc": self.assoc, "league": self.league,
                "ok": self.ok, "reason": self.reason, "detail": self.detail,
                "matches": self.matches, "latest": self.latest,
                "clubs": self.clubs, "overlap_season": self.overlap_season,
                "compared": self.compared, "agreed": self.agreed,
                "unresolved": list(self.unresolved), "new_clubs": list(self.new_clubs),
                "watching": self.watching}


@dataclass
class ExternalSource:
    """One league from one host outside GitHub.

    `load` fetches and parses and returns matches with raw club names still in
    them; resolution happens here, so that every source is gated by the same
    rule and no reader can accidentally skip it.
    """

    source: str                 # host-level name, e.g. 'football-data.co.uk'
    assoc: str                  # UEFA three-letter code: POL, ROU, SUI
    league: str                 # human name, for the log and coverage.json
    group: str                  # competition group id, must match europe.py's
    load: object                # callable(reg) -> list[tuple[Match, str, str]]
    note: str = ""              # what the source does and does not carry
    #: False for a source being watched rather than used. A candidate is
    #: fetched, parsed and judged exactly like any other -- the runner prints
    #: its verdict beside the rest -- and then contributes nothing whatever it
    #: says. That is the whole way a Wikipedia league gets added: the probe
    #: reports what would happen, the club names it cannot resolve are written
    #: down as aliases, and only when a green verdict is on the record does
    #: anyone arm it. A guess about an article's contents is not evidence, and
    #: this flag is what keeps the guess out of the ranking while it is checked.
    contributes: bool = True
    #: Names the twin check flags that are genuinely a club of their own. Every
    #: one of these is a decision someone made after reading a refusal, which is
    #: the point: the check is deliberately eager, and this is where the answer
    #: goes when the eager reading is wrong.
    distinct: frozenset[str] = field(default=frozenset(), repr=False)
    verdict: Verdict | None = field(default=None, repr=False)
    allow_new: frozenset[str] = field(default=frozenset(), repr=False)
    #: Seasons where the GitHub feed has a stub and this source has the season.
    #: The caller drops the stub, or the two are added together and every match
    #: the stub holds is counted twice.
    superseded: frozenset[str] = field(default=frozenset(), repr=False)

    @property
    def armed(self) -> bool:
        return (self.contributes
                and self.verdict is not None and self.verdict.ok)


def _score_counts(matches: list[Match]) -> "collections.Counter":
    """How many times the trusted feed records each `(season, home, away, hg, ag)`.

    A counter and not a dict, because a fixture is not unique. Romania's Liga I
    splits into a championship round after the regular season and Switzerland's
    twelve clubs play each other three times, so the same pair meets twice at
    the same ground in one season. Keyed as a dict, the second meeting would
    overwrite the first, and every first meeting would then read as a
    disagreement: the two leagues after Poland both failed this way, and both of
    their feeds were correct.
    """
    out: collections.Counter = collections.Counter()
    for m in matches:
        if m.played and m.hg is not None and m.ag is not None:
            out[(m.season, m.home, m.away, m.hg, m.ag)] += 1
    return out


#: How alike two club names have to read before one is treated as a second
#: spelling of the other rather than a club in its own right. Measured against
#: the pairs this actually has to separate: "chornomorets" against
#: "chernomorets odessa" scores above it, and "dinamo minsk" against "dinamo
#: brest" -- two real clubs a hundred miles apart -- scores well below.
TWIN_RATIO = 0.85


def _twin(name: str, held: set[str], reg: TeamRegistry) -> str | None:
    """The club in `held` that `name` is probably another spelling of.

    Two ways a feed shortens a name, both of them here. It drops a word --
    "Obolon" for "FK Obolon", "Zorya" for "Zorya Lugansk" -- which is a token
    subset. Or it transliterates the same Cyrillic differently, which no amount
    of token matching reaches and a similarity ratio does.
    """
    key = normalise(name)
    toks = set(key.split())
    if not toks:
        return None
    best, score = None, 0.0
    for other in held:
        okey = normalise(other)
        if okey == key:
            continue
        otoks = set(okey.split())
        if toks < otoks or otoks < toks:
            return other
        r = difflib.SequenceMatcher(None, key, okey).ratio()
        if r > score:
            best, score = other, r
    return best if score >= TWIN_RATIO else None


def probe(src: ExternalSource, reg: TeamRegistry, existing: list[Match],
          *, today: dt.date | None = None) -> Verdict:
    """Fetch, parse, line up against a season we already know, judge.

    Never raises: a source that cannot be reached is a source we do not use, not
    a build that fails.
    """
    v = Verdict(src.source, src.assoc, src.league, False,
                watching=not src.contributes)
    src.allow_new = frozenset()
    try:
        rows = src.load(reg)                       # type: ignore[operator]
    except Exception as exc:                       # noqa: BLE001
        # Plain English in `reason` because it is published on the method page;
        # the exception goes in `detail`, where the build log will show it.
        v.reason = "could not be fetched"
        v.detail = f"{type(exc).__name__}: {exc}"
        src.verdict = v
        return v

    if not rows:
        v.reason = "reachable but no played matches parsed"
        src.verdict = v
        return v

    theirs_by_season: dict[str, list[tuple[Match, str, str]]] = {}
    names: set[str] = set()
    latest = None
    for m, home_raw, away_raw in rows:
        theirs_by_season.setdefault(m.season, []).append((m, home_raw, away_raw))
        names.update((home_raw, away_raw))
        if latest is None or m.date > latest:
            latest = m.date
    v.matches, v.clubs = len(rows), len(names)
    v.latest = latest.isoformat() if latest else None

    # -- 1. an overlap season, and enough of it to mean anything ------------
    ours = _score_counts(existing)
    our_seasons = {key[0] for key in ours}
    our_fixtures = {key[:3] for key in ours}
    shared = sorted(our_seasons & set(theirs_by_season), reverse=True)
    if not shared:
        v.reason = ("no season in common with the feed we already trust, so "
                    "nothing here can be checked against anything")
        src.verdict = v
        return v
    # The newest shared season we hold a real season's worth of, not simply the
    # newest we hold anything of. openfootball opens a file when a season starts
    # and then, for the leagues that went quiet, stops: its 2025 Norway file has
    # 44 matches of 240 and its 2025 Belarus file has 8. Anchoring on one of
    # those fails the thinness test below, and worse, reads every club promoted
    # into that season as a misspelling of a club we hold -- because the rule
    # that makes an unresolved name a misspelling is that we know who played
    # that season, and in a stub we do not.
    counts: dict[str, int] = {}
    for key, n in ours.items():
        counts[key[0]] = counts.get(key[0], 0) + n
    full = [s for s in shared if counts.get(s, 0) >= MIN_OVERLAP_MATCHES]
    season = full[0] if full else shared[0]
    v.overlap_season = season

    # -- 2. names in the overlap must all be clubs we already hold ----------
    overlap_names = {n for _, h, a in theirs_by_season[season] for n in (h, a)}
    blocking = sorted(n for n in overlap_names if reg.known(n) is None)
    if blocking:
        v.unresolved = tuple(blocking)
        v.reason = (f"{len(blocking)} club name(s) in {season} do not resolve, and "
                    f"{season} is a season we already hold, so these are spellings "
                    "rather than new clubs")
        src.verdict = v
        return v

    # -- 2b. and no alias may fold two clubs into one ----------------------
    # An alias is written by hand and the way one goes wrong is by being too
    # broad: "Zaglebie" is Lubin while Lubin is the only Zagłębie in the league,
    # and becomes wrong the season Sosnowiec comes up. What that looks like from
    # here is unmistakable and cannot happen any other way -- the two clubs'
    # fixture against each other becomes a club playing itself.
    #
    # Two spellings of one club inside a season is NOT that, and refusing it
    # was wrong: football-data.co.uk writes both "Dinamo Bucuresti" and "Dinamo
    # Bucureşti" in the same file, and they are the same club by any reading.
    # Only the self-fixture is evidence, so only the self-fixture refuses.
    for _, home_raw, away_raw in rows:
        h, a = reg.known(home_raw), reg.known(away_raw)
        if h is not None and h == a:
            v.reason = (f"{home_raw!r} and {away_raw!r} are both read as "
                        f"{reg.display(h)}, which would have the club playing "
                        "itself, so one of the aliases covers two clubs")
            src.verdict = v
            return v

    # -- 3. do the two feeds agree about that season? -----------------------
    left = collections.Counter(ours)
    for m, home_raw, away_raw in theirs_by_season[season]:
        fixture = (season, reg.known(home_raw), reg.known(away_raw))
        if fixture not in our_fixtures:
            continue                               # a fixture we never had
        v.compared += 1
        key = fixture + (m.hg, m.ag)
        if left[key] > 0:
            left[key] -= 1                         # each result matched once only
            v.agreed += 1
    if v.compared < MIN_OVERLAP_MATCHES:
        v.reason = (f"only {v.compared} fixture(s) of {season} line up against the "
                    f"feed we already trust, below the {MIN_OVERLAP_MATCHES} needed "
                    "to call it the same competition")
        src.verdict = v
        return v
    if v.agreement < MIN_AGREEMENT:
        v.reason = (f"agrees on only {v.agreed}/{v.compared} scores in {season}; "
                    "this is not the competition it claims to be")
        src.verdict = v
        return v

    # -- 4. clubs seen only after the overlap are promotions, and allowed ---
    newer = {n for s, rowset in theirs_by_season.items() if s > season
             for _, h, a in rowset for n in (h, a)}
    v.new_clubs = tuple(sorted(n for n in newer - overlap_names if reg.known(n) is None))

    # Unless one of them is a club we already have under a longer name. The
    # overlap test cannot see this: it checks the season we hold, and the
    # mistake happens in the seasons we do not. Wikipedia's Ukrainian articles
    # write "Chornomorets Odesa" in one season and "Chornomorets" in the next,
    # and minting the second is how a club ends up in the ranking twice, each
    # copy with half a record and neither one right.
    seasons_of: dict[str, set[str]] = {}
    for szn, rowset in theirs_by_season.items():
        for _, h, a in rowset:
            for raw in (h, a):
                seasons_of.setdefault(raw, set()).add(szn)
    twins = []
    for name in v.new_clubs:
        if name in src.distinct:
            continue
        held = _twin(name, overlap_names, reg)
        # Unless the two turn up in the same season, which settles it: no feed
        # writes a club under two names in one table, so a season holding both
        # 'Poltava' and 'Vorskla Poltava' is a season with two clubs in it.
        if held and seasons_of.get(name, set()) & seasons_of.get(held, set()):
            held = None
        if held:
            twins.append(f"{name!r} looks like {held!r}")
    if twins:
        v.new_clubs = ()
        v.reason = ("; ".join(twins) + " -- a club this league already has under "
                    "another name, so an alias is needed before it is minted")
        src.verdict = v
        return v
    src.allow_new = frozenset(v.new_clubs)

    # -- 5. and it has to actually be current -------------------------------
    ref = today or dt.date.today()
    if latest is None or (ref - latest).days > FRESH_DAYS:
        v.reason = f"latest match is {v.latest}, older than {FRESH_DAYS} days"
        src.verdict = v
        return v

    v.ok, v.reason = True, "armed"
    src.verdict = v
    return v


def matches(src: ExternalSource, reg: TeamRegistry,
            existing: list[Match] | None = None) -> list[Match]:
    """The matches an armed source contributes, with ids filled in.

    Where both feeds describe a season, the GitHub one wins: it is the one with
    dates on it and the one whose club names defined the ids in the first place.

    Except where the GitHub feed's copy is a stub. openfootball opens a file
    when a season kicks off and then, for the leagues that went quiet, stops --
    its 2025 Norway file holds 44 matches of 240 -- and letting 44 matches shut
    out 240 is how a league stays a year stale beside a feed that has the whole
    thing. Those seasons are named in `src.superseded`, and the caller drops the
    stub before adding these.

    Calling this on a source that did not arm returns nothing rather than
    raising, so the pipeline reads the same whether a probe passed or failed.
    """
    if not src.armed:
        return []
    counts: dict[str, int] = {}
    for m in (existing or []):
        if m.played:
            counts[m.season] = counts.get(m.season, 0) + 1
    rows = list(src.load(reg))                     # type: ignore[operator]
    theirs: dict[str, int] = {}
    for m, _h, _a in rows:
        theirs[m.season] = theirs.get(m.season, 0) + 1
    have = {s for s, n in counts.items()
            if n >= MIN_OVERLAP_MATCHES or n >= theirs.get(s, 0)}
    src.superseded = frozenset(s for s in counts if s not in have)
    out: list[Match] = []
    for m, home_raw, away_raw in rows:
        if m.season in have:
            continue
        ids = []
        for raw in (home_raw, away_raw):
            tid = reg.known(raw)
            if tid is None and raw in src.allow_new:
                tid = reg.resolve(raw)             # a promotion, blessed by the probe
            ids.append(tid)
        if ids[0] is None or ids[1] is None:
            continue
        m.home, m.away, m.comp = ids[0], ids[1], src.group
        out.append(m)
    return out
