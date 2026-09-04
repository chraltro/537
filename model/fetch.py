"""Source downloads with an on-disk cache.

Everything comes from raw.githubusercontent.com, so the same code path runs
locally and on an Actions runner with no API keys and no secrets.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from . import leagues

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, ".cache")
UA = "football-forecast/1.0 (+https://github.com/chraltro/537)"


class SourceError(RuntimeError):
    """A source we depend on is unavailable. Loud on purpose: a forecast built
    from a partially-loaded season is worse than no forecast at all."""


def _cache_path(url: str) -> str:
    return os.path.join(CACHE, url.replace("://", "_").replace("/", "_"))


#: URLs that could not be reached this run. A host that is blocked or down
#: answers identically every time, and the corpus is rebuilt once per
#: competition, so without this a single unreachable feed costs its retries and
#: its backoff nine times over. Only optional sources go in: a required one
#: raises rather than returning, and is not something to remember.
_DEAD: set[str] = set()

#: ...and the same verdict, kept on disk between runs. `cProfile` over one
#: league put 456s of an 802s build inside `time.sleep`: ~152 unreachable
#: second-feed URLs, each paying `tries` attempts and the backoff between them
#: to be told what the first attempt already said. In-process memory only saves
#: the repeats *within* one build, and every scheduled build re-learned the
#: same 152 answers from scratch.
#:
#: The TTL is half a day rather than the six-hour content TTL: a host that
#: refused this morning has not usually come back by lunchtime, and a feed that
#: does come back is picked up by the next build after that. `--probe` clears
#: the file first, so the one command whose whole job is to ask the second
#: feeds a fresh question always does.
DEAD_TTL = 12 * 3600
_DEAD_FILE = os.path.join(CACHE, "dead-urls.json")
_DEAD_DISK: dict[str, float] | None = None


def _dead_disk() -> dict[str, float]:
    """The persisted dead-URL verdicts, still inside their TTL."""
    global _DEAD_DISK
    if _DEAD_DISK is None:
        try:
            with open(_DEAD_FILE, encoding="utf-8") as fh:
                raw = json.load(fh)
            now = time.time()
            _DEAD_DISK = {u: float(t) for u, t in raw.items()
                          if isinstance(t, (int, float)) and now - t < DEAD_TTL}
        except (OSError, ValueError, AttributeError):
            _DEAD_DISK = {}
    return _DEAD_DISK


def _remember_dead(url: str) -> None:
    disk = _dead_disk()
    disk[url] = time.time()
    try:
        os.makedirs(CACHE, exist_ok=True)
        with open(_DEAD_FILE, "w", encoding="utf-8") as fh:
            json.dump(disk, fh)
    except OSError:
        pass                      # a cache that cannot be written is not fatal


def forget_dead() -> None:
    """Drop every dead-URL verdict, in memory and on disk.

    For `--probe`, whose entire purpose is to find out what the second feeds
    say *now*: answering it from a cache of yesterday's refusals would make the
    check report its own memory.
    """
    global _DEAD_DISK
    _DEAD.clear()
    _DEAD_DISK = {}
    try:
        os.remove(_DEAD_FILE)
    except OSError:
        pass


def get(url: str, max_age: float = 6 * 3600, required: bool = True,
        tries: int = 4) -> str | None:
    """Fetch `url`, using a cached copy when it is younger than `max_age`.

    Falls back to a stale cache entry if the network fails, so a transient
    GitHub blip degrades to slightly-old data rather than a broken site.

    `tries` exists for the second feeds. A host that is blocked outright answers
    immediately and identically every time, so four attempts with a backoff
    between them spends fifteen seconds to learn what the first one said; the
    GitHub sources keep the full four because their failures really are
    transient, and the second feeds pass 1.

    An optional source that has already refused -- this run or, through
    `_dead_disk`, within the last `DEAD_TTL` -- is not asked again. A *required*
    source never takes that path: it raises rather than returning None, so it is
    never remembered, and turning a missing required feed into a silent None
    would replace a loud failure with an empty forecast.
    """
    path = _cache_path(url)
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < max_age:
        return open(path, encoding="utf-8").read()
    # A remembered refusal only ever short-circuits an optional source. A
    # required one raises when it fails, so it is never remembered in the first
    # place -- and if one were, silently returning None for it would turn a
    # loud missing feed into an empty forecast.
    if not required and (url in _DEAD or url in _dead_disk()):
        return None

    last: Exception | None = None
    for attempt in range(max(1, tries)):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=45) as resp:
                text = resp.read().decode("utf-8", "replace")
            os.makedirs(CACHE, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            return text
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None          # a season that has not started yet
            last = exc
        except Exception as exc:      # noqa: BLE001 - network is genuinely varied
            last = exc
        if attempt + 1 < max(1, tries):
            # No backoff after the last attempt: there is nothing left to back
            # off *for*. With `tries=1` this used to sleep a second before
            # giving up, which across ~152 dead second-feed URLs was 152
            # seconds of a build spent waiting for nothing.
            time.sleep(2 ** attempt)

    if os.path.exists(path):
        print(f"  ! {url} unreachable, using cached copy")
        return open(path, encoding="utf-8").read()
    if required:
        raise SourceError(f"cannot fetch {url}: {last}")
    _DEAD.add(url)
    _remember_dead(url)
    print(f"  ! optional source unavailable: {url}")
    return None


def results_csv(league: leagues.League, code: str, required: bool = True) -> str | None:
    """One season of results from the mirror. `code` is a code such as '2526'."""
    return get(league.fd_csv_url(code), required=required)


def fixtures_text(league: leagues.League, season: str, tier: str = "top",
                  required: bool = True) -> str | None:
    """One openfootball league file. `tier` is 'top' or 'second'."""
    return get(league.of_url(season, tier), required=required)


# -- Premier League shorthands, kept so callers with no league in hand still work
def premier_league_csv(code: str, required: bool = True) -> str | None:
    return results_csv(leagues.PREMIER_LEAGUE, code, required=required)


def openfootball(season: str, tier_file: str, required: bool = True) -> str | None:
    """`tier_file` is '1-premierleague' or '2-championship'."""
    tier = "second" if tier_file.startswith("2-") else "top"
    return fixtures_text(leagues.PREMIER_LEAGUE, season, tier, required=required)
