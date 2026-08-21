"""Source downloads with an on-disk cache.

Everything comes from raw.githubusercontent.com, so the same code path runs
locally and on an Actions runner with no API keys and no secrets.
"""
from __future__ import annotations

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


def get(url: str, max_age: float = 6 * 3600, required: bool = True,
        tries: int = 4) -> str | None:
    """Fetch `url`, using a cached copy when it is younger than `max_age`.

    Falls back to a stale cache entry if the network fails, so a transient
    GitHub blip degrades to slightly-old data rather than a broken site.

    `tries` exists for the second feeds. A host that is blocked outright answers
    immediately and identically every time, so four attempts with a backoff
    between them spends fifteen seconds to learn what the first one said; the
    GitHub sources keep the full four because their failures really are
    transient.
    """
    path = _cache_path(url)
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < max_age:
        return open(path, encoding="utf-8").read()

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
        time.sleep(2 ** attempt)

    if os.path.exists(path):
        print(f"  ! {url} unreachable, using cached copy")
        return open(path, encoding="utf-8").read()
    if required:
        raise SourceError(f"cannot fetch {url}: {last}")
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
