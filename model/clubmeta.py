"""Facts about a club that no result can tell you: when it was founded, and where.

`openfootball/clubs` is a set of plain-text club registers, one per country,
listing a club's name, the year it was founded, its stadium, its city, and every
alias anybody writes it under. The aliases are the reason it is worth reading:
they are how "Bayern Munich" in the results feed becomes "Bayern München" in the
register, which is a problem this pipeline otherwise has to solve by hand.

Two things it carries are deliberately not used.

*Stadiums.* The England register has West Ham at the Boleyn Ground and Tottenham
at White Hart Lane -- grounds they left in 2016 and 2017. The field is not
maintained, and a wrong stadium is worse than no stadium, so it is parsed and
thrown away rather than published.

*Crests.* Not in this feed and not wanted: they are trademarks, and this site
does not carry them.

Coverage is 91% of the clubs the site forecasts, measured. The gaps are the
clubs from countries with no register here -- the Champions League brings in
Norway, Azerbaijan, Kazakhstan -- and every consumer treats a missing club as
missing rather than as an error.
"""
from __future__ import annotations

import json
import os
import re

from . import fetch
from .parse import normalise

RAW = "https://raw.githubusercontent.com/openfootball/clubs/master/europe"

#: One register per country the site has a competition in, plus Wales, whose
#: clubs play in the English pyramid and whose register England's file points at
#: by name ("note: see wal.txt for teams from wales").
REGISTERS: dict[str, str] = {
    "England": "england/eng.clubs.txt",
    "Wales": "wales/wal.clubs.txt",
    "Spain": "spain/es.clubs.txt",
    "Italy": "italy/it.clubs.txt",
    "Germany": "germany/de.clubs.txt",
    "France": "france/fr.clubs.txt",
    "Netherlands": "netherlands/nl.clubs.txt",
    "Portugal": "portugal/pt.clubs.txt",
    "Belgium": "belgium/be.clubs.txt",
}

_YEAR = re.compile(r"^\d{4}$")
#: A trailing "(District)" on a city is more precision than a club page wants:
#: "London (Fulham)" is London.
_DISTRICT = re.compile(r"\s*\([^)]*\)\s*$")
#: Belgium's register writes a city as a hierarchy -- "Brugge › West-Vlaanderen
#: › Vlaanderen". The first segment is the city; the rest is the province.
_HIER = "\u203a"


def parse(text: str, country: str) -> list[dict]:
    """One register file into `[{name, founded, city, country, aliases}]`.

    The format is a club line, then any number of alias lines beginning `|`:

        Arsenal FC, 1886, @ Emirates Stadium, London (Highbury)   ## Greater London
          | Arsenal | FC Arsenal
          | Arsenal Football Club

    Everything after `##` is the maintainer's own note and is dropped. A field
    beginning `@` is the stadium, which is read only so that the field after it
    is understood to be the city.
    """
    out: list[dict] = []
    cur: dict | None = None
    for raw in text.splitlines():
        line = raw.split("##")[0].rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith(("=", "#")):
            continue
        if stripped.startswith("|"):
            if cur:
                cur["aliases"] += [a.strip() for a in stripped.lstrip("|").split("|")
                                   if a.strip()]
            continue
        parts = [p.strip() for p in line.split(",")]
        cur = {"name": parts[0], "founded": None, "city": None,
               "country": country, "aliases": [parts[0]]}
        for p in parts[1:]:
            if _YEAR.fullmatch(p):
                cur["founded"] = int(p)
            elif p.startswith("@"):
                continue                      # stadium: see the module docstring
            elif p and cur["city"] is None:
                city = p.split(_HIER)[0]
                cur["city"] = _DISTRICT.sub("", city).strip() or None
        out.append(cur)
    return out


def index(*, quiet: bool = True) -> dict[str, dict]:
    """Every alias in every register, normalised, pointing at its club.

    A name claimed by two registers keeps the first: the order in `REGISTERS`
    puts a club's own country ahead of a neighbour that lists it as a rival.
    """
    idx: dict[str, dict] = {}
    for country, rel in REGISTERS.items():
        text = fetch.get(f"{RAW}/{rel}", max_age=7 * 86400, required=False)
        if not text:
            if not quiet:
                print(f"  · {country}: register unavailable, skipped")
            continue
        clubs = parse(text, country)
        for club in clubs:
            for alias in club["aliases"]:
                idx.setdefault(normalise(alias), club)
        if not quiet:
            print(f"  · {country}: {len(clubs)} clubs")
    return idx


def lookup(idx: dict[str, dict], name: str,
           aliases: list[str] | None = None) -> dict | None:
    """The register entry for one club, or None. Only facts, no aliases back.

    Tries every name this pipeline has ever seen the club under, not just its
    display name, because the two sides disagree in ways accent-folding cannot
    reach: this site writes "Bayern Munich" and the register writes "Bayern
    München", and Munich is a translation of München rather than a spelling of
    it. `data/team_meta.json` already carries both, so the bridge is a list this
    repository maintains anyway rather than a table of special cases.
    """
    club = None
    for candidate in [name] + list(aliases or []):
        club = idx.get(normalise(candidate))
        if club:
            break
    if not club or not (club["founded"] or club["city"]):
        return None
    out = {}
    if club["founded"]:
        out["founded"] = club["founded"]
    if club["city"]:
        out["city"] = club["city"]
    return out or None


def build(out_dir: str, meta: dict[str, dict], *, quiet: bool = True) -> dict:
    """Write `clubs.json`: `{club_id: {founded, city}}` for everything matched.

    `meta` is this repository's own club table, keyed by id, each entry carrying
    a `name` and the `aliases` every feed writes it under. A club with no
    register entry is simply absent, which is what every reader of this file
    expects -- there is no placeholder to mistake for a fact.
    """
    idx = index(quiet=quiet)
    got = {}
    for cid, club in sorted(meta.items()):
        found = lookup(idx, club.get("name", cid), club.get("aliases"))
        if found:
            got[cid] = found
    path = os.path.join(out_dir, "clubs.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"source": "openfootball/clubs",
                   "url": "https://github.com/openfootball/clubs",
                   "note": ("Founded year and city only. The register's stadium "
                            "field is years out of date and is not read."),
                   "clubs": got}, fh, indent=1)
    return {"matched": len(got), "of": len(meta)}
