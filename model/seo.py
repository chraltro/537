"""The files that let anything except a person find this site.

Nine pages across nine competitions, and until now no `robots.txt`, no sitemap
and no per-club URL a link scraper could resolve. A static site is the easiest
kind in the world to index and this one was invisible.

Three things live here, all written at build time from the manifest so a new
competition appears in them without anybody remembering:

* `robots.txt` and `sitemap.xml` -- an index of every page that exists, per
  competition, with the date the forecast was last rebuilt.
* Per-club stub pages. `team.html?t=arsenal` needs JavaScript to become
  anything, and no scraper runs JavaScript, so a shared club link showed the
  competition's card and a generic title. A tiny static page per club carries
  the right title, description and share card, and sends a browser onward.
* JSON-LD, so a search result can show a fixture rather than a URL.
"""
from __future__ import annotations

import datetime as dt
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(HERE, "site")
HOME = "https://chraltro.github.io/537/"

#: Pages that exist once per competition, with `?lg=`.
LEAGUE_PAGES = ("index.html", "matches.html", "team.html", "races.html",
                "simulator.html", "review.html", "method.html")
#: Pages that are not scoped to a competition at all.
SITE_PAGES = ("rankings.html", "compare.html", "projection.html")


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def sitemap(ready: list[dict], default: str, stamp: str) -> str:
    """Every reachable URL, with the build date as its last-modified.

    A competition that is not `ready` is left out rather than listed and served
    a not-yet-live page, which is the same rule the switcher follows.
    """
    urls: list[tuple[str, str]] = []
    for page in SITE_PAGES:
        urls.append((f"{HOME}{page}", "0.8"))
    for lg in ready:
        for page in LEAGUE_PAGES:
            q = "" if lg["slug"] == default else f"?lg={lg['slug']}"
            pri = "1.0" if page == "index.html" and lg["slug"] == default else (
                "0.9" if page == "index.html" else "0.6")
            urls.append((f"{HOME}{page}{q}", pri))
    body = "".join(
        f"<url><loc>{_esc(u)}</loc><lastmod>{stamp}</lastmod>"
        f"<changefreq>daily</changefreq><priority>{p}</priority></url>"
        for u, p in urls)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            + body + "</urlset>\n")


def robots() -> str:
    return ("User-agent: *\n"
            "Allow: /\n"
            "# The share cards are for link previews, not for crawling.\n"
            "Disallow: /537/og/\n"
            f"Sitemap: {HOME}sitemap.xml\n")


def club_stub(club: dict, league: dict, season: str, rank: int) -> str:
    """A static page per club, for the things that do not run JavaScript.

    It carries the club's own title, description and share card, then sends a
    browser on to the real page. The body is readable on its own, because a stub
    that is blank without JavaScript has only moved the problem.
    """
    name = club.get("name", club["id"])
    slug = league["slug"]
    target = f"{HOME}team.html?t={club['id']}&lg={slug}"
    card = f"{HOME}og/{slug}/{club['id']}.png"
    desc = (f"{name} in the {season} {league['name']}: projected to finish "
            f"{rank}, on {club.get('pts', 0):.0f} points. Rating, finishing "
            "positions and every remaining fixture.")
    ld = json.dumps({
        "@context": "https://schema.org",
        "@type": "SportsTeam",
        "name": name,
        "sport": "Association football",
        "memberOf": {"@type": "SportsOrganization", "name": league["name"]},
        "url": target,
    }, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(name)} — {_esc(league['name'])} {_esc(season)} forecast</title>
<meta name="description" content="{_esc(desc)}">
<link rel="canonical" href="{_esc(target)}">
<meta property="og:title" content="{_esc(name)} — {_esc(league['name'])} forecast">
<meta property="og:description" content="{_esc(desc)}">
<meta property="og:type" content="website">
<meta property="og:image" content="{_esc(card)}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="{_esc(card)}">
<link rel="stylesheet" href="../assets/style.css">
<script type="application/ld+json">{ld}</script>
<meta http-equiv="refresh" content="0; url={_esc(target)}">
</head>
<body>
<main class="wrap" style="padding-top:60px">
  <h1>{_esc(name)}</h1>
  <p class="lede">{_esc(desc)}</p>
  <p><a href="{_esc(target)}">Open the full {_esc(name)} forecast →</a></p>
</main>
</body>
</html>
"""


def build(out_dir: str, manifest: dict, forecasts: dict) -> dict:
    """Write robots, the sitemap and one stub per club. Returns a small summary."""
    stamp = dt.date.today().isoformat()
    ready = [lg for lg in manifest["leagues"] if lg.get("ready")]
    write(os.path.join(SITE, "robots.txt"), robots())
    write(os.path.join(SITE, "sitemap.xml"),
          sitemap(ready, manifest.get("default", "premier-league"), stamp))
    n = 0
    for lg in ready:
        fc = forecasts.get(lg["slug"])
        if not fc:
            continue
        for i, club in enumerate(fc.get("teams", []), 1):
            write(os.path.join(SITE, "club", f"{club['id']}-{lg['slug']}.html"),
                  club_stub(club, lg, fc.get("season", ""), i))
            n += 1
    return {"leagues": len(ready), "stubs": n}
