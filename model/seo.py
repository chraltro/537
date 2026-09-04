"""The files that let anything except a person find this site.

Nine pages across nine competitions, and until now no `robots.txt`, no sitemap
and no per-club URL a link scraper could resolve. A static site is the easiest
kind in the world to index and this one was invisible.

Four things live here, all written at build time from the manifest so a new
competition appears in them without anybody remembering:

* `robots.txt` and `sitemap.xml` -- an index of every page that exists, per
  competition, with the date the forecast was last rebuilt. The stubs below are
  in it, and so is every rated-not-forecast league's projection page: a URL
  nothing links to and the sitemap omits does not exist as far as a crawler is
  concerned.
* Per-club stub pages. `team.html?t=arsenal` needs JavaScript to become
  anything, and no scraper runs JavaScript, so a shared club link showed the
  competition's card and a generic title. A tiny static page per club carries
  the right title, description and share card, and sends a browser onward.
* Per-fixture stub pages, the same idea for `matches.html?m=arsenal--chelsea`:
  one page per unplayed fixture in the next fortnight, carrying the model's
  probabilities in words, pruned when the fixture is played or the window
  moves past it.
* JSON-LD, so a search result can show a fixture rather than a URL.

Three rules every stub follows, learned the hard way:

* **The redirect is relative.** It used to be an absolute
  `https://chraltro.github.io/537/...`, which meant all 210 club pages walked a
  reader off any other origin -- a fork, a preview deploy, a local server --
  and onto a host they could not reach.
* **The canonical points at the stub itself.** Pointing it at
  `team.html?t=arsenal&lg=premier-league`, whose own canonical is `team.html`,
  formed a chain that collapsed 210 distinct pages into one URL.
* **The body says something.** A page whose only content is a redirect is a
  doorway page; these carry the club's or the fixture's actual numbers, so they
  are worth landing on when the redirect is refused or JavaScript is off.
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
SITE_PAGES = ("rankings.html", "compare.html", "projection.html", "week.html")

#: How far ahead a fixture gets its own page. Two weeks is one international
#: break's worth of fixtures: far enough that a link shared today still resolves
#: on matchday, near enough that the file count stays in the hundreds.
MATCH_DAYS = 14

MONTHS = ("January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December")


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _ord(n) -> str:
    """1st, 2nd, 3rd, 4th ... 11th, 21st, 24th.

    The club stubs read "projected to finish 1, on 73 points" for as long as
    they existed. The teens are the special case, not the twenties.
    """
    try:
        n = int(n)
    except (TypeError, ValueError):
        return str(n)
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def _pct(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if x >= 0.9995:
        return ">99%"
    if 0 < x < 0.005:
        return "<1%"
    return f"{x * 100:.0f}%"


def _date_words(s: str) -> str:
    """`2026-09-12` -> `Saturday 12 September 2026`, or the raw string."""
    try:
        d = dt.date.fromisoformat(str(s))
    except (TypeError, ValueError):
        return str(s)
    return f"{d:%A} {d.day} {MONTHS[d.month - 1]} {d.year}"


def write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _read(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def sitemap(ready: list[dict], default: str, stamp: str,
            extra: "list[tuple[str, str]] | None" = None) -> str:
    """Every reachable URL, with the build date as its last-modified.

    A competition that is not `ready` is left out rather than listed and served
    a not-yet-live page, which is the same rule the switcher follows. `extra`
    carries the URLs that are not one of the two page lists -- the club and
    fixture stubs, and the projection page of every league this site rates and
    does not forecast.
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
    urls += list(extra or [])
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


#: What a competition's three headline chances are called. The keys on the row
#: are the same three everywhere -- `title`, `ucl`, `releg` -- and mean
#: something different in each competition, which is exactly the mistake the
#: rest of the site takes care not to make.
STAKES = {
    "league": ("Title", "Qualify", "Relegated"),
    "promotion": ("Title", "Promoted", "Relegated"),
    "cup": ("Win it", "Top eight", "Eliminated"),
}


def _card(slug: str, club: str) -> str:
    """The club's own share card if it was drawn, the competition's otherwise."""
    if os.path.exists(os.path.join(SITE, "og", slug, f"{club}.png")):
        return f"{HOME}og/{slug}/{club}.png"
    return f"{HOME}og/{slug}.png"


def club_stub(club: dict, league: dict, season: str, rank: int,
              also: "list[dict] | None" = None) -> str:
    """A static page per club, for the things that do not run JavaScript.

    It carries the club's own title, description and share card, then sends a
    browser on to the real page. The body is readable on its own, because a stub
    that is blank without JavaScript has only moved the problem.

    `also` is the club's other ready competitions this season. Arsenal have two
    club pages that did not know the other existed, and on the Tuesday the
    Champions League starts that is the one thing a shared Premier League link
    should say.
    """
    name = club.get("name", club["id"])
    slug = league["slug"]
    stakes = STAKES.get(league.get("kind", "league"), STAKES["league"])
    stakes_line = (f"{stakes[0]} {_pct(club.get('title'))} · "
                   f"{stakes[1].lower()} {_pct(club.get('ucl'))} · "
                   f"{stakes[2].lower()} {_pct(club.get('releg'))}.")
    target = f"../team.html?t={club['id']}&amp;lg={slug}"
    canonical = f"{HOME}club/{club['id']}-{slug}.html"
    card = _card(slug, club["id"])
    others = [o for o in (also or []) if o.get("slug") != slug]
    elsewhere = ""
    if others:
        names = [o.get("name", o["slug"]) for o in others]
        joined = names[0] if len(names) == 1 else (
            ", ".join(names[:-1]) + " and " + names[-1])
        elsewhere = f" Also in the {joined} this season."
    desc = (f"{name} in the {season} {league['name']}: projected to finish "
            f"{_ord(rank)}, on {club.get('pts', 0):.0f} points.{elsewhere} "
            "Rating, finishing positions and every remaining fixture.")
    ld = json.dumps({
        "@context": "https://schema.org",
        "@type": "SportsTeam",
        "name": name,
        "sport": "Association football",
        "memberOf": [{"@type": "SportsOrganization", "name": league["name"]}]
                    + [{"@type": "SportsOrganization",
                        "name": o.get("name", o["slug"])} for o in others],
        "url": canonical,
    }, separators=(",", ":"))
    links = "".join(
        f'<li><a href="{_esc(club["id"])}-{_esc(o["slug"])}.html">'
        f'{_esc(name)} in the {_esc(o.get("name", o["slug"]))}</a></li>'
        for o in others)
    return f"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(name)} — {_esc(league['name'])} {_esc(season)} forecast</title>
<meta name="description" content="{_esc(desc)}">
<link rel="canonical" href="{_esc(canonical)}">
<meta property="og:title" content="{_esc(name)} — {_esc(league['name'])} forecast">
<meta property="og:description" content="{_esc(desc)}">
<meta property="og:type" content="website">
<meta property="og:url" content="{_esc(canonical)}">
<meta property="og:image" content="{_esc(card)}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="{_esc(card)}">
<link rel="stylesheet" href="../assets/style.css">
<script type="application/ld+json">{ld}</script>
<meta http-equiv="refresh" content="0; url={target}">
</head>
<body>
<main class="wrap" style="padding-top:60px">
  <h1>{_esc(name)}</h1>
  <p class="lede">{_esc(desc)}</p>
  <ul>
    <li>Projected finish: {_esc(_ord(rank))}, on {club.get('pts', 0):.0f} points.</li>
    <li>{_esc(stakes_line)}</li>
    {links}
  </ul>
  <p><a href="{target}">Open the full {_esc(name)} forecast →</a></p>
</main>
</body>
</html>
"""


def _match_words(m: dict, hn: str, an: str) -> str:
    """The model's probabilities, in a sentence."""
    line = (f"The model gives {hn} {_pct(m.get('ph'))}, the draw "
            f"{_pct(m.get('pd'))} and {an} {_pct(m.get('pa'))}")
    sc = m.get("sc")
    if isinstance(sc, (list, tuple)) and len(sc) == 2:
        line += (f"; the likeliest score is {sc[0]}–{sc[1]}"
                 + (f" at {_pct(m.get('scp'))}" if m.get("scp") else ""))
    return line + "."


def match_stub(m: dict, league: dict, season: str, names: dict,
               swing_line: str = "") -> str:
    """One unplayed fixture, as a page.

    `matches.html?m=home--away` opens the fixture's dialog, which is the link
    the site itself shares -- and which is blank to anything that does not run
    JavaScript. This is that dialog's contents as HTML.
    """
    h, a = m["h"], m["a"]
    hn, an = names.get(h, h), names.get(a, a)
    slug = league["slug"]
    target = f"../../matches.html?m={h}--{a}&amp;lg={slug}"
    canonical = f"{HOME}match/{slug}/{m['date']}-{h}-{a}.html"
    card = _card(slug, h)
    when = _date_words(m.get("date"))
    kick = f", {m['time']} kick-off" if m.get("time") else ""
    words = _match_words(m, hn, an)
    odds_line = (f"{hn} win {_pct(m.get('ph'))} · draw {_pct(m.get('pd'))} · "
                 f"{an} win {_pct(m.get('pa'))}")
    md = m.get("md")
    round_word = ("Matchday" if league.get("kind") == "cup" else "Matchweek")
    where = (f"{league['name']} {round_word.lower()} {md}" if md is not None
             else league["name"])
    desc = f"{hn} v {an}, {where}, {when}{kick}. {words}"
    ld = json.dumps({
        "@context": "https://schema.org",
        "@type": "SportsEvent",
        "name": f"{hn} v {an}",
        "startDate": m.get("date"),
        "eventStatus": "https://schema.org/EventScheduled",
        "sport": "Association football",
        "competitor": [{"@type": "SportsTeam", "name": hn},
                       {"@type": "SportsTeam", "name": an}],
        "superEvent": {"@type": "SportsOrganization", "name": league["name"]},
        "url": canonical,
    }, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(hn)} v {_esc(an)} — {_esc(league['name'])} prediction, {_esc(when)}</title>
<meta name="description" content="{_esc(desc)}">
<link rel="canonical" href="{_esc(canonical)}">
<meta property="og:title" content="{_esc(hn)} v {_esc(an)} — {_esc(league['name'])} prediction">
<meta property="og:description" content="{_esc(desc)}">
<meta property="og:type" content="article">
<meta property="og:url" content="{_esc(canonical)}">
<meta property="og:image" content="{_esc(card)}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="{_esc(card)}">
<link rel="stylesheet" href="../../assets/style.css">
<script type="application/ld+json">{ld}</script>
<meta http-equiv="refresh" content="0; url={target}">
</head>
<body>
<main class="wrap" style="padding-top:60px">
  <h1>{_esc(hn)} v {_esc(an)}</h1>
  <p class="lede">{_esc(where)}, {_esc(season)} · {_esc(when)}{_esc(kick)}</p>
  <ul>
    <li>{_esc(odds_line)}</li>
    <li>{_esc(words)}</li>
    {f"<li>{_esc(swing_line)}</li>" if swing_line else ""}
  </ul>
  <p><a href="{target}">Open the live forecast for {_esc(hn)} v {_esc(an)} →</a></p>
  <p><a href="../../club/{_esc(h)}-{_esc(slug)}.html">{_esc(hn)}</a> ·
     <a href="../../club/{_esc(a)}-{_esc(slug)}.html">{_esc(an)}</a></p>
</main>
</body>
</html>
"""


def _swing_line(m: dict, names: dict, kind: str) -> str:
    """What the fixture is worth to somebody who is not playing in it."""
    swings = m.get("swings") or []
    if not swings or not isinstance(swings[0], dict):
        return ""
    s = swings[0]
    try:
        pts = abs(float(s.get("swing"))) * 100
    except (TypeError, ValueError):
        return ""
    if pts < 1:
        return ""
    words = {"title": "winning the league", "ucl": "qualifying",
             "releg": "going down", "out": "going out",
             "qualify": "qualifying", "top": "a top place"}
    who = names.get(s.get("team"), s.get("team"))
    what = words.get(s.get("event"), str(s.get("event")))
    if kind == "cup" and s.get("event") == "title":
        what = "topping the league phase"
    return (f"The result moves {who}'s chance of {what} by "
            f"{pts:.0f} percentage points — the most this fixture carries.")


def match_stubs(out_dir: str, ready: list[dict], forecasts: dict,
                today: dt.date | None = None) -> list[tuple[str, str]]:
    """Write one page per unplayed fixture inside the window; prune the rest.

    Returns the sitemap rows for what now exists. `out_dir` is `site/data`,
    which is where the fixtures are: this reads the same `matches.json` the
    site does rather than asking the caller for a copy.
    """
    today = today or dt.date.today()
    hi = today + dt.timedelta(days=MATCH_DAYS)
    rows: list[tuple[str, str]] = []
    for lg in ready:
        slug = lg["slug"]
        fc = forecasts.get(slug) or {}
        names = {t["id"]: t.get("name", t["id"]) for t in fc.get("teams", [])}
        season = fc.get("season", "")
        doc = _read(os.path.join(out_dir, slug, "matches.json")) or {}
        matches = doc.get("matches") if isinstance(doc, dict) else None
        d_out = os.path.join(SITE, "match", slug)
        keep: set[str] = set()
        for m in matches or []:
            if not isinstance(m, dict) or m.get("played"):
                continue
            try:
                d = dt.date.fromisoformat(str(m.get("date")))
            except (TypeError, ValueError):
                continue
            if not (today <= d <= hi) or not m.get("h") or not m.get("a"):
                continue
            fname = f"{m['date']}-{m['h']}-{m['a']}.html"
            keep.add(fname)
            write(os.path.join(d_out, fname),
                  match_stub(m, lg, season, names,
                             _swing_line(m, names, lg.get("kind", "league"))))
            rows.append((f"{HOME}match/{slug}/{fname}", "0.5"))
        if os.path.isdir(d_out):
            for stale in os.listdir(d_out):
                if stale.endswith(".html") and stale not in keep:
                    try:
                        os.remove(os.path.join(d_out, stale))
                    except OSError:
                        pass
    return rows


def other_league_urls(out_dir: str) -> list[tuple[str, str]]:
    """`projection.html?league=<slug>` for every league rated and not forecast.

    Fifty-one competitions had a page each and no way in from outside: nothing
    links to them and the sitemap did not know they existed.
    """
    slugs: dict[str, None] = {}
    d = os.path.join(out_dir, "league")
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".json"):
                slugs.setdefault(f[:-5], None)
    manifest = _read(os.path.join(out_dir, "leagues.json")) or {}
    for key in ("projected", "rated"):
        for lg in manifest.get(key) or []:
            if isinstance(lg, dict) and lg.get("slug"):
                slugs.setdefault(lg["slug"], None)
    return [(f"{HOME}projection.html?league={s}", "0.4") for s in sorted(slugs)]


def build(out_dir: str, manifest: dict, forecasts: dict) -> dict:
    """Write robots, the sitemap and one stub per club and per fixture.

    Returns a small summary.
    """
    stamp = dt.date.today().isoformat()
    ready = [lg for lg in manifest["leagues"] if lg.get("ready")]

    # Cross-competition membership, computed here rather than read from
    # `clubindex.json`: that file is written after this step runs, so using it
    # would put yesterday's memberships on today's pages.
    elsewhere: dict[str, list[dict]] = {}
    for lg in ready:
        for club in (forecasts.get(lg["slug"]) or {}).get("teams", []):
            elsewhere.setdefault(club["id"], []).append(
                {"slug": lg["slug"], "name": lg["name"]})

    extra: list[tuple[str, str]] = []
    n = 0
    for lg in ready:
        fc = forecasts.get(lg["slug"])
        if not fc:
            continue
        for i, club in enumerate(fc.get("teams", []), 1):
            write(os.path.join(SITE, "club", f"{club['id']}-{lg['slug']}.html"),
                  club_stub(club, lg, fc.get("season", ""), i,
                            elsewhere.get(club["id"])))
            extra.append((f"{HOME}club/{club['id']}-{lg['slug']}.html", "0.5"))
            n += 1
    fixtures = match_stubs(out_dir, ready, forecasts)
    extra += fixtures
    extra += other_league_urls(out_dir)

    write(os.path.join(SITE, "robots.txt"), robots())
    write(os.path.join(SITE, "sitemap.xml"),
          sitemap(ready, manifest.get("default", "premier-league"), stamp,
                  extra))
    return {"leagues": len(ready), "stubs": n, "matches": len(fixtures),
            "urls": len(extra)}
