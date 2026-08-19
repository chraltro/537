"""Things that leave the site: calendar subscriptions and a change feed.

A static site cannot push. It can, however, publish files that other people's
software polls, and that is the only honest "notify me" a project with no
server and no mailing list can offer.

*Calendars.* One `.ics` per competition and per club. A calendar client refetches
the URL on its own schedule, so a fixture's description carries whatever the
last build believed -- win, draw and loss probabilities, the likeliest score and
whether anything rides on it -- and updates itself as the season moves.

*A change feed.* `recap.json` already works out which clubs the model changed
its mind about since roughly a week ago. Publishing that as JSON Feed and RSS
lets somebody follow the forecast rather than visit it. An item is written only
when something actually moved, because four items a day saying nothing is how a
feed gets unsubscribed from.
"""
from __future__ import annotations

import datetime as dt
import os

#: RFC 5545 asks clients not to refetch more often than this. Six hours matches
#: the build cadence exactly; asking for less would be asking for stale data,
#: asking for more would be asking other people's servers for nothing.
REFRESH = "PT6H"

#: Below this a "mover" is simulation noise, and the feed should stay quiet.
FEED_MIN_DELTA = 0.03


def _esc(s: str) -> str:
    """RFC 5545 text escaping."""
    return (str(s).replace("\\", "\\\\").replace(";", "\\;")
            .replace(",", "\\,").replace("\n", "\\n"))


def _fold(line: str) -> str:
    """RFC 5545 lines are at most 75 octets; continuations start with a space."""
    raw = line.encode("utf-8")
    if len(raw) <= 73:
        return line
    out, cur = [], b""
    for ch in line:
        b = ch.encode("utf-8")
        if len(cur) + len(b) > 72:
            out.append(cur.decode("utf-8"))
            cur = b" " + b
        else:
            cur += b
    out.append(cur.decode("utf-8"))
    return "\r\n".join(out)


def _stamp(d: dt.date, time: str | None) -> tuple[str, str]:
    """DTSTART/DTEND for one fixture.

    Kick-off times in the fixture feed are local wall-clock with no zone, and
    inventing one would be worse than saying so: a timed fixture is written as
    RFC 5545 *floating* local time, which every client renders in the reader's
    own zone. A fixture with no time at all becomes an all-day event.
    """
    if not time:
        nxt = d + dt.timedelta(days=1)
        return (f"DTSTART;VALUE=DATE:{d:%Y%m%d}", f"DTEND;VALUE=DATE:{nxt:%Y%m%d}")
    hh, _, mm = time.partition(":")
    try:
        start = dt.datetime(d.year, d.month, d.day, int(hh), int(mm))
    except ValueError:
        start = dt.datetime(d.year, d.month, d.day, 15, 0)
    end = start + dt.timedelta(hours=2)
    return (f"DTSTART:{start:%Y%m%dT%H%M%S}", f"DTEND:{end:%Y%m%dT%H%M%S}")


def calendar(matches: list[dict], meta: dict, *, title: str, uid_ns: str,
             team: str | None = None, round_label=None) -> str:
    """One iCalendar document. `matches` is the site's own `matches.json` rows."""
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = [
        "BEGIN:VCALENDAR", "VERSION:2.0",
        "PRODID:-//chraltro//537 football forecast//EN",
        "CALSCALE:GREGORIAN", "METHOD:PUBLISH",
        f"X-WR-CALNAME:{_esc(title)}",
        f"NAME:{_esc(title)}",
        f"REFRESH-INTERVAL;VALUE=DURATION:{REFRESH}",
        f"X-PUBLISHED-TTL:{REFRESH}",
    ]
    for m in matches:
        if team and team not in (m["h"], m["a"]):
            continue
        h = meta.get(m["h"], {}).get("name", m["h"])
        a = meta.get(m["a"], {}).get("name", m["a"])
        try:
            d = dt.date.fromisoformat(m["date"])
        except (TypeError, ValueError):
            continue
        start, end = _stamp(d, m.get("time"))
        if m.get("played"):
            summary = f"{h} {m['hg']}–{m['ag']} {a}"
            body = "Final score."
        else:
            summary = f"{h} v {a}"
            body = (f"{h} {round(m['ph'] * 100)}% · draw {round(m['pd'] * 100)}% · "
                    f"{a} {round(m['pa'] * 100)}%. "
                    f"Likeliest score {m['sc'][0]}–{m['sc'][1]} "
                    f"({round(m['scp'] * 100)}%). "
                    f"Expected goals {m['xgh']}–{m['xga']}.")
            if m.get("swings"):
                s = m["swings"][0]
                who = meta.get(s["team"], {}).get("name", s["team"])
                body += (f" Biggest swing: {who}, "
                         f"{round(abs(s['swing']) * 100)} points of it.")
        rnd = round_label(m.get("md")) if round_label else str(m.get("md"))
        uid = f"{m['h']}-{m['a']}-{m['date']}@{uid_ns}"
        out += [
            "BEGIN:VEVENT",
            _fold(f"UID:{uid}"),
            f"DTSTAMP:{now}",
            start, end,
            _fold(f"SUMMARY:{_esc(summary)}"),
            _fold(f"DESCRIPTION:{_esc(body)}"),
            _fold(f"CATEGORIES:{_esc(rnd)}"),
            "TRANSP:TRANSPARENT",
            "END:VEVENT",
        ]
    out.append("END:VCALENDAR")
    return "\r\n".join(out) + "\r\n"


# --------------------------------------------------------------------------
# Change feed
# --------------------------------------------------------------------------
_EVENT_WORDS = {"title": "title chance", "ucl": "qualification chance",
                "releg": "relegation risk"}


def _sentence(mv: dict, meta: dict) -> str:
    name = meta.get(mv["id"], {}).get("name", mv["id"])
    word = _EVENT_WORDS.get(mv["metric"], mv["metric"])
    verb = "rose" if mv["delta"] > 0 else "fell"
    return (f"{name}'s {word} {verb} from {round(mv['before'] * 100)}% to "
            f"{round(mv['after'] * 100)}%.")


def feed_items(leagues_data: list[dict]) -> list[dict]:
    """One item per league that actually moved, newest league data first.

    `leagues_data` is a list of `{slug, name, recap, meta, url}`.
    """
    items = []
    for d in leagues_data:
        recap = d.get("recap") or {}
        movers = [m for m in recap.get("movers", [])
                  if abs(m.get("delta", 0)) >= FEED_MIN_DELTA]
        if not movers:
            continue
        lines = [_sentence(m, d["meta"]) for m in movers[:5]]
        items.append({
            "id": f"{d['slug']}-{recap.get('asof')}",
            "url": d["url"],
            "title": f"{d['name']}: {len(movers)} club"
                     f"{'' if len(movers) == 1 else 's'} moved",
            "content_text": " ".join(lines),
            "date_published": f"{recap.get('asof')}T12:00:00Z",
            "tags": [d["name"]],
        })
    return items


def json_feed(items: list[dict], *, home: str, title: str) -> dict:
    return {
        "version": "https://jsonfeed.org/version/1.1",
        "title": title,
        "home_page_url": home,
        "feed_url": f"{home}feed.json",
        "description": "What the forecast changed its mind about.",
        "items": items,
    }


def rss(items: list[dict], *, home: str, title: str) -> str:
    def esc(s):
        return (str(s).replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;"))

    now = dt.datetime.now(dt.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")
    body = []
    for it in items:
        try:
            d = dt.date.fromisoformat(it["date_published"][:10])
            pub = d.strftime("%a, %d %b %Y 12:00:00 +0000")
        except (TypeError, ValueError):
            pub = now
        body.append(
            "<item>"
            f"<title>{esc(it['title'])}</title>"
            f"<link>{esc(it['url'])}</link>"
            f"<guid isPermaLink=\"false\">{esc(it['id'])}</guid>"
            f"<pubDate>{pub}</pubDate>"
            f"<description>{esc(it['content_text'])}</description>"
            "</item>")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<rss version="2.0"><channel>'
            f"<title>{esc(title)}</title>"
            f"<link>{esc(home)}</link>"
            "<description>What the forecast changed its mind about.</description>"
            f"<lastBuildDate>{now}</lastBuildDate>"
            + "".join(body) +
            "</channel></rss>\n")


def write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
