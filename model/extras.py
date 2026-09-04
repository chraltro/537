"""The cross-competition outputs, written after every league has been built.

Everything here is a *post-pass*: it reads the JSON the run has just written
under `site/data` and writes new JSON beside it. Nothing re-simulates, nothing
fetches, and no league's problem is allowed to cost another league its file --
a missing or malformed input prints a line and that competition is skipped.

Four files, plus one calendar per fixture:

* `clubindex.json` -- one club, every competition it is in this season, with
  its three headline chances and its next five fixtures. The site has nine
  competitions and, until this file existed, no way to tell that the Arsenal on
  the Premier League page is the Arsenal starting the Champions League on
  Tuesday.
* `upcoming.json` -- the fortnight around today across every ready competition,
  in kick-off order, so "what is on this week and which of it matters" is one
  20KB fetch rather than ten of 300KB.
* `<slug>/recaps.json` -- the weekly review, archived. `recap.json` is
  overwritten every six hours; this keeps each completed matchweek's version of
  it, appended once and never rewritten, the same discipline `history.json`
  follows.
* `global_history.json` -- the pooled ranking, one column per ISO week, so a
  club's rating can be drawn as a line rather than a single number.
* `site/cal/<slug>/match/<date>-<h>-<a>.ics` -- one event, so "add to calendar"
  on a fixture stops subscribing the reader to a whole club's season.
"""
from __future__ import annotations

import datetime as dt
import json
import os

from . import feeds

#: `upcoming.json` covers two days back -- so a reader who missed the weekend
#: still sees it -- and twelve days forward, which always contains the next
#: round in every competition on the site, including a midweek European one.
WINDOW_BACK = 2
WINDOW_FWD = 12

#: How many fixtures `clubindex.json` carries per club per competition.
NEXT_FIXTURES = 5

#: A single event per unplayed fixture inside this many days.
MATCH_ICS_DAYS = 14

#: `global_history.json` is committed by CI every six hours for a season, so it
#: has a size budget rather than an appetite. 981 clubs x ~44 weekly columns is
#: 226KB of parallel arrays; the ranking's top 400 is 92KB and contains every
#: club in every competition this site forecasts.
MAX_HISTORY_CLUBS = 400
HISTORY_BUDGET = 100_000

#: The three headline events, in the order the site labels them. Cups write the
#: same three keys under different names, and the aliases are already in
#: `forecast.json`; the fallbacks are here for a file written before they were.
EVENT_KEYS = (("title", "p_win"), ("ucl", "p_top8"), ("out", "p_out"))


# ---------------------------------------------------------------- plumbing

def _read(path: str):
    """The parsed file, or None if it is missing, unreadable or corrupt."""
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write(path: str, doc) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    text = json.dumps(doc, separators=(",", ":"))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return len(text)


def _stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _date(s):
    try:
        return dt.date.fromisoformat(str(s))
    except (TypeError, ValueError):
        return None


def _sort_key(m: dict) -> tuple:
    """Chronological, with untimed fixtures last inside their own day -- which
    is also how the site orders them, because a kick-off nobody has announced
    yet is not at midnight."""
    return (str(m.get("date") or ""), str(m.get("time") or "99:99"))


def _round(x, n=4):
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return None


class Competition:
    """One ready competition's manifest entry and the two files it is read from."""

    def __init__(self, entry: dict, forecast: dict, matches: list[dict]):
        self.entry = entry
        self.slug = entry["slug"]
        self.name = entry.get("name", entry["slug"])
        self.kind = entry.get("kind", "league")
        self.fc = forecast
        self.matches = matches
        self.teams = [t for t in forecast.get("teams", []) if t.get("id")]
        self.names = {t["id"]: t.get("name", t["id"]) for t in self.teams}

    def table_pos(self) -> dict:
        """Current league position per club, or an empty map before kick-off.

        Points, then goal difference, then goals scored -- the tie-break every
        competition here uses at this level of detail. Where a federation
        separates level clubs on head-to-head instead, the order can differ from
        the official table by a place; the site says the table is the model's.
        """
        if not any(int(t.get("played") or 0) for t in self.teams):
            return {}
        rows = sorted(
            self.teams,
            key=lambda t: (-float(t.get("cur_pts") or 0),
                           -(float(t.get("gf") or 0) - float(t.get("ga") or 0)),
                           -float(t.get("gf") or 0),
                           self.names.get(t["id"], t["id"])))
        return {t["id"]: i for i, t in enumerate(rows, 1)}

    def events(self, team: dict) -> dict:
        out = {}
        for key, alt in EVENT_KEYS:
            v = team.get(key)
            if v is None:
                v = team.get(alt)
            out["out" if key == "out" else key] = _round(v)
        # `out` is spelled `releg` on a league row and `p_out` on a cup row.
        if out.get("out") is None:
            out["out"] = _round(team.get("releg"))
        return {"title": out.get("title"), "top": out.get("ucl"),
                "out": out.get("out")}


def _competitions(out_dir: str, ready) -> list[Competition]:
    """Every ready competition that has both files on disk, in manifest order."""
    want = set(ready or ())
    manifest = _read(os.path.join(out_dir, "leagues.json")) or {}
    entries = [lg for lg in manifest.get("leagues", [])
               if isinstance(lg, dict) and lg.get("slug") in want]
    # The caller's set is the truth about what is ready; the manifest is only
    # where the names come from. A competition the manifest has not heard of
    # yet -- a first run, or a manifest written before this one built -- still
    # gets its rows, under its slug.
    named = {lg["slug"] for lg in entries}
    entries += [{"slug": s, "name": s, "kind": "league"}
                for s in sorted(want - named)]
    out = []
    for entry in entries:
        slug = entry["slug"]
        fc = _read(os.path.join(out_dir, slug, "forecast.json"))
        ms = _read(os.path.join(out_dir, slug, "matches.json"))
        rows = (ms or {}).get("matches") if isinstance(ms, dict) else None
        if not isinstance(fc, dict) or not isinstance(rows, list):
            print(f"  !  {slug}: no readable forecast.json/matches.json, skipped")
            continue
        out.append(Competition(entry, fc, [m for m in rows if isinstance(m, dict)]))
    return out


def _global_spi(out_dir: str) -> dict:
    g = _read(os.path.join(out_dir, "global.json")) or {}
    spi = {}
    for c in g.get("clubs", []) if isinstance(g.get("clubs"), list) else []:
        if isinstance(c, dict) and c.get("id") is not None:
            spi[c["id"]] = _round(c.get("spi"), 1)
    return spi


def _registry_names():
    """Display names for ids a forecast does not carry. Optional by design: the
    registry is a 500KB read and nothing here fails without it."""
    try:
        from .parse import TeamRegistry
        reg = TeamRegistry()
        return {tid: m.get("name", tid) for tid, m in reg.meta.items()}
    except Exception:                                  # noqa: BLE001
        return {}


# ------------------------------------------------------- 1. clubindex.json

def build_clubindex(out_dir: str, comps: list[Competition], spi: dict,
                    fallback_names: dict) -> dict:
    clubs: dict[str, list] = {}
    for c in comps:
        pos = c.table_pos()
        upcoming: dict[str, list] = {}
        for m in sorted((x for x in c.matches if not x.get("played")),
                        key=_sort_key):
            for side, opp, home in ((m.get("h"), m.get("a"), True),
                                    (m.get("a"), m.get("h"), False)):
                if side is None or len(upcoming.setdefault(side, [])) >= NEXT_FIXTURES:
                    continue
                upcoming[side].append({
                    "opp": opp,
                    "opp_name": c.names.get(opp, fallback_names.get(opp, opp)),
                    "home": home,
                    "date": m.get("date"), "time": m.get("time"),
                    "md": m.get("md"),
                    "ph": _round(m.get("ph")), "pd": _round(m.get("pd")),
                    "pa": _round(m.get("pa")),
                })
        for t in c.teams:
            ev = c.events(t)
            clubs.setdefault(t["id"], []).append({
                "slug": c.slug, "name": c.name, "kind": c.kind,
                "pos": pos.get(t["id"]),
                "pts": int(t.get("cur_pts") or 0),
                "played": int(t.get("played") or 0),
                "spi": spi.get(t["id"]),
                "title": ev["title"], "top": ev["top"], "out": ev["out"],
                "next": upcoming.get(t["id"], []),
            })
    doc = {"generated": _stamp(), "clubs": clubs}
    size = _write(os.path.join(out_dir, "clubindex.json"), doc)
    n_multi = sum(1 for v in clubs.values() if len(v) > 1)
    print(f"  → clubindex.json ({len(clubs)} rows, {n_multi} in more than one "
          f"competition, {size / 1024:.0f}KB)")
    return doc


# -------------------------------------------------------- 2. upcoming.json

def _biggest_swing(m: dict) -> dict | None:
    """The one event this match moves most, as `{club, event, delta}`.

    `swings` carries a signed difference between the home-win and away-win
    conditional probabilities; which way round that is depends on nothing the
    reader can see, so the published delta is its size.
    """
    best = None
    for s in m.get("swings") or []:
        if not isinstance(s, dict):
            continue
        try:
            d = abs(float(s.get("swing")))
        except (TypeError, ValueError):
            continue
        if best is None or d > best[0]:
            best = (d, s)
    if best is None:
        return None
    _, s = best
    return {"club": s.get("team"), "event": s.get("event"),
            "delta": _round(best[0])}


def build_upcoming(out_dir: str, comps: list[Competition], today: dt.date,
                   fallback_names: dict) -> dict:
    lo = today - dt.timedelta(days=WINDOW_BACK)
    hi = today + dt.timedelta(days=WINDOW_FWD)
    rows = []
    for c in comps:
        for m in c.matches:
            d = _date(m.get("date"))
            if d is None or not (lo <= d <= hi):
                continue
            h, a = m.get("h"), m.get("a")
            rows.append({
                "slug": c.slug, "league": c.name, "kind": c.kind,
                "md": m.get("md"), "date": m.get("date"), "time": m.get("time"),
                "h": h, "a": a,
                "hn": c.names.get(h, fallback_names.get(h, h)),
                "an": c.names.get(a, fallback_names.get(a, a)),
                "ph": _round(m.get("ph")), "pd": _round(m.get("pd")),
                "pa": _round(m.get("pa")),
                "lev": _round(m.get("lev")),
                "swing": _biggest_swing(m),
                "played": bool(m.get("played")),
                "hg": m.get("hg"), "ag": m.get("ag"),
            })
    rows.sort(key=lambda r: (_sort_key(r), r["slug"], r["h"] or ""))
    doc = {"generated": _stamp(), "from": lo.isoformat(), "to": hi.isoformat(),
           "matches": rows}
    size = _write(os.path.join(out_dir, "upcoming.json"), doc)
    print(f"  → upcoming.json ({len(rows)} rows, {lo} to {hi}, "
          f"{size / 1024:.0f}KB)")
    return doc


# ------------------------------------------------ 3. <slug>/recaps.json

def _complete_rounds(matches: list[dict]) -> list:
    """Every matchweek whose last fixture has been played, in order.

    A round that is half played is not a round in review, whatever the recap
    happens to say today, so nothing is archived for it.
    """
    by_md: dict = {}
    for m in matches:
        by_md.setdefault(m.get("md"), []).append(m)
    done = [md for md, rows in by_md.items()
            if rows and all(r.get("played") for r in rows)]

    def key(md):
        return (0, int(md)) if str(md).isdigit() else (1, str(md))
    return sorted(done, key=key)


def _round_rows(md, played: list[dict], report: dict, frozen: dict,
                names: dict) -> tuple[list[dict], list[dict]]:
    """The round's results, twice: as match rows and as the week's shocks.

    The spine is the fixture list, not `season_report.json`: a match played
    before this build first saw it has no frozen forecast on record and is
    absent from the report, and a matchweek in review missing one of its
    matches is not the matchweek. Where the report does have the fixture, its
    probability is the honest pre-kick-off one and is used in preference to the
    model's current view of a result it has already seen.
    """
    rows, graded = [], []
    for m in played:
        h, a, hg, ag = m.get("h"), m.get("a"), m.get("hg"), m.get("ag")
        if hg is None or ag is None:
            continue
        rep = report.get((h, a)) or {}
        f = frozen.get(f"{h}|{a}") or m
        p = [_round(f.get("ph")), _round(f.get("pd")), _round(f.get("pa"))]
        if None in p:
            continue
        y = 0 if hg > ag else (1 if hg == ag else 2)
        called = rep.get("called")
        if called is None:
            called = max(range(3), key=lambda i: p[i]) == y
        label = rep.get("label") or (f"{names.get(h, h)} {hg}-{ag} "
                                     f"{names.get(a, a)}")
        base = {"md": md, "date": m.get("date"), "h": h, "a": a,
                "hg": hg, "ag": ag, "called": bool(called), "label": label}
        # The shock row is the shape `recap.json` and `season_report.json`
        # already use: one probability, the one the result actually had.
        graded.append({**base, "p": _round(rep.get("p", p[y]))})
        rows.append({**base, "p": p, "surprise": _round(1 - p[y])})
    graded.sort(key=lambda r: (float(r["p"]), r["date"] or "", r["h"] or ""))
    return rows, graded[:5]


def archive_recaps(out_dir: str, c: Competition, today: dt.date) -> dict | None:
    """Append any newly completed matchweek to `<slug>/recaps.json`.

    Append-only, keyed by matchweek: a round already in the file is never
    rewritten, so re-running the build four times a day is a no-op after the
    first. The latest completed round carries the narrative and movers the
    current `recap.json` was written about; a round completed before this file
    existed is backfilled with its results alone rather than with today's prose
    about a different week.
    """
    path = os.path.join(out_dir, c.slug, "recaps.json")
    doc = _read(path)
    if not isinstance(doc, dict) or not isinstance(doc.get("rounds"), list):
        doc = {"generated": _stamp(), "rounds": []}
    have = {r.get("md") for r in doc["rounds"] if isinstance(r, dict)}

    complete = _complete_rounds(c.matches)
    todo = [md for md in complete if md not in have]
    if not todo:
        # A competition that has kicked off but finished no round yet still
        # gets an empty archive: the review page fetches this file for every
        # competition with a result, and a missing one is a console 404.
        kicked_off = any(m.get("played") for m in c.matches)
        if kicked_off and not os.path.exists(path):
            _write(path, doc)
            print(f"  → {c.slug}/recaps.json (0 rounds, nothing complete yet)")
        return None

    recap = _read(os.path.join(out_dir, c.slug, "recap.json")) or {}
    report = _read(os.path.join(out_dir, c.slug, "season_report.json")) or {}
    preds = _read(os.path.join(out_dir, c.slug, "predictions.json")) or {}
    frozen = preds.get("frozen") if isinstance(preds.get("frozen"), dict) else {}
    report_rows = {(r.get("h"), r.get("a")): r
                   for r in report.get("matches", []) if isinstance(r, dict)}
    latest = complete[-1]

    added = 0
    for md in todo:
        played = [m for m in c.matches if m.get("md") == md]
        dates = sorted(str(m.get("date")) for m in played if m.get("date"))
        rows, shocks = _round_rows(md, played, report_rows, frozen, c.names)
        entry = {
            "md": md,
            "from": dates[0] if dates else None,
            "to": dates[-1] if dates else None,
            "written": today.isoformat(),
            "narrative": recap.get("narrative", []) if md == latest else [],
            "movers": recap.get("movers", []) if md == latest else [],
            "shocks": shocks,
            "matches": rows,
        }
        doc["rounds"].append(entry)
        added += 1

    def key(r):
        md = r.get("md")
        return (0, int(md)) if str(md).isdigit() else (1, str(md))
    doc["rounds"].sort(key=key)
    doc["generated"] = _stamp()
    size = _write(path, doc)
    print(f"  → {c.slug}/recaps.json ({len(doc['rounds'])} rounds, "
          f"{added} new, {size / 1024:.0f}KB)")
    return doc


# -------------------------------------------- 4. global_history.json

def _monday(d: dt.date) -> str:
    return (d - dt.timedelta(days=d.weekday())).isoformat()


def build_global_history(out_dir: str, comps: list[Competition],
                         today: dt.date) -> dict | None:
    g = _read(os.path.join(out_dir, "global.json"))
    clubs = (g or {}).get("clubs")
    if not isinstance(clubs, list) or not clubs:
        print("  !  global.json unreadable, global_history.json left alone")
        return None
    path = os.path.join(out_dir, "global_history.json")
    doc = _read(path)
    if not isinstance(doc, dict) or not isinstance(doc.get("weeks"), list) \
            or not isinstance(doc.get("clubs"), dict):
        doc = {"weeks": [], "clubs": {}}
    weeks = [str(w) for w in doc["weeks"]]
    lines = {k: list(v) for k, v in doc["clubs"].items() if isinstance(v, list)}

    week = _monday(today)
    if week in weeks:
        print(f"  → global_history.json ({len(lines)} clubs x {len(weeks)} "
              f"weeks, {week} already recorded)")
        return doc

    spi = {c["id"]: _round(c.get("spi"), 1) for c in clubs
           if isinstance(c, dict) and c.get("id")}
    ranked = sorted((c for c in clubs if isinstance(c, dict) and c.get("id")),
                    key=lambda c: (c.get("rank") if isinstance(c.get("rank"), int)
                                   else 10 ** 6, c["id"]))
    # Everything already in the file stays in it, every club in a competition
    # this site forecasts is in it, and the rest of the ranking fills the
    # budget from the top down.
    keep = set(lines) | {t["id"] for c in comps for t in c.teams}
    for c in ranked:
        if len(keep) >= MAX_HISTORY_CLUBS:
            break
        keep.add(c["id"])

    n = len(weeks)
    for cid in sorted(keep):
        line = lines.get(cid, [])
        line = (line + [None] * n)[:n]
        line.append(spi.get(cid))
        lines[cid] = line
    weeks.append(week)

    doc = {"generated": _stamp(), "weeks": weeks, "clubs": lines}
    # The budget is a season's worth of columns, so the guard trims the oldest
    # rather than refusing to record this week.
    while len(json.dumps(doc, separators=(",", ":"))) > HISTORY_BUDGET \
            and len(doc["weeks"]) > 2:
        doc["weeks"] = doc["weeks"][1:]
        doc["clubs"] = {k: v[1:] for k, v in doc["clubs"].items()}
        doc["clubs"] = {k: v for k, v in doc["clubs"].items() if any(
            x is not None for x in v)}
    size = _write(path, doc)
    print(f"  → global_history.json ({len(doc['clubs'])} clubs x "
          f"{len(doc['weeks'])} weeks, {size / 1024:.0f}KB)")
    return doc


# ----------------------------------------- 5. one calendar file per fixture

def build_match_calendars(out_dir: str, comps: list[Competition],
                          today: dt.date) -> int:
    """`site/cal/<slug>/match/<date>-<h>-<a>.ics`: one fixture, one event.

    The match dialog's "add to calendar" had only the club feed to offer, which
    subscribes a reader to a whole season to record one Saturday.
    """
    base = os.path.join(os.path.dirname(os.path.abspath(out_dir)), "cal")
    hi = today + dt.timedelta(days=MATCH_ICS_DAYS)
    total = 0
    for c in comps:
        meta = {t["id"]: t for t in c.teams}
        d_out = os.path.join(base, c.slug, "match")
        wanted = {}
        for m in c.matches:
            d = _date(m.get("date"))
            if m.get("played") or d is None or not (today <= d <= hi):
                continue
            wanted[f"{m['date']}-{m['h']}-{m['a']}.ics"] = m
        for fname, m in wanted.items():
            label = (f"{'Matchday' if c.kind == 'cup' else 'Matchweek'} "
                     f"{m.get('md')}")
            text = feeds.calendar(
                [m], meta,
                title=f"{meta.get(m['h'], {}).get('name', m['h'])} v "
                      f"{meta.get(m['a'], {}).get('name', m['a'])}",
                uid_ns=f"{c.slug}.537", round_label=lambda _md, s=label: s)
            feeds.write(os.path.join(d_out, fname), text)
        if os.path.isdir(d_out):
            for stale in os.listdir(d_out):
                if stale.endswith(".ics") and stale not in wanted:
                    try:
                        os.remove(os.path.join(d_out, stale))
                    except OSError:
                        pass
        total += len(wanted)
    print(f"  → cal/<slug>/match/*.ics ({total} fixtures inside "
          f"{MATCH_ICS_DAYS} days)")
    return total


# ------------------------------------------------------------------ entry

def build_all(out_dir: str, ready, today: dt.date | None = None) -> None:
    """Every cross-competition extra, from the files the run has just written.

    Called at the end of `run.main()`, after the manifest and the global
    ranking exist. One competition's missing or malformed file costs that
    competition its rows and nothing else; nothing raised in here reaches the
    caller.
    """
    today = today or dt.date.today()
    try:
        comps = _competitions(out_dir, ready)
    except Exception as exc:                            # noqa: BLE001
        print(f"  !  extras: no competitions readable ({exc})")
        return
    if not comps:
        print("  !  extras: nothing ready, no cross-competition files written")
        return

    fallback = _registry_names()
    spi = _global_spi(out_dir)
    for name, fn in (
            ("clubindex.json",
             lambda: build_clubindex(out_dir, comps, spi, fallback)),
            ("upcoming.json",
             lambda: build_upcoming(out_dir, comps, today, fallback)),
            ("global_history.json",
             lambda: build_global_history(out_dir, comps, today)),
            ("match calendars",
             lambda: build_match_calendars(out_dir, comps, today))):
        try:
            fn()
        except Exception as exc:                        # noqa: BLE001
            print(f"  !  {name} failed: {exc}")

    for c in comps:
        try:
            archive_recaps(out_dir, c, today)
        except Exception as exc:                        # noqa: BLE001
            print(f"  !  {c.slug}/recaps.json failed: {exc}")
