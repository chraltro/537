"""The weekly review: what the forecast changed its mind about, and what it got wrong.

A forecast that only publishes today's number is unfalsifiable. `history.json`
already keeps one snapshot per day and `season_report.json` already scores the
model against the probabilities it published before kick-off; this module joins
the two into the single small file the Races page reads, so the page never has
to reason about which snapshot counts as "last week".

Nothing here re-simulates anything. Every input is a file the pipeline has
already written, and every failure to read one degrades to preseason mode rather
than taking the build down with it.
"""
from __future__ import annotations

import datetime as dt
import json
import os

# Metrics compared between snapshots, in the order they are quoted on the site.
METRICS = ("title", "ucl", "releg")

# A move smaller than a percentage point is noise in a 50,000-season simulation.
MIN_DELTA = 0.01
MAX_MOVERS = 8

# The comparison window: far enough back to be a week's worth of movement,
# never today's own snapshot (which would compare the forecast with itself).
WINDOW_DAYS = 10
MIN_AGE_DAYS = 1

PRESEASON_NOTE = ("Baseline preseason forecast — movement tracking begins "
                  "once results arrive.")


def _read_json(path: str):
    """Return the parsed file, or None if it is missing, unreadable or corrupt."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _parse_date(s):
    try:
        return dt.date.fromisoformat(str(s))
    except (TypeError, ValueError):
        return None


def baseline_snapshot(snapshots: list, today: dt.date) -> dict | None:
    """The snapshot today's forecast is measured against.

    Preferred: the oldest snapshot inside the last ten days that is at least a
    day old — that is "roughly a week ago", and it survives the site being
    rebuilt several times a day. If nothing falls in that window (a gap in the
    run, or a season that has only just started keeping records), fall back to
    whatever the most recent earlier snapshot is, so a comparison is still made.
    """
    dated = []
    for s in snapshots or []:
        if not isinstance(s, dict):
            continue
        d = _parse_date(s.get("date"))
        if d is None or not isinstance(s.get("teams"), dict):
            continue
        if (today - d).days < MIN_AGE_DAYS:
            continue                     # today's own snapshot, or the future
        dated.append((d, s))
    if not dated:
        return None
    dated.sort(key=lambda x: x[0])
    window = [s for d, s in dated if (today - d).days <= WINDOW_DAYS]
    return window[0] if window else dated[-1][1]


def movers(rows: list, before: dict) -> list:
    """Which clubs the model changed its mind about, largest move first.

    One entry per club: a side that drifts on all three metrics at once is
    telling one story, not three, so only its biggest move is reported.
    """
    teams = (before or {}).get("teams", {}) or {}
    out = []
    for r in rows or []:
        was = teams.get(r.get("id"))
        if not isinstance(was, dict):
            continue
        best = None
        for m in METRICS:
            a, b = r.get(m), was.get(m)
            if a is None or b is None:
                continue
            try:
                delta = float(a) - float(b)
            except (TypeError, ValueError):
                continue
            if abs(delta) < MIN_DELTA:
                continue
            if best is None or abs(delta) > abs(best["delta"]):
                best = {"id": r["id"], "metric": m,
                        "before": round(float(b), 4), "after": round(float(a), 4),
                        "delta": round(delta, 4)}
        if best:
            out.append(best)
    out.sort(key=lambda x: -abs(x["delta"]))
    return out[:MAX_MOVERS]


def write_recap(path: str, rows: list, played: int) -> dict:
    """Build recap.json next to the rest of the site's data. Never raises."""
    here = os.path.dirname(os.path.abspath(path))
    today = dt.date.today()

    hist = _read_json(os.path.join(here, "history.json")) or {}
    snaps = hist.get("snapshots") if isinstance(hist, dict) else None
    before = baseline_snapshot(snaps if isinstance(snaps, list) else [], today)

    report = _read_json(os.path.join(here, "season_report.json")) or {}
    shocks = []
    if isinstance(report, dict) and report.get("n"):
        raw = report.get("surprises")
        if isinstance(raw, list):
            shocks = raw

    inseason = bool(before) and bool(played)
    recap = {
        "mode": "inseason" if inseason else "preseason",
        "asof": today.isoformat(),
        "played": int(played or 0),
        "movers": movers(rows, before) if before else [],
        "shocks": shocks,
        "note": (f"Compared with the forecast of {before['date']}."
                 if inseason else PRESEASON_NOTE),
    }
    with open(path, "w") as fh:
        json.dump(recap, fh, separators=(",", ":"))
    return recap
