"""Build the frozen market and ClubElo baselines the backtest is scored against.

    python -m tools.extract_baselines

Run by hand, not by CI, and the output is committed. Two reasons for that.

*The source is a snapshot, not a feed.* `xgabora/Club-Football-Match-Data-2000-2025`
carries closing bookmaker odds and twice-monthly ClubElo ratings for forty-two
leagues, but its last commit is 2025-08-09 and its data stops on 2025-06-01. A
baseline the model is judged against should be frozen anyway — a benchmark that
moves under you is not a benchmark — so the staleness is the right property
here even though it would disqualify the source for anything live.

*The source is 53 MB.* Downloading that every six hours to read eighteen
thousand rows would be rude to a stranger's repository and slow for us.

What comes out: `data/baselines/<league>.json`, one row per match in the
backtest window, carrying the de-vigged closing 1X2 probabilities and each
club's ClubElo rating as of the most recent snapshot before kick-off.
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import urllib.request
from bisect import bisect_right

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import leagues                                    # noqa: E402
from model.parse import TeamRegistry, normalise               # noqa: E402
from model.priors import devig                                # noqa: E402

RAW = ("https://raw.githubusercontent.com/xgabora/"
       "Club-Football-Match-Data-2000-2025/main/data")
MATCHES = f"{RAW}/Matches.csv"
ELO = f"{RAW}/EloRatings.csv"

#: football-data.co.uk division codes, as the source spells them.
DIVISION = {"premier-league": "E0", "la-liga": "SP1", "serie-a": "I1",
            "bundesliga": "D1", "ligue-1": "F1"}

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "data", "baselines")

#: The window the published backtest covers. Every league starts at 2015-16 and
#: the source's last match is 2025-06-01, so 2024-25 is the final full season.
FROM_SEASON = "2015-16"
#: Five extra seasons before the window are extracted purely so the ClubElo
#: rating -> 1X2 mapping can be fitted on matches the backtest never scores.
#: Without them that mapping would be fitted on the same results it is judged
#: against, which is the exact sin the rest of this project is built to avoid.
CALIBRATE_FROM = "2010-07-01"
FROM_DATE = CALIBRATE_FROM


def fetch(url: str) -> str:
    print(f"  fetching {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "football-forecast/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read().decode("utf-8", "replace")


def load_elo(reg: TeamRegistry) -> dict[str, tuple[list[str], list[float]]]:
    """Club id -> (sorted snapshot dates, ratings), for an as-of lookup."""
    text = fetch(ELO)
    by: dict[str, list[tuple[str, float]]] = {}
    for row in csv.DictReader(io.StringIO(text)):
        d = (row.get("date") or "").strip()
        club = (row.get("club") or "").strip()
        if not d or not club or d < FROM_DATE:
            continue
        try:
            elo = float(row["elo"])
        except (KeyError, TypeError, ValueError):
            continue
        by.setdefault(reg.resolve(club), []).append((d, elo))
    out = {}
    for tid, rows in by.items():
        rows.sort()
        out[tid] = ([r[0] for r in rows], [r[1] for r in rows])
    print(f"  · {len(out)} clubs with ClubElo history")
    return out


def elo_at(store, tid: str, date: str) -> float | None:
    rec = store.get(tid)
    if not rec:
        return None
    dates, vals = rec
    i = bisect_right(dates, date)
    return vals[i - 1] if i else None


def main() -> None:
    reg = TeamRegistry()
    elo = load_elo(reg)
    text = fetch(MATCHES)

    want = {v: k for k, v in DIVISION.items()}
    rows: dict[str, list] = {k: [] for k in DIVISION}
    missing_odds = {k: 0 for k in DIVISION}
    unmapped: set[str] = set()
    known = set(reg.meta)

    for r in csv.DictReader(io.StringIO(text)):
        slug = want.get((r.get("Division") or "").strip())
        if slug is None:
            continue
        d = (r.get("MatchDate") or "").strip()
        if not d or d < FROM_DATE:
            continue
        h = reg.resolve((r.get("HomeTeam") or "").strip())
        a = reg.resolve((r.get("AwayTeam") or "").strip())
        for t, raw in ((h, r.get("HomeTeam")), (a, r.get("AwayTeam"))):
            if t not in known:
                unmapped.add(f"{raw} -> {t}")
        try:
            odds = {"H": float(r["OddHome"]), "D": float(r["OddDraw"]),
                    "A": float(r["OddAway"])}
        except (KeyError, TypeError, ValueError):
            odds = None
        if not odds or min(odds.values()) <= 1.0:
            missing_odds[slug] += 1
            p = None
        else:
            q = devig(odds)
            p = [round(q["H"], 4), round(q["D"], 4), round(q["A"], 4)]
        eh, ea = elo_at(elo, h, d), elo_at(elo, a, d)
        rows[slug].append([d, h, a, p,
                           round(eh, 1) if eh else None,
                           round(ea, 1) if ea else None])

    os.makedirs(OUT, exist_ok=True)
    for slug, rs in rows.items():
        rs.sort(key=lambda x: (x[0], x[1]))
        with_odds = sum(1 for x in rs if x[3])
        with_elo = sum(1 for x in rs if x[4] and x[5])
        payload = {
            "source": ("xgabora/Club-Football-Match-Data-2000-2025 "
                       "(football-data.co.uk closing odds + ClubElo snapshots)"),
            "url": "https://github.com/xgabora/Club-Football-Match-Data-2000-2025",
            "extracted_from": FROM_SEASON,
            "calibration_from": CALIBRATE_FROM,
            "note": ("Frozen snapshot: the upstream repository's last commit is "
                     "2025-08-09 and its last match is 2025-06-01, so seasons "
                     "after 2024-25 have no market or ClubElo row and are "
                     "scored on fewer matches. Odds are the closing 1X2 prices, "
                     "de-vigged proportionally."),
            "cols": ["date", "home", "away", "market", "elo_home", "elo_away"],
            "n": len(rs), "with_odds": with_odds, "with_elo": with_elo,
            "rows": rs,
        }
        path = os.path.join(OUT, f"{slug}.json")
        with open(path, "w") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        print(f"  → {path}: {len(rs)} matches, {with_odds} with odds, "
              f"{with_elo} with ClubElo, {missing_odds[slug]} priceless")
    if unmapped:
        print(f"\n  ! {len(unmapped)} club names resolved to ids that are not in "
              "team_meta.json (harmless if they are relegated sides, but check):")
        for u in sorted(unmapped)[:40]:
            print(f"      {u}")


if __name__ == "__main__":
    main()
