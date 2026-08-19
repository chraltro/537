"""Derived views that make the forecast useful rather than merely correct.

Everything here is computed from the same simulation and rating objects the
rest of the pipeline already produces; nothing re-fits or re-simulates.
"""
from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: Default output directory. Both state files below live *inside* a league's own
#: directory, so every function here takes `out_dir`; this is only the fallback
#: for a caller that has none, and the one thing tests monkeypatch.
OUT = os.path.join(HERE, "site", "data")


# --------------------------------------------------------------------------
# Strength of schedule
# --------------------------------------------------------------------------
def strength_of_schedule(fixtures, teams, spi: dict[str, float],
                         home_adv: float) -> dict[str, dict]:
    """How hard is what is left, and how hard has what is gone been?

    Difficulty is the opponent's rating adjusted for venue: the same opponent
    away is a harder fixture than at home, and a schedule is only as unfair as
    the balance of the two.
    """
    venue = home_adv * 6.0            # rating points a home tie is worth, roughly
    out: dict[str, dict] = {}
    for t in teams:
        played, left = [], []
        for f in fixtures:
            if t not in (f.home, f.away):
                continue
            opp = f.away if f.home == t else f.home
            diff = spi[opp] + (-venue if f.home == t else venue)
            (played if f.played else left).append(
                {"opp": opp, "home": f.home == t, "md": f.matchday,
                 "date": f.date.isoformat(), "difficulty": round(diff, 1),
                 "played": f.played,
                 "gf": (f.hg if f.home == t else f.ag) if f.played else None,
                 "ga": (f.ag if f.home == t else f.hg) if f.played else None})
        left.sort(key=lambda r: (r["md"] or 0, r["date"]))
        out[t] = {
            "remaining": round(float(np.mean([r["difficulty"] for r in left])), 1) if left else None,
            "played": round(float(np.mean([r["difficulty"] for r in played])), 1) if played else None,
            "next": left[:6],
            "fixtures": played + left,
        }
    ranked = sorted((t for t in teams if out[t]["remaining"] is not None),
                    key=lambda t: -out[t]["remaining"])
    for rank, t in enumerate(ranked, 1):
        out[t]["rank"] = rank            # 1 = hardest run of fixtures left
    return out


# --------------------------------------------------------------------------
# Forecast history
# --------------------------------------------------------------------------
def append_history(rows: list[dict], played: int, out_dir: str | None = None) -> list[dict]:
    """Keep one snapshot per day so the forecast's own movement is visible.

    A forecast that only ever shows today's number is impossible to argue with.
    This is what makes 'the model liked them in August' checkable in April.

    The file is per league: two leagues sharing one history would overwrite each
    other's snapshot every run and silently destroy the record.
    """
    path = os.path.join(out_dir or OUT, "history.json")
    hist = []
    if os.path.exists(path):
        try:
            hist = json.load(open(path)).get("snapshots", [])
        except (ValueError, OSError):
            hist = []
    today = dt.date.today().isoformat()
    snap = {
        "date": today,
        "played": played,
        "teams": {r["id"]: {"title": round(r["title"], 4), "ucl": round(r["ucl"], 4),
                            "releg": round(r["releg"], 4), "pts": r["pts"],
                            "spi": r["spi"]} for r in rows},
    }
    hist = [h for h in hist if h["date"] != today]
    hist.append(snap)
    hist.sort(key=lambda h: h["date"])
    hist = hist[-400:]
    json.dump({"snapshots": hist}, open(path, "w"), separators=(",", ":"))
    return hist


# --------------------------------------------------------------------------
# Honest in-season scoring
# --------------------------------------------------------------------------
def freeze_predictions(matches: list[dict], out_dir: str | None = None) -> dict:
    """Archive each match's probabilities from the last build before kick-off.

    Scoring the model against results using probabilities computed *after* those
    results is meaningless -- the model would be marking its own homework with
    the answers in front of it. This stores the genuine pre-match forecast the
    first time a fixture is seen, and never overwrites it once played.
    """
    path = os.path.join(out_dir or OUT, "predictions.json")
    store = {}
    if os.path.exists(path):
        try:
            store = json.load(open(path)).get("frozen", {})
        except (ValueError, OSError):
            store = {}
    for m in matches:
        key = f"{m['h']}|{m['a']}"
        if m["played"]:
            continue                    # whatever was frozen before kick-off stands
        store[key] = {"ph": m["ph"], "pd": m["pd"], "pa": m["pa"],
                      "xgh": m["xgh"], "xga": m["xga"], "md": m["md"],
                      "asof": dt.date.today().isoformat()}
    json.dump({"frozen": store}, open(path, "w"), separators=(",", ":"))
    return store


def season_report(matches: list[dict], frozen: dict, names: dict) -> dict:
    """Score this season's completed matches against their pre-kick-off forecast."""
    rows, probs, ys = [], [], []
    for m in matches:
        if not m["played"]:
            continue
        f = frozen.get(f"{m['h']}|{m['a']}")
        if not f:
            continue                    # no honest pre-match forecast on record
        p = np.array([f["ph"], f["pd"], f["pa"]], dtype=float)
        y = 0 if m["hg"] > m["ag"] else (1 if m["hg"] == m["ag"] else 2)
        probs.append(p)
        ys.append(y)
        rows.append({"md": m["md"], "date": m["date"], "h": m["h"], "a": m["a"],
                     "hg": m["hg"], "ag": m["ag"],
                     "p": round(float(p[y]), 4),
                     "called": int(np.argmax(p)) == y,
                     "label": f"{names.get(m['h'], m['h'])} {m['hg']}-{m['ag']} "
                              f"{names.get(m['a'], m['a'])}"})
    if not rows:
        return {"n": 0, "matches": []}
    P, Y = np.array(probs), np.array(ys)
    ll = float(-np.mean(np.log(np.clip(P[np.arange(len(Y)), Y], 1e-12, 1))))
    surprises = sorted(rows, key=lambda r: r["p"])[:5]
    best = sorted((r for r in rows if r["called"]), key=lambda r: -r["p"])[:5]
    return {
        "n": len(rows),
        "log_loss": round(ll, 4),
        "accuracy": round(float(np.mean([r["called"] for r in rows])), 4),
        "surprises": surprises,
        "confident": best,
        "matches": rows,
    }
