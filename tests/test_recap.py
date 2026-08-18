"""Tests for the weekly recap.

The recap reads two optional files that may be missing, empty or half-written
when the pipeline runs, and it runs inside the build that produces the forecast.
So the properties worth pinning down are: it never raises, it only claims to be
comparing forecasts when there is genuinely something to compare against, and
the deltas it publishes are the arithmetic it says they are.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import recap                                            # noqa: E402


def _rows(**by_team) -> list:
    """Per-team forecast rows in the shape run.py hands to write_recap."""
    return [{"id": t, "name": t.title(), "short": t[:3].upper(),
             "primary": "#3987e5", "spi": 70.0, "pts": 60.0, **vals}
            for t, vals in by_team.items()]


def _snapshot(date: str, played: int = 0, **by_team) -> dict:
    return {"date": date, "played": played,
            "teams": {t: {"title": v.get("title", 0.0), "ucl": v.get("ucl", 0.0),
                          "releg": v.get("releg", 0.0), "pts": 60.0, "spi": 70.0}
                      for t, v in by_team.items()}}


def _read(path) -> dict:
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------- no history
def test_no_history_gives_a_valid_preseason_recap(tmp_path):
    """The first build of a season has nothing to compare against."""
    out = tmp_path / "recap.json"
    got = recap.write_recap(str(out), _rows(arsenal={"title": 0.4, "ucl": 0.9,
                                                     "releg": 0.0}), 0)
    on_disk = _read(out)
    assert on_disk == got
    assert on_disk["mode"] == "preseason"
    assert on_disk["movers"] == [] and on_disk["shocks"] == []
    assert on_disk["played"] == 0
    assert on_disk["note"] == recap.PRESEASON_NOTE
    assert dt.date.fromisoformat(on_disk["asof"]) == dt.date.today()


# ---------------------------------------------------------------- movement
def test_two_snapshots_produce_sorted_one_per_club_movers(tmp_path):
    today = dt.date.today()
    old = (today - dt.timedelta(days=7)).isoformat()
    yday = (today - dt.timedelta(days=1)).isoformat()
    (tmp_path / "history.json").write_text(json.dumps({"snapshots": [
        _snapshot(old,
                  arsenal={"title": 0.40, "ucl": 0.90, "releg": 0.00},
                  chelsea={"title": 0.10, "ucl": 0.50, "releg": 0.01},
                  hull={"title": 0.00, "ucl": 0.02, "releg": 0.55},
                  fulham={"title": 0.02, "ucl": 0.20, "releg": 0.10}),
        # a later snapshot exists, but the oldest inside the window is the one
        # the recap must measure against
        _snapshot(yday,
                  arsenal={"title": 0.42, "ucl": 0.91, "releg": 0.00},
                  chelsea={"title": 0.11, "ucl": 0.52, "releg": 0.01},
                  hull={"title": 0.00, "ucl": 0.02, "releg": 0.54},
                  fulham={"title": 0.02, "ucl": 0.20, "releg": 0.10}),
    ]}))
    out = tmp_path / "recap.json"
    got = recap.write_recap(str(out), _rows(
        arsenal={"title": 0.55, "ucl": 0.95, "releg": 0.00},   # title +0.15
        chelsea={"title": 0.09, "ucl": 0.42, "releg": 0.02},   # ucl   -0.08
        hull={"title": 0.00, "ucl": 0.02, "releg": 0.75},      # releg +0.20
        fulham={"title": 0.02, "ucl": 0.201, "releg": 0.099},  # all below 1pt
    ), 30)

    assert got == _read(out)
    assert got["mode"] == "inseason"
    assert got["played"] == 30
    assert got["note"] == f"Compared with the forecast of {old}."

    movers = got["movers"]
    assert [m["id"] for m in movers] == ["hull", "arsenal", "chelsea"], \
        "sorted by absolute move, and fulham's sub-point drift is not a story"
    assert [m["metric"] for m in movers] == ["releg", "title", "ucl"]
    assert len({m["id"] for m in movers}) == len(movers), "one metric per club"
    for m in movers:
        assert abs(m["delta"]) >= recap.MIN_DELTA
        assert abs(m["after"] - m["before"] - m["delta"]) < 1e-9
    assert movers[0]["before"] == 0.55 and movers[0]["after"] == 0.75
    assert abs(movers[1]["delta"] - 0.15) < 1e-9
    assert abs(movers[2]["delta"] + 0.08) < 1e-9


def test_only_the_largest_metric_is_reported_for_a_club(tmp_path):
    (tmp_path / "history.json").write_text(json.dumps({"snapshots": [
        _snapshot((dt.date.today() - dt.timedelta(days=3)).isoformat(),
                  chelsea={"title": 0.30, "ucl": 0.60, "releg": 0.02}),
    ]}))
    got = recap.write_recap(str(tmp_path / "recap.json"), _rows(
        chelsea={"title": 0.20, "ucl": 0.35, "releg": 0.05}), 12)
    assert len(got["movers"]) == 1
    assert got["movers"][0]["metric"] == "ucl"
    assert abs(got["movers"][0]["delta"] + 0.25) < 1e-9


def test_todays_snapshot_is_never_the_baseline(tmp_path):
    """Comparing the forecast with itself would report no movement at all."""
    (tmp_path / "history.json").write_text(json.dumps({"snapshots": [
        _snapshot(dt.date.today().isoformat(), arsenal={"title": 0.30}),
    ]}))
    got = recap.write_recap(str(tmp_path / "recap.json"),
                            _rows(arsenal={"title": 0.60, "ucl": 0.9, "releg": 0.0}), 20)
    assert got["mode"] == "preseason"
    assert got["movers"] == []


def test_nothing_played_stays_preseason_even_with_history(tmp_path):
    (tmp_path / "history.json").write_text(json.dumps({"snapshots": [
        _snapshot((dt.date.today() - dt.timedelta(days=2)).isoformat(),
                  arsenal={"title": 0.30}),
    ]}))
    got = recap.write_recap(str(tmp_path / "recap.json"),
                            _rows(arsenal={"title": 0.45, "ucl": 0.9, "releg": 0.0}), 0)
    assert got["mode"] == "preseason"
    assert got["note"] == recap.PRESEASON_NOTE


# ---------------------------------------------------------------- shocks
def test_shocks_are_copied_from_the_season_report(tmp_path):
    surprises = [{"md": 2, "date": "2026-08-29", "h": "hull", "a": "arsenal",
                  "hg": 2, "ag": 1, "p": 0.07, "called": False,
                  "label": "Hull 2-1 Arsenal"}]
    (tmp_path / "season_report.json").write_text(json.dumps(
        {"n": 20, "log_loss": 1.01, "accuracy": 0.5,
         "surprises": surprises, "confident": [], "matches": []}))
    got = recap.write_recap(str(tmp_path / "recap.json"),
                            _rows(arsenal={"title": 0.4, "ucl": 0.9, "releg": 0.0}), 20)
    assert got["shocks"] == surprises


def test_an_unscored_season_report_contributes_no_shocks(tmp_path):
    (tmp_path / "season_report.json").write_text(json.dumps({"n": 0, "matches": []}))
    got = recap.write_recap(str(tmp_path / "recap.json"),
                            _rows(arsenal={"title": 0.4, "ucl": 0.9, "releg": 0.0}), 0)
    assert got["shocks"] == []


# ---------------------------------------------------------------- corruption
def test_corrupt_history_degrades_to_preseason(tmp_path):
    """A truncated write must not take the whole build down."""
    (tmp_path / "history.json").write_text('{"snapshots": [{"date": "2026-0')
    out = tmp_path / "recap.json"
    got = recap.write_recap(str(out), _rows(arsenal={"title": 0.4, "ucl": 0.9,
                                                     "releg": 0.0}), 30)
    assert got["mode"] == "preseason"
    assert got["movers"] == []
    assert got["note"] == recap.PRESEASON_NOTE
    assert _read(out) == got


def test_corrupt_season_report_degrades_to_no_shocks(tmp_path):
    (tmp_path / "season_report.json").write_text("not json at all")
    got = recap.write_recap(str(tmp_path / "recap.json"),
                            _rows(arsenal={"title": 0.4, "ucl": 0.9, "releg": 0.0}), 5)
    assert got["shocks"] == []


def test_junk_shaped_history_is_survived(tmp_path):
    """Right file, wrong shape: entries missing dates, teams or the list itself."""
    (tmp_path / "history.json").write_text(json.dumps(
        {"snapshots": [{"date": None}, "nonsense", {"teams": {}},
                       {"date": "not-a-date", "teams": {"arsenal": {"title": 0.1}}}]}))
    got = recap.write_recap(str(tmp_path / "recap.json"),
                            _rows(arsenal={"title": 0.4, "ucl": 0.9, "releg": 0.0}), 30)
    assert got["mode"] == "preseason"
    assert got["movers"] == []


def test_movers_are_capped(tmp_path):
    old = (dt.date.today() - dt.timedelta(days=4)).isoformat()
    ids = [f"club{i}" for i in range(12)]
    (tmp_path / "history.json").write_text(json.dumps({"snapshots": [
        _snapshot(old, **{t: {"title": 0.0} for t in ids})]}))
    got = recap.write_recap(str(tmp_path / "recap.json"), _rows(
        **{t: {"title": 0.05 * (i + 1), "ucl": 0.0, "releg": 0.0}
           for i, t in enumerate(ids)}), 30)
    assert len(got["movers"]) == recap.MAX_MOVERS
    deltas = [abs(m["delta"]) for m in got["movers"]]
    assert deltas == sorted(deltas, reverse=True)
