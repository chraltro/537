"""The cross-competition post-pass, against a synthetic `site/data` tree.

`model/extras.py` reads only files the run has already written, so every test
here builds a tiny two-competition site in `tmp_path` and asserts on the JSON
that comes out. Nothing touches the network and nothing reads the real data
files: a test that passes only because the Premier League happens to be three
matchweeks in is not a test of the contract.

The failure modes worth catching are the quiet ones. An archive that rewrites a
completed matchweek loses the record it exists to keep; a weekly ranking column
appended twice in one day doubles the file every six hours; a fixture window
computed off by a day silently drops the round the page is for; and one
competition's malformed file must never cost the other nine theirs.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import extras                                        # noqa: E402

TODAY = dt.date(2026, 9, 4)


# --------------------------------------------------------------- a toy site
def _p(ph, pd, pa):
    return {"ph": ph, "pd": pd, "pa": pa}


def _match(md, date, h, a, *, time="15:00", played=False, hg=None, ag=None,
           p=(0.5, 0.25, 0.25), lev=0.0, swings=()):
    m = {"md": md, "date": date, "time": time, "h": h, "a": a,
         "ph": p[0], "pd": p[1], "pa": p[2], "sc": [1, 1], "scp": 0.12,
         "xgh": 1.4, "xga": 1.1, "lev": lev, "swings": list(swings),
         "played": played, "hg": hg, "ag": ag}
    return m


def _team(tid, name, **kw):
    row = {"id": tid, "name": name, "short": name[:3].upper(),
           "primary": "#123456", "pts": 60.0, "cur_pts": 0, "played": 0,
           "gf": 0, "ga": 0, "title": 0.25, "ucl": 0.6, "releg": 0.05}
    row.update(kw)
    return row


def _write(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(doc, fh)


def toy_site(root, *, cup=True) -> str:
    """A `site/data` tree with one league and one cup, sharing a club."""
    out = os.path.join(str(root), "site", "data")
    leagues = [{"slug": "toy-league", "name": "Toy League", "country": "Nowhere",
                "ready": True, "kind": "league", "n_teams": 4}]
    if cup:
        leagues.append({"slug": "toy-cup", "name": "Toy Cup", "country": "Europe",
                        "ready": True, "kind": "cup", "n_teams": 4})
    leagues.append({"slug": "not-ready", "name": "Not Ready", "ready": False,
                    "kind": "league"})
    _write(os.path.join(out, "leagues.json"),
           {"default": "toy-league", "leagues": leagues,
            "rated": [{"name": "Elsewhere", "slug": "elsewhere"}]})

    # --- the league: matchweek 1 complete, matchweek 2 half played, 3 to come
    league_matches = [
        _match(1, "2026-08-29", "alpha", "beta", played=True, hg=2, ag=1),
        _match(1, "2026-08-30", "gamma", "delta", played=True, hg=0, ag=0),
        _match(2, "2026-09-05", "beta", "gamma", played=True, hg=1, ag=3),
        _match(2, "2026-09-06", "delta", "alpha", time=None),
        _match(3, "2026-09-12", "alpha", "gamma", lev=0.3, swings=[
            {"team": "beta", "event": "title", "home": 0.1, "away": 0.24,
             "swing": -0.14},
            {"team": "delta", "event": "releg", "home": 0.4, "away": 0.2,
             "swing": 0.2}]),
        _match(3, "2026-09-13", "beta", "delta"),
        _match(4, "2026-10-01", "gamma", "alpha"),        # outside every window
    ]
    _write(os.path.join(out, "toy-league", "matches.json"),
           {"matches": league_matches})
    _write(os.path.join(out, "toy-league", "forecast.json"), {
        "generated": "2026-09-04T00:00:00+00:00", "season": "2026/27",
        "league": {"slug": "toy-league", "name": "Toy League"},
        "matches_played": 3, "matches_total": 7,
        "teams": [
            _team("alpha", "Alpha FC", cur_pts=3, played=1, gf=2, ga=1,
                  title=0.5, ucl=0.9, releg=0.0),
            _team("beta", "Beta FC", cur_pts=0, played=2, gf=2, ga=5,
                  title=0.1, ucl=0.3, releg=0.4),
            _team("gamma", "Gamma FC", cur_pts=4, played=2, gf=3, ga=1,
                  title=0.3, ucl=0.7, releg=0.02),
            _team("delta", "Delta FC", cur_pts=1, played=1, gf=0, ga=0,
                  title=0.1, ucl=0.4, releg=0.2)]})
    _write(os.path.join(out, "toy-league", "predictions.json"), {"frozen": {
        "alpha|beta": {"ph": 0.6, "pd": 0.25, "pa": 0.15, "md": 1},
        "gamma|delta": {"ph": 0.4, "pd": 0.3, "pa": 0.3, "md": 1},
        "beta|gamma": {"ph": 0.3, "pd": 0.3, "pa": 0.4, "md": 2}}})
    _write(os.path.join(out, "toy-league", "recap.json"), {
        "mode": "inseason", "asof": "2026-09-04", "played": 3,
        "movers": [{"id": "beta", "metric": "releg", "before": 0.2,
                    "after": 0.4, "delta": 0.2}],
        "shocks": [], "narrative": ["Alpha lead the race for the title."],
        "note": "Compared with the forecast of 2026-08-29."})
    _write(os.path.join(out, "toy-league", "season_report.json"), {
        "n": 3, "log_loss": 1.0, "accuracy": 0.66,
        "matches": [
            {"md": 1, "date": "2026-08-29", "h": "alpha", "a": "beta",
             "hg": 2, "ag": 1, "p": 0.6, "called": True,
             "label": "Alpha FC 2-1 Beta FC"},
            {"md": 1, "date": "2026-08-30", "h": "gamma", "a": "delta",
             "hg": 0, "ag": 0, "p": 0.3, "called": False,
             "label": "Gamma FC 0-0 Delta FC"},
            {"md": 2, "date": "2026-09-05", "h": "beta", "a": "gamma",
             "hg": 1, "ag": 3, "p": 0.4, "called": True,
             "label": "Beta FC 1-3 Gamma FC"}]})

    if cup:
        _write(os.path.join(out, "toy-cup", "matches.json"), {"matches": [
            _match(1, "2026-09-08", "alpha", "epsilon", time="21:00", lev=0.26,
                   swings=[{"team": "epsilon", "event": "out", "home": 0.9,
                            "away": 0.65, "swing": 0.25}]),
            _match(2, "2026-09-30", "epsilon", "alpha")]})
        _write(os.path.join(out, "toy-cup", "forecast.json"), {
            "generated": "2026-09-04T00:00:00+00:00", "season": "2026/27",
            "league": {"slug": "toy-cup", "name": "Toy Cup"},
            "teams": [
                {"id": "alpha", "name": "Alpha FC", "pts": 12.0, "cur_pts": 0,
                 "played": 0, "gf": 0, "ga": 0, "p_win": 0.3, "p_top8": 0.8,
                 "p_out": 0.004, "title": 0.3, "ucl": 0.8, "releg": 0.004},
                {"id": "epsilon", "name": "Epsilon FC", "pts": 6.0,
                 "cur_pts": 0, "played": 0, "gf": 0, "ga": 0, "p_win": 0.02,
                 "p_top8": 0.2, "p_out": 0.3, "title": 0.02, "ucl": 0.2,
                 "releg": 0.3}]})

    _write(os.path.join(out, "global.json"), {"clubs": [
        {"id": "alpha", "spi": 80.1, "rank": 1},
        {"id": "gamma", "spi": 70.0, "rank": 2},
        {"id": "beta", "spi": 60.5, "rank": 3},
        {"id": "delta", "spi": 50.2, "rank": 4},
        {"id": "epsilon", "spi": 40.0, "rank": 5},
        {"id": "outsider", "spi": 30.0, "rank": 6}]})
    return out


def build(root, today=TODAY, ready=("toy-league", "toy-cup"), **kw):
    out = kw.pop("out", None) or toy_site(root, **kw)
    extras.build_all(out, set(ready), today=today)
    return out


def read(out, *parts):
    with open(os.path.join(out, *parts)) as fh:
        return json.load(fh)


# ------------------------------------------------------------- clubindex.json
def test_clubindex_lists_every_ready_competition_a_club_is_in(tmp_path):
    out = build(tmp_path)
    doc = read(out, "clubindex.json")
    assert set(doc) == {"generated", "clubs"}
    dt.datetime.fromisoformat(doc["generated"])
    assert sorted(doc["clubs"]) == ["alpha", "beta", "delta", "epsilon", "gamma"]
    slugs = [e["slug"] for e in doc["clubs"]["alpha"]]
    assert slugs == ["toy-league", "toy-cup"], "manifest order, both competitions"
    assert [e["slug"] for e in doc["clubs"]["beta"]] == ["toy-league"]


def test_clubindex_carries_the_three_headline_events_for_both_kinds(tmp_path):
    out = build(tmp_path)
    doc = read(out, "clubindex.json")
    league, cup = doc["clubs"]["alpha"]
    assert (league["title"], league["top"], league["out"]) == (0.5, 0.9, 0.0)
    # A cup row spells the same three `p_win` / `p_top8` / `p_out`.
    assert (cup["title"], cup["top"], cup["out"]) == (0.3, 0.8, 0.004)
    assert cup["kind"] == "cup" and league["kind"] == "league"
    assert league["spi"] == cup["spi"] == 80.1, "one rating, both competitions"


def test_clubindex_position_is_the_table_and_is_null_before_kick_off(tmp_path):
    out = build(tmp_path)
    doc = read(out, "clubindex.json")
    pos = {c: doc["clubs"][c][0]["pos"] for c in ("alpha", "beta", "gamma", "delta")}
    # gamma 4pts, alpha 3, delta 1, beta 0.
    assert pos == {"gamma": 1, "alpha": 2, "delta": 3, "beta": 4}
    # The cup has not kicked off, so it has no table to have a position in.
    assert all(e["pos"] is None for c in doc["clubs"].values() for e in c
               if e["slug"] == "toy-cup")


def test_clubindex_next_is_at_most_five_unplayed_fixtures_soonest_first(tmp_path):
    out = build(tmp_path)
    nxt = read(out, "clubindex.json")["clubs"]["alpha"][0]["next"]
    assert [(n["date"], n["home"], n["opp"]) for n in nxt] == [
        ("2026-09-06", False, "delta"),
        ("2026-09-12", True, "gamma"),
        ("2026-10-01", False, "gamma")]
    assert nxt[0]["opp_name"] == "Delta FC" and nxt[0]["md"] == 2
    assert nxt[0]["time"] is None, "an unannounced kick-off stays unannounced"
    assert [round(nxt[1][k], 4) for k in ("ph", "pd", "pa")] == [0.5, 0.25, 0.25]


def test_clubindex_next_is_capped_at_five(tmp_path):
    out = toy_site(tmp_path)
    rows = [_match(md, f"2026-10-{md:02d}", "alpha", "beta")
            for md in range(5, 12)]
    doc = read(out, "toy-league", "matches.json")
    doc["matches"] += rows
    _write(os.path.join(out, "toy-league", "matches.json"), doc)
    extras.build_all(out, {"toy-league"}, today=TODAY)
    assert len(read(out, "clubindex.json")["clubs"]["alpha"][0]["next"]) == 5


# -------------------------------------------------------------- upcoming.json
def test_upcoming_covers_two_days_back_and_twelve_forward(tmp_path):
    out = build(tmp_path)
    doc = read(out, "upcoming.json")
    assert doc["from"] == "2026-09-02" and doc["to"] == "2026-09-16"
    dates = [m["date"] for m in doc["matches"]]
    assert min(dates) >= doc["from"] and max(dates) <= doc["to"]
    assert "2026-08-29" not in dates, "before the window"
    assert "2026-10-01" not in dates, "after the window"


def test_upcoming_is_sorted_by_date_then_time_then_competition(tmp_path):
    out = build(tmp_path)
    rows = read(out, "upcoming.json")["matches"]
    keys = [(m["date"], m["time"] or "99:99", m["slug"]) for m in rows]
    assert keys == sorted(keys)
    # An untimed fixture sorts after the timed ones on its own day.
    assert [m["h"] for m in rows if m["date"] == "2026-09-06"] == ["delta"]


def test_upcoming_carries_leverage_and_the_single_largest_swing(tmp_path):
    out = build(tmp_path)
    rows = {(m["h"], m["a"]): m for m in read(out, "upcoming.json")["matches"]}
    big = rows[("alpha", "gamma")]
    assert big["lev"] == 0.3
    # delta's 0.20 beats beta's -0.14, and the published delta is its size.
    assert big["swing"] == {"club": "delta", "event": "releg", "delta": 0.2}
    assert rows[("beta", "delta")]["swing"] is None, "no swings, no swing"


def test_upcoming_names_both_clubs_and_reports_a_finished_result(tmp_path):
    out = build(tmp_path)
    rows = {(m["h"], m["a"]): m for m in read(out, "upcoming.json")["matches"]}
    done = rows[("beta", "gamma")]
    assert done["played"] is True and (done["hg"], done["ag"]) == (1, 3)
    assert (done["hn"], done["an"]) == ("Beta FC", "Gamma FC")
    assert done["league"] == "Toy League" and done["kind"] == "league"
    assert rows[("alpha", "epsilon")]["league"] == "Toy Cup"
    assert rows[("beta", "delta")]["hg"] is None


# ---------------------------------------------------------- <slug>/recaps.json
def test_recaps_archive_only_matchweeks_that_are_finished(tmp_path):
    out = build(tmp_path)
    doc = read(out, "toy-league", "recaps.json")
    assert [r["md"] for r in doc["rounds"]] == [1], \
        "matchweek 2 is half played and matchweek 3 has not started"
    r = doc["rounds"][0]
    assert (r["from"], r["to"]) == ("2026-08-29", "2026-08-30")
    assert r["written"] == "2026-09-04"
    assert r["narrative"] == ["Alpha lead the race for the title."]
    assert r["movers"][0]["id"] == "beta"
    assert {m["h"] for m in r["matches"]} == {"alpha", "gamma"}


def test_recap_match_rows_carry_the_frozen_triple_and_the_surprise(tmp_path):
    out = build(tmp_path)
    rows = {m["h"]: m for m in
            read(out, "toy-league", "recaps.json")["rounds"][0]["matches"]}
    assert rows["alpha"]["p"] == [0.6, 0.25, 0.15], "frozen before kick-off"
    assert rows["alpha"]["called"] is True
    assert rows["alpha"]["surprise"] == 0.4, "1 - what it gave the home win"
    assert rows["gamma"]["surprise"] == 0.7, "the 0-0 was given 0.3"
    assert rows["gamma"]["called"] is False


def test_an_archived_round_is_never_rewritten(tmp_path):
    """The archive is the record. A later run must not restate it, because the
    numbers it would restate are a different week's."""
    out = build(tmp_path)
    path = os.path.join(out, "toy-league", "recaps.json")
    doc = read(out, "toy-league", "recaps.json")
    doc["rounds"][0]["narrative"] = ["as it was written at the time"]
    _write(path, doc)
    extras.build_all(out, {"toy-league", "toy-cup"}, today=TODAY)
    assert read(out, "toy-league", "recaps.json")["rounds"][0]["narrative"] == \
        ["as it was written at the time"]


def test_recaps_are_byte_identical_across_same_day_runs(tmp_path):
    """The build runs every six hours; four runs a day must not mean four
    copies of Saturday."""
    out = build(tmp_path)
    path = os.path.join(out, "toy-league", "recaps.json")
    first = open(path).read()
    for _ in range(3):
        extras.build_all(out, {"toy-league", "toy-cup"}, today=TODAY)
    assert open(path).read() == first


def test_a_round_finishing_later_is_appended_beside_the_first(tmp_path):
    out = build(tmp_path)
    doc = read(out, "toy-league", "matches.json")
    for m in doc["matches"]:
        if m["md"] == 2 and m["h"] == "delta":
            m.update(played=True, hg=2, ag=2)
    _write(os.path.join(out, "toy-league", "matches.json"), doc)
    extras.build_all(out, {"toy-league"}, today=dt.date(2026, 9, 7))
    rounds = read(out, "toy-league", "recaps.json")["rounds"]
    assert [r["md"] for r in rounds] == [1, 2]
    assert len(rounds[1]["matches"]) == 2, "both of matchweek 2, including the "\
        "one with no frozen forecast on record"
    assert rounds[1]["written"] == "2026-09-07"


def test_a_competition_that_has_not_kicked_off_gets_no_archive(tmp_path):
    out = build(tmp_path)
    assert not os.path.exists(os.path.join(out, "toy-cup", "recaps.json"))


# ------------------------------------------------------- global_history.json
def test_global_history_appends_one_column_per_iso_week(tmp_path):
    out = build(tmp_path)
    doc = read(out, "global_history.json")
    assert doc["weeks"] == ["2026-08-31"], "the Monday of the week of the 4th"
    assert doc["clubs"]["alpha"] == [80.1]
    assert set(doc) == {"generated", "weeks", "clubs"}


def test_global_history_records_a_week_once_however_often_the_build_runs(tmp_path):
    out = build(tmp_path)
    path = os.path.join(out, "global_history.json")
    first = open(path).read()
    for _ in range(3):
        extras.build_all(out, {"toy-league", "toy-cup"}, today=TODAY)
    assert open(path).read() == first
    # A day later in the same ISO week is still that week.
    extras.build_all(out, {"toy-league"}, today=dt.date(2026, 9, 6))
    assert read(out, "global_history.json")["weeks"] == ["2026-08-31"]


def test_global_history_pads_a_club_that_was_not_ranked_that_week(tmp_path):
    out = build(tmp_path)
    g = read(out, "global.json")
    g["clubs"] = [c for c in g["clubs"] if c["id"] != "delta"]
    g["clubs"].append({"id": "newcomer", "spi": 20.0, "rank": 9})
    _write(os.path.join(out, "global.json"), g)
    extras.build_all(out, {"toy-league", "toy-cup"}, today=dt.date(2026, 9, 9))
    doc = read(out, "global_history.json")
    assert doc["weeks"] == ["2026-08-31", "2026-09-07"]
    assert doc["clubs"]["delta"] == [50.2, None], "absent is not zero"
    assert doc["clubs"]["newcomer"] == [None, 20.0], "and it is not backdated"
    assert all(len(v) == 2 for v in doc["clubs"].values())


def test_global_history_stays_inside_its_budget(tmp_path):
    """981 clubs x a season of weekly columns is 226KB; the file has a budget
    of 100KB and trims the oldest column rather than refusing to record now."""
    out = toy_site(tmp_path)
    _write(os.path.join(out, "global.json"),
           {"clubs": [{"id": f"club-{i:04d}-with-a-long-name", "spi": 50.0,
                       "rank": i} for i in range(981)]})
    day = dt.date(2026, 1, 5)
    for _ in range(60):
        extras.build_all(out, {"toy-league"}, today=day)
        day += dt.timedelta(days=7)
    doc = read(out, "global_history.json")
    path = os.path.join(out, "global_history.json")
    assert os.path.getsize(path) <= extras.HISTORY_BUDGET
    assert len(doc["clubs"]) <= extras.MAX_HISTORY_CLUBS
    assert all(len(v) == len(doc["weeks"]) for v in doc["clubs"].values())


def test_every_club_in_a_forecast_competition_is_in_the_ranking_history(tmp_path):
    """The budget trims the tail of a 981-club ranking, never a club whose own
    page draws the line."""
    out = toy_site(tmp_path)
    _write(os.path.join(out, "global.json"),
           {"clubs": [{"id": f"filler-{i}", "spi": 50.0, "rank": i}
                      for i in range(extras.MAX_HISTORY_CLUBS + 50)]
           + [{"id": "alpha", "spi": 80.1, "rank": 999},
              {"id": "beta", "spi": 60.0, "rank": 1000}]})
    extras.build_all(out, {"toy-league"}, today=TODAY)
    clubs = read(out, "global_history.json")["clubs"]
    assert clubs["alpha"] == [80.1] and clubs["beta"] == [60.0]


# ------------------------------------------------------- per-fixture calendars
def test_one_calendar_per_fixture_inside_the_window(tmp_path):
    out = build(tmp_path)
    d = os.path.join(str(tmp_path), "site", "cal", "toy-league", "match")
    assert sorted(os.listdir(d)) == ["2026-09-06-delta-alpha.ics",
                                     "2026-09-12-alpha-gamma.ics",
                                     "2026-09-13-beta-delta.ics"]
    text = open(os.path.join(d, "2026-09-12-alpha-gamma.ics")).read()
    assert text.count("BEGIN:VEVENT") == 1
    assert "SUMMARY:Alpha FC v Gamma FC" in text
    assert "CATEGORIES:Matchweek 3" in text
    cup = os.path.join(str(tmp_path), "site", "cal", "toy-cup", "match")
    assert "CATEGORIES:Matchday 1" in \
        open(os.path.join(cup, "2026-09-08-alpha-epsilon.ics")).read()


def test_a_fixture_that_has_been_played_loses_its_calendar(tmp_path):
    out = build(tmp_path)
    d = os.path.join(str(tmp_path), "site", "cal", "toy-league", "match")
    stale = os.path.join(d, "2026-09-01-old-gone.ics")
    with open(stale, "w") as fh:
        fh.write("BEGIN:VCALENDAR\r\nEND:VCALENDAR\r\n")
    doc = read(out, "toy-league", "matches.json")
    for m in doc["matches"]:
        if m["date"] == "2026-09-12":
            m.update(played=True, hg=1, ag=1)
    _write(os.path.join(out, "toy-league", "matches.json"), doc)
    extras.build_all(out, {"toy-league"}, today=TODAY)
    assert not os.path.exists(stale)
    assert not os.path.exists(os.path.join(d, "2026-09-12-alpha-gamma.ics"))
    assert os.path.exists(os.path.join(d, "2026-09-13-beta-delta.ics"))


# ------------------------------------------------------------------ robustness
def test_one_broken_competition_does_not_cost_the_others_their_files(tmp_path,
                                                                    capsys):
    out = toy_site(tmp_path)
    with open(os.path.join(out, "toy-cup", "matches.json"), "w") as fh:
        fh.write("{not json at all")
    extras.build_all(out, {"toy-league", "toy-cup"}, today=TODAY)
    printed = capsys.readouterr().out
    assert "toy-cup" in printed and "skipped" in printed
    doc = read(out, "clubindex.json")
    assert [e["slug"] for e in doc["clubs"]["alpha"]] == ["toy-league"]
    assert "epsilon" not in doc["clubs"]
    assert os.path.exists(os.path.join(out, "upcoming.json"))
    assert os.path.exists(os.path.join(out, "toy-league", "recaps.json"))


def test_a_missing_global_ranking_leaves_the_history_alone(tmp_path):
    out = toy_site(tmp_path)
    os.remove(os.path.join(out, "global.json"))
    extras.build_all(out, {"toy-league"}, today=TODAY)
    assert not os.path.exists(os.path.join(out, "global_history.json"))
    assert os.path.exists(os.path.join(out, "clubindex.json")), \
        "and costs nothing else"


def test_nothing_ready_writes_nothing_and_raises_nothing(tmp_path, capsys):
    out = toy_site(tmp_path)
    extras.build_all(out, set(), today=TODAY)
    assert "nothing ready" in capsys.readouterr().out
    assert not os.path.exists(os.path.join(out, "clubindex.json"))


def test_an_unreadable_manifest_still_builds_from_the_caller_s_set(tmp_path):
    """`build_all` is handed the ready set by the run; the manifest is where the
    names come from, not where the truth does."""
    out = toy_site(tmp_path)
    with open(os.path.join(out, "leagues.json"), "w") as fh:
        fh.write("")
    extras.build_all(out, {"toy-league"}, today=TODAY)
    doc = read(out, "clubindex.json")
    assert doc["clubs"]["alpha"][0]["slug"] == "toy-league"


def test_every_file_is_compact_json_with_a_generated_stamp(tmp_path):
    out = build(tmp_path)
    for parts in (("clubindex.json",), ("upcoming.json",),
                  ("global_history.json",), ("toy-league", "recaps.json")):
        raw = open(os.path.join(out, *parts)).read()
        assert ", " not in raw and '": ' not in raw, f"{parts} is not compact"
        dt.datetime.fromisoformat(json.loads(raw)["generated"])
