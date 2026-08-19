"""Tests for the browser simulator's input file.

sim_input.json is the only thing standing between the published forecast and a
page that re-runs the season itself. If a lambda goes missing, or the standings
carried into the client are wrong, every conditional table the reader sees is
quietly wrong too — and nothing on the server would notice.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import config, ratings, siminput                       # noqa: E402
from model.data import Dataset                                     # noqa: E402
from model.parse import Match                                      # noqa: E402


@pytest.fixture(scope="module")
def ds() -> Dataset:
    return Dataset().load()


# ---------------------------------------------------------------- synthetic
TEAMS = ["alpha", "bravo", "charlie", "delta"]
META = {t: {"name": t.title(), "short": t[:3].upper(), "primary": "#3987e5"} for t in TEAMS}


def _fit() -> ratings.Fit:
    return ratings.Fit(TEAMS,
                       np.array([0.35, 0.10, -0.10, -0.35]),
                       np.array([0.30, 0.05, -0.05, -0.30]),
                       float(np.log(1.35)), 0.25, -0.06)


def _fixtures() -> list[Match]:
    d = dt.date(2026, 8, 21)
    return [
        # played: alpha 3-1 bravo, charlie 0-0 delta
        Match(date=d, home="alpha", away="bravo", hg=3, ag=1, matchday=1, played=True),
        Match(date=d, home="charlie", away="delta", hg=0, ag=0, matchday=1, played=True),
        # played next week: bravo 2-2 charlie
        Match(date=d + dt.timedelta(days=7), home="bravo", away="charlie",
              hg=2, ag=2, matchday=2, played=True),
        # still to come
        Match(date=d + dt.timedelta(days=7), home="delta", away="alpha", matchday=2),
        Match(date=d + dt.timedelta(days=14), home="alpha", away="charlie", matchday=3),
        Match(date=d + dt.timedelta(days=14), home="bravo", away="delta", matchday=3),
    ]


@pytest.fixture()
def written(tmp_path):
    path = tmp_path / "sim_input.json"
    siminput.write_sim_input(_fit(), _fixtures(), TEAMS, {"alpha": 0.2}, META, str(path))
    return path


def test_file_is_valid_json_with_the_expected_shape(written):
    doc = json.loads(written.read_text())
    assert set(doc) == {"generated", "rho", "ucl_places", "releg_places",
                        "n_teams", "teams", "fixtures"}
    assert doc["rho"] == pytest.approx(-0.06)
    dt.datetime.fromisoformat(doc["generated"])          # parses, or this raises
    assert len(doc["teams"]) == len(TEAMS)
    for t in doc["teams"]:
        assert set(t) == {"id", "name", "short", "primary", "pts", "gf", "ga", "played"}
    assert len(doc["fixtures"]) == len(_fixtures())
    for f in doc["fixtures"]:
        assert {"h", "a", "md", "date", "played"} <= set(f)


def test_league_shape_travels_with_the_payload(tmp_path):
    """The browser worker draws the European and relegation lines from this
    file, so a Bundesliga payload has to say 18 clubs and two down."""
    from model import leagues
    path = tmp_path / "sim_input.json"
    siminput.write_sim_input(_fit(), _fixtures(), TEAMS, {}, META, str(path),
                             league=leagues.BUNDESLIGA)
    doc = json.loads(path.read_text())
    assert (doc["n_teams"], doc["ucl_places"], doc["releg_places"]) == (18, 4, 2)


def test_lambdas_are_present_only_for_unplayed_fixtures(written):
    doc = json.loads(written.read_text())
    for f in doc["fixtures"]:
        if f["played"]:
            assert "lh" not in f and "la" not in f
            assert isinstance(f["hg"], int) and isinstance(f["ag"], int)
        else:
            assert "hg" not in f and "ag" not in f
            assert f["lh"] > 0 and f["la"] > 0
            # 4dp, and no more: the payload is fetched before anything is drawn
            assert round(f["lh"], 4) == f["lh"] and round(f["la"], 4) == f["la"]


def test_lambdas_match_the_pipeline_including_the_rating_adjustment(written):
    from model import simulate
    doc = json.loads(written.read_text())
    fit, adj = _fit(), {"alpha": 0.2}
    for f in doc["fixtures"]:
        if f["played"]:
            continue
        lh, la = simulate._lambdas(fit, f["h"], f["a"], adj)
        assert f["lh"] == pytest.approx(round(lh, 4))
        assert f["la"] == pytest.approx(round(la, 4))


def test_standings_arithmetic(written):
    """alpha 3-1 bravo, charlie 0-0 delta, bravo 2-2 charlie."""
    doc = json.loads(written.read_text())
    rows = {t["id"]: t for t in doc["teams"]}
    assert (rows["alpha"]["pts"], rows["alpha"]["gf"], rows["alpha"]["ga"],
            rows["alpha"]["played"]) == (3, 3, 1, 1)
    assert (rows["bravo"]["pts"], rows["bravo"]["gf"], rows["bravo"]["ga"],
            rows["bravo"]["played"]) == (1, 3, 5, 2)
    assert (rows["charlie"]["pts"], rows["charlie"]["gf"], rows["charlie"]["ga"],
            rows["charlie"]["played"]) == (2, 2, 2, 2)
    assert (rows["delta"]["pts"], rows["delta"]["gf"], rows["delta"]["ga"],
            rows["delta"]["played"]) == (1, 0, 0, 1)


def test_fixtures_are_ordered_by_matchweek_then_date_then_home(written):
    doc = json.loads(written.read_text())
    keys = [(f["md"], f["date"], f["h"]) for f in doc["fixtures"]]
    assert keys == sorted(keys)


def test_teams_carry_the_metadata_the_page_renders(written):
    doc = json.loads(written.read_text())
    for t in doc["teams"]:
        assert t["name"] == META[t["id"]]["name"]
        assert t["short"] == META[t["id"]]["short"]
        assert t["primary"].startswith("#")


# ---------------------------------------------------------------- real season
def test_real_season_payload_is_complete(ds, tmp_path):
    fit = ratings.fit([m for m in ds.pl if m.season >= "2024-25"],
                      ds.teams, dt.date(2026, 8, 21))
    path = tmp_path / "sim_input.json"
    siminput.write_sim_input(fit, ds.fixtures, ds.teams, {}, ds.reg.meta, str(path))
    doc = json.loads(path.read_text())

    assert len(doc["teams"]) == config.N_TEAMS
    assert len(doc["fixtures"]) == config.N_MATCHES
    assert {t["id"] for t in doc["teams"]} == set(ds.teams)

    ids = {t["id"] for t in doc["teams"]}
    for f in doc["fixtures"]:
        assert f["h"] in ids and f["a"] in ids
        if not f["played"]:
            assert f["lh"] > 0, f
            assert f["la"] > 0, f
            # a Premier League fixture nobody would recognise is a bug upstream
            assert 0.1 < f["lh"] < 6 and 0.1 < f["la"] < 6

    played = sum(1 for f in doc["fixtures"] if f["played"])
    assert sum(t["played"] for t in doc["teams"]) == 2 * played
    # The file is fetched before the page draws anything, so its size is part of
    # the contract, not an incidental detail.
    assert os.path.getsize(path) < 120_000
