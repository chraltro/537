"""The static stubs, the sitemap, and the three rules they used to break.

A stub page exists for the readers that are not people: a crawler, a link
preview, a browser with JavaScript off. All three failure modes here are
invisible to a person clicking around the site, which is exactly why they
survived: an absolute redirect works perfectly on the production host and
nowhere else, a canonical chain looks like a working link, and "projected to
finish 1" reads as a typo rather than as 210 pages of one.

Everything writes into `tmp_path` -- `seo.SITE` is monkeypatched -- so no test
here can touch the real `site/`.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import seo                                            # noqa: E402

TODAY = dt.date(2026, 9, 4)

LEAGUE = {"slug": "toy-league", "name": "Toy League", "ready": True,
          "kind": "league"}
CUP = {"slug": "toy-cup", "name": "Toy Cup", "ready": True, "kind": "cup"}
MANIFEST = {"default": "toy-league", "leagues": [LEAGUE, CUP,
                                                 {"slug": "asleep",
                                                  "name": "Asleep",
                                                  "ready": False,
                                                  "kind": "league"}]}


def _club(tid, name, **kw):
    row = {"id": tid, "name": name, "pts": 73.4, "title": 0.43, "ucl": 0.94,
           "releg": 0.0001}
    row.update(kw)
    return row


FORECASTS = {
    "toy-league": {"season": "2026/27", "teams": [
        _club("alpha", "Alpha FC"),
        _club("wanderers", "Wolverhampton Wanderers", pts=41.2, title=0.0,
              ucl=0.02, releg=0.31)]},
    "toy-cup": {"season": "2026/27", "teams": [
        _club("alpha", "Alpha FC", pts=12.0, title=0.3, ucl=0.8,
              releg=0.004)]},
}


def _matches(root):
    """A `site/data` tree with fixtures inside, on and outside the window."""
    out = os.path.join(str(root), "data")
    rows = [
        {"md": 3, "date": "2026-09-02", "time": "20:00", "h": "alpha",
         "a": "wanderers", "ph": 0.6, "pd": 0.25, "pa": 0.15, "sc": [2, 1],
         "scp": 0.11, "played": True, "hg": 2, "ag": 1, "swings": []},
        {"md": 4, "date": "2026-09-12", "time": "16:30", "h": "alpha",
         "a": "wanderers", "ph": 0.481, "pd": 0.252, "pa": 0.267,
         "sc": [2, 1], "scp": 0.09, "played": False, "hg": None, "ag": None,
         "swings": [{"team": "wanderers", "event": "releg", "home": 0.4,
                     "away": 0.26, "swing": 0.14}]},
        {"md": 5, "date": "2026-09-18", "time": None, "h": "wanderers",
         "a": "alpha", "ph": 0.3, "pd": 0.3, "pa": 0.4, "sc": [1, 1],
         "scp": 0.13, "played": False, "hg": None, "ag": None, "swings": []},
        {"md": 9, "date": "2026-11-01", "time": "15:00", "h": "alpha",
         "a": "wanderers", "ph": 0.5, "pd": 0.25, "pa": 0.25, "sc": [1, 0],
         "scp": 0.1, "played": False, "hg": None, "ag": None, "swings": []},
    ]
    os.makedirs(os.path.join(out, "toy-league"), exist_ok=True)
    with open(os.path.join(out, "toy-league", "matches.json"), "w") as fh:
        json.dump({"matches": rows}, fh)
    os.makedirs(os.path.join(out, "toy-cup"), exist_ok=True)
    with open(os.path.join(out, "toy-cup", "matches.json"), "w") as fh:
        json.dump({"matches": [
            {"md": 1, "date": "2026-09-08", "time": "21:00", "h": "alpha",
             "a": "beta", "ph": 0.45, "pd": 0.27, "pa": 0.28, "sc": [1, 1],
             "scp": 0.12, "played": False, "hg": None, "ag": None,
             "swings": []}]}, fh)
    os.makedirs(os.path.join(out, "league"), exist_ok=True)
    for slug in ("elsewhere-first", "elsewhere-second"):
        with open(os.path.join(out, "league", f"{slug}.json"), "w") as fh:
            json.dump({"slug": slug, "name": slug, "table": []}, fh)
    with open(os.path.join(out, "leagues.json"), "w") as fh:
        json.dump({"rated": [{"name": "Third", "slug": "elsewhere-third"},
                             {"name": "Unlinkable", "slug": ""}]}, fh)
    return out


@pytest.fixture()
def site(tmp_path, monkeypatch):
    monkeypatch.setattr(seo, "SITE", str(tmp_path))
    monkeypatch.setattr(seo, "MATCH_DAYS", 14)
    out = _matches(tmp_path)
    seo.match_stubs.__defaults__  # noqa: B018  (documented: today defaults)
    monkeypatch.setattr(seo.dt, "date", _FrozenDate)
    seo.build(out, MANIFEST, FORECASTS)
    return tmp_path


class _FrozenDate(dt.date):
    """`seo.build` stamps the sitemap and windows the fixtures with today."""
    @classmethod
    def today(cls):
        return TODAY


def read(root, *parts):
    with open(os.path.join(str(root), *parts), encoding="utf-8") as fh:
        return fh.read()


# ------------------------------------------------------------------ ordinals
def test_a_projected_finish_is_an_ordinal():
    """"Projected to finish 1, on 73 points" shipped on 210 pages."""
    assert [seo._ord(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 23, 24, 111)] == \
        ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "23rd",
         "24th", "111th"]


def test_the_club_stub_says_the_ordinal_not_the_number(site):
    html = read(site, "club", "alpha-toy-league.html")
    assert "projected to finish 1st, on 73 points" in html
    assert "finish 1," not in html


# ---------------------------------------------------------------- club stubs
def test_the_club_stub_redirect_is_relative(site):
    """An absolute redirect walked every reader on a fork, a preview deploy or
    a local server off to a host they could not reach."""
    html = read(site, "club", "alpha-toy-league.html")
    assert 'content="0; url=../team.html?t=alpha&amp;lg=toy-league"' in html
    assert "https://chraltro.github.io/537/team.html" not in html


def test_the_club_stub_is_its_own_canonical(site):
    """It used to point at `team.html?t=…`, whose own canonical is `team.html`
    -- a chain that collapsed 210 pages into one URL."""
    html = read(site, "club", "alpha-toy-league.html")
    assert ('<link rel="canonical" href="https://chraltro.github.io/537/club/'
            'alpha-toy-league.html">') in html


def test_the_club_stub_has_content_of_its_own(site):
    """A page whose only content is a redirect is a doorway page."""
    html = read(site, "club", "wanderers-toy-league.html")
    body = html.split("<body>")[1]
    assert "Wolverhampton Wanderers" in body
    assert "Projected finish: 2nd, on 41 points." in body
    assert "Title 0%" in body and "relegated 31%" in body


def test_the_club_stub_names_the_other_competition_the_club_is_in(site):
    """Arsenal have two club pages that did not know the other existed, on the
    week the Champions League starts."""
    pl = read(site, "club", "alpha-toy-league.html")
    assert "Also in the Toy Cup this season." in pl
    assert 'href="alpha-toy-cup.html"' in pl
    cup = read(site, "club", "alpha-toy-cup.html")
    assert "Also in the Toy League this season." in cup
    # A club in one competition says nothing about competitions it is not in.
    assert "Also in" not in read(site, "club", "wanderers-toy-league.html")
    ld = json.loads(re.search(r'application/ld\+json">(.*?)</script>', pl)[1])
    assert [m["name"] for m in ld["memberOf"]] == ["Toy League", "Toy Cup"]


def test_a_cup_stub_does_not_talk_about_relegation(site):
    """The three keys are the same in every competition and mean something
    different in each -- which is the one mistake the rest of the site avoids."""
    cup = read(site, "club", "alpha-toy-cup.html")
    assert "Win it 30%" in cup and "eliminated" in cup
    assert "relegated" not in cup


# --------------------------------------------------------------- match stubs
def test_a_stub_exists_for_every_unplayed_fixture_in_the_window(site):
    assert sorted(os.listdir(os.path.join(str(site), "match", "toy-league"))) \
        == ["2026-09-12-alpha-wanderers.html", "2026-09-18-wanderers-alpha.html"]
    assert os.listdir(os.path.join(str(site), "match", "toy-cup")) == \
        ["2026-09-08-alpha-beta.html"]


def test_a_played_fixture_and_a_distant_one_get_no_stub(site):
    """Two days ago is behind the window; 1 November is beyond it."""
    files = os.listdir(os.path.join(str(site), "match", "toy-league"))
    assert not any(f.startswith("2026-09-02") for f in files)
    assert not any(f.startswith("2026-11-01") for f in files)


def test_stubs_outside_the_window_are_pruned_on_the_next_build(site,
                                                               monkeypatch):
    d = os.path.join(str(site), "match", "toy-league")
    stale = os.path.join(d, "2026-08-01-alpha-wanderers.html")
    with open(stale, "w") as fh:
        fh.write("<!doctype html><p>last month</p>")
    monkeypatch.setattr(seo.dt, "date", _FrozenDate)
    seo.build(os.path.join(str(site), "data"), MANIFEST, FORECASTS)
    assert not os.path.exists(stale)
    assert os.path.exists(os.path.join(d, "2026-09-12-alpha-wanderers.html"))


def test_the_match_stub_puts_the_probabilities_into_words(site):
    html = read(site, "match", "toy-league", "2026-09-12-alpha-wanderers.html")
    desc = re.search(r'<meta name="description" content="([^"]+)"', html)[1]
    assert "Alpha FC v Wolverhampton Wanderers" in desc
    assert "Toy League matchweek 4" in desc
    assert "Saturday 12 September 2026, 16:30 kick-off" in desc
    assert ("The model gives Alpha FC 48%, the draw 25% and Wolverhampton "
            "Wanderers 27%") in desc
    assert "the likeliest score is 2–1 at 9%" in desc
    assert "percentage points" in html, "and what the result is worth"


def test_the_match_stub_redirects_to_the_deep_link_the_site_already_supports(site):
    """`matches.html?m=home--away` is the link the match dialog itself shares."""
    html = read(site, "match", "toy-league", "2026-09-12-alpha-wanderers.html")
    assert ('content="0; url=../../matches.html?m=alpha--wanderers'
            '&amp;lg=toy-league"') in html
    assert ('<link rel="canonical" href="https://chraltro.github.io/537/match/'
            'toy-league/2026-09-12-alpha-wanderers.html">') in html
    body = html.split("<body>")[1]
    assert "Alpha FC win 48%" in body and "draw 25%" in body
    assert '<a href="../../club/alpha-toy-league.html">' in body


def test_a_cup_stub_counts_matchdays_not_matchweeks(site):
    html = read(site, "match", "toy-cup", "2026-09-08-alpha-beta.html")
    assert "Toy Cup matchday 1" in html and "matchweek" not in html


def test_the_stub_falls_back_to_the_competition_card_when_a_club_has_none(site):
    html = read(site, "match", "toy-league", "2026-09-12-alpha-wanderers.html")
    assert 'og:image" content="https://chraltro.github.io/537/og/toy-league.png' \
        in html
    os.makedirs(os.path.join(str(site), "og", "toy-league"), exist_ok=True)
    with open(os.path.join(str(site), "og", "toy-league", "alpha.png"), "wb") as f:
        f.write(b"\x89PNG")
    assert seo._card("toy-league", "alpha").endswith("og/toy-league/alpha.png")


# ------------------------------------------------------------------ sitemap
def _locs(site):
    return re.findall(r"<loc>(.*?)</loc>", read(site, "sitemap.xml"))


def test_the_sitemap_lists_the_stubs_it_used_to_omit(site):
    locs = set(_locs(site))
    assert "https://chraltro.github.io/537/club/alpha-toy-league.html" in locs
    assert "https://chraltro.github.io/537/club/alpha-toy-cup.html" in locs
    assert ("https://chraltro.github.io/537/match/toy-league/"
            "2026-09-12-alpha-wanderers.html") in locs


def test_the_sitemap_lists_every_league_that_is_rated_and_not_forecast(site):
    """Fifty-one competitions had a page each, nothing linking to them and no
    sitemap entry: content that does not exist as far as a crawler knows."""
    locs = set(_locs(site))
    for slug in ("elsewhere-first", "elsewhere-second", "elsewhere-third"):
        assert f"https://chraltro.github.io/537/projection.html?league={slug}" \
            in locs
    assert not [u for u in locs if u.endswith("projection.html?league=")]


def test_the_sitemap_still_holds_every_page_of_every_ready_competition(site):
    locs = _locs(site)
    assert len(locs) == len(set(locs)), "no duplicate URLs"
    for page in seo.LEAGUE_PAGES:
        assert f"https://chraltro.github.io/537/{page}" in locs, "the default"
        assert f"https://chraltro.github.io/537/{page}?lg=toy-cup" in locs
        assert f"https://chraltro.github.io/537/{page}?lg=asleep" not in locs
    for page in seo.SITE_PAGES:
        assert f"https://chraltro.github.io/537/{page}" in locs
    assert read(site, "sitemap.xml").startswith('<?xml version="1.0"')


def test_build_reports_what_it_wrote(site):
    out = seo.build(os.path.join(str(site), "data"), MANIFEST, FORECASTS)
    assert out["leagues"] == 2 and out["stubs"] == 3
    assert out["matches"] >= 3 and out["urls"] == out["stubs"] + out["matches"] + 3


def test_a_competition_with_no_fixture_file_costs_nothing(site, monkeypatch):
    os.remove(os.path.join(str(site), "data", "toy-cup", "matches.json"))
    monkeypatch.setattr(seo.dt, "date", _FrozenDate)
    out = seo.build(os.path.join(str(site), "data"), MANIFEST, FORECASTS)
    assert out["stubs"] == 3
    assert os.path.exists(os.path.join(str(site), "match", "toy-league",
                                       "2026-09-12-alpha-wanderers.html"))
