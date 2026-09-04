"""What leaves the site as a subscription: calendars, and the change feed's prose.

Both failures fixed here are the kind nobody reports. A calendar with no
timezone renders perfectly on the machine that generated it and five hours wrong
in New York, and the reader blames their own client. "Wolverhampton Wanderers's
title chance" is grammatical enough to read past and wrong on every club whose
name is a plural, which in England is most of them.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import feeds                                          # noqa: E402

META = {"a": {"name": "Alpha FC"}, "b": {"name": "Beta FC"}}
TIMED = {"h": "a", "a": "b", "date": "2026-08-21", "time": "20:00", "md": 1,
         "ph": 0.5, "pd": 0.3, "pa": 0.2, "xgh": 1.7, "xga": 1.1,
         "sc": [2, 1], "scp": 0.11, "played": False, "swings": []}
UNTIMED = {**TIMED, "date": "2026-12-26", "time": None, "md": 18}


def cal(rows, ns, **kw):
    return feeds.calendar(rows, META, title="Toy", uid_ns=ns, **kw)


# ---------------------------------------------------------------- timezones
def test_a_kick_off_carries_the_competition_s_own_zone():
    """`DTSTART:20260821T200000` is RFC 5545 *floating* time: a client renders
    it in the reader's zone, so a 20:00 BST kick-off was shown at 20:00 EDT."""
    ics = cal([TIMED], "premier-league.537")
    assert "DTSTART;TZID=Europe/London:20260821T200000" in ics
    assert "DTEND;TZID=Europe/London:20260821T220000" in ics
    assert "\r\nDTSTART:20260821" not in ics


def test_each_competition_gets_the_zone_its_fixture_list_is_written_in():
    for ns, tz in (("premier-league.537", "Europe/London"),
                   ("championship.537", "Europe/London"),
                   ("la-liga.537", "Europe/Madrid"),
                   ("serie-a.537", "Europe/Rome"),
                   ("bundesliga.537", "Europe/Berlin"),
                   ("ligue-1.537", "Europe/Paris"),
                   ("eredivisie.537", "Europe/Amsterdam"),
                   ("pro-league.537", "Europe/Brussels"),
                   ("primeira-liga.537", "Europe/Lisbon"),
                   # UEFA publishes kick-offs in CET/CEST whoever is at home.
                   ("champions-league.537", "Europe/Paris")):
        assert f"DTSTART;TZID={tz}:" in cal([TIMED], ns), ns
        assert feeds.zone_for(ns) == tz


def test_a_competition_with_no_known_zone_stays_floating():
    """Inventing a zone would be worse than saying there is none."""
    ics = cal([TIMED], "toy.537")
    assert "DTSTART:20260821T200000" in ics and "TZID" not in ics


def test_an_untimed_fixture_is_still_an_all_day_event():
    """Most kick-offs are unannounced months ahead; an all-day event is the
    honest shape and needs no zone at all."""
    ics = cal([UNTIMED], "premier-league.537")
    assert "DTSTART;VALUE=DATE:20261226" in ics
    assert "BEGIN:VTIMEZONE" not in ics, "nothing timed, nothing to define"


def test_the_timezone_definition_travels_with_the_calendar():
    """Most clients resolve a TZID themselves; the ones that do not need the
    block, and the EU switch dates are the same for both zones here."""
    ics = cal([TIMED, UNTIMED], "premier-league.537")
    assert ics.count("BEGIN:VTIMEZONE") == 1
    assert "TZID:Europe/London" in ics
    assert "TZNAME:BST" in ics and "TZNAME:GMT" in ics
    assert "RRULE:FREQ=YEARLY;BYMONTH=3;BYDAY=-1SU" in ics
    cet = cal([TIMED], "la-liga.537")
    assert "TZNAME:CEST" in cet and "TZOFFSETTO:+0200" in cet


def test_the_calendar_is_still_well_formed_with_a_zone_on_it():
    ics = cal([TIMED, UNTIMED], "bundesliga.537")
    assert ics.startswith("BEGIN:VCALENDAR\r\n") and ics.endswith(
        "END:VCALENDAR\r\n")
    assert ics.count("BEGIN:VEVENT") == ics.count("END:VEVENT") == 2
    assert ics.count("BEGIN:VTIMEZONE") == ics.count("END:VTIMEZONE") == 1
    for line in ics.split("\r\n"):
        assert len(line.encode("utf-8")) <= 75, line


def test_one_fixture_makes_one_valid_calendar():
    """What the match dialog's "add to calendar" hands a reader, instead of a
    subscription to a whole club's season."""
    ics = cal([TIMED], "premier-league.537")
    assert ics.count("BEGIN:VEVENT") == 1
    assert "UID:a-b-2026-08-21@premier-league.537" in ics
    assert "SUMMARY:Alpha FC v Beta FC" in ics


def test_filtering_to_one_club_does_not_drag_in_a_pointless_timezone():
    rows = [{**TIMED, "h": "c", "a": "b"}, UNTIMED]
    ics = feeds.calendar(rows, META, title="t", uid_ns="premier-league.537",
                         team="b")
    assert ics.count("BEGIN:VEVENT") == 2 and "BEGIN:VTIMEZONE" in ics
    only_untimed = feeds.calendar([UNTIMED], META, title="t",
                                  uid_ns="premier-league.537", team="b")
    assert "BEGIN:VTIMEZONE" not in only_untimed


# --------------------------------------------------------------- possessives
def test_a_plural_club_name_takes_a_bare_apostrophe():
    assert feeds.possessive("Wolverhampton Wanderers") == \
        "Wolverhampton Wanderers'"
    assert feeds.possessive("Arsenal") == "Arsenal's"
    assert feeds.possessive("Queens Park Rangers") == "Queens Park Rangers'"
    assert feeds.possessive("Barcelona") == "Barcelona's"


def test_the_change_feed_does_not_write_wanderers_s():
    meta = {"w": {"name": "Wolverhampton Wanderers"}, "a": {"name": "Arsenal"}}
    rows = [{"slug": "x", "name": "X", "url": "u", "meta": meta,
             "recap": {"asof": "2026-09-01", "movers": [
                 {"id": "w", "metric": "releg", "delta": 0.12, "before": 0.2,
                  "after": 0.32},
                 {"id": "a", "metric": "title", "delta": 0.05, "before": 0.4,
                  "after": 0.45}]}}]
    text = feeds.feed_items(rows)[0]["content_text"]
    assert "Wanderers's" not in text
    assert "Wolverhampton Wanderers' relegation risk rose" in text
    assert "Arsenal's title chance rose" in text


def test_the_weekly_narrative_uses_the_same_possessive():
    from model import recap
    movers = [{"id": "w", "metric": "ucl", "before": 0.2, "after": 0.32,
               "delta": 0.12}]
    lines = recap.narrative(
        [{"id": "w", "title": 0.1, "ucl": 0.32, "releg": 0.3},
         {"id": "a", "title": 0.5, "ucl": 0.9, "releg": 0.0}],
        movers, [], {"date": "2026-08-29"},
        {"w": "Wolverhampton Wanderers", "a": "Arsenal"})
    assert any("Wolverhampton Wanderers' chance of qualifying rose" in ln
               for ln in lines)
    assert not any("Wanderers's" in ln for ln in lines)
