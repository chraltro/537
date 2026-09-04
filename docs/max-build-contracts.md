# Build contracts — the September 2026 improvement round

Two agents build in parallel. The model agent owns `model/`, `tests/` and the
generated JSON; the site agent owns `site/*.html` and `site/assets/*`. They never
touch each other's files, so the JSON shapes below are the contract between
them, fixed by the lead before either starts. A shape here is binding; a field
not here does not exist until both sides agree in this file.

All files are written by `python -m model.run` (the default, all-competitions
run) into `site/data/`, compact JSON (`separators=(",", ":")`), and every file
carries `"generated": ISO-8601 UTC`. Club ids are registry ids (`arsenal`,
`real-madrid`); competition slugs are manifest slugs (`premier-league`,
`champions-league`).

## 1. `site/data/clubindex.json` — one club, every competition

```json
{"generated": "...",
 "clubs": {
   "arsenal": [
     {"slug": "premier-league", "name": "Premier League", "kind": "league",
      "pos": 1, "pts": 9, "played": 3, "spi": 91.2,
      "title": 0.43, "top": 0.98, "out": 0.0,
      "next": [{"opp": "man-city", "opp_name": "Manchester City", "home": true,
                "date": "2026-09-13", "time": "16:30", "md": 4,
                "ph": 0.48, "pd": 0.25, "pa": 0.27}]},
     {"slug": "champions-league", "name": "Champions League", "kind": "cup",
      "pos": null, "pts": 0, "played": 0, "spi": 91.2,
      "title": 0.30, "top": 0.80, "out": 0.004, "next": [...]}
   ]}}
```

* Only competitions with `ready: true` in the manifest. `title`/`top`/`out` are
  the competition's three headline events (title / top-N or top-8 / relegation
  or elimination) so the site can label them through `eventLabels()`.
* `next` holds up to 5 unplayed fixtures in that competition, soonest first.
* `pos` is the current table position (null before a ball is kicked).

## 2. `site/data/upcoming.json` — this week across Europe

```json
{"generated": "...", "from": "2026-09-02", "to": "2026-09-14",
 "matches": [
   {"slug": "champions-league", "league": "Champions League", "kind": "cup",
    "md": 1, "date": "2026-09-08", "time": "21:00",
    "h": "real-madrid", "a": "inter", "hn": "Real Madrid", "an": "Inter",
    "ph": 0.46, "pd": 0.26, "pa": 0.28,
    "lev": 0.31, "swing": {"club": "inter", "event": "top", "delta": 0.14},
    "played": false, "hg": null, "ag": null}
 ]}
```

* Window: from two days ago to twelve days ahead, all ready competitions,
  sorted by date then time then league. `lev` is the match's existing leverage
  score; `swing` is the single largest event swing the match carries (club,
  event key, absolute probability delta) or null. Times are local kick-off
  times as the fixture files carry them, with no timezone conversion.

## 3. `site/data/<slug>/recaps.json` — the matchweek in review, archived

```json
{"generated": "...",
 "rounds": [
   {"md": 3, "from": "2026-09-01", "to": "2026-09-03", "written": "2026-09-04",
    "narrative": "...", "movers": [...], "shocks": [...],
    "matches": [{"h": "...", "a": "...", "hg": 2, "ag": 1, "p": [0.5, 0.25, 0.25],
                 "called": true, "surprise": 0.31}]}
 ]}
```

* One entry per completed matchweek, appended when the round is fully played
  and never rewritten afterwards (same append discipline as
  `insight.append_history`). `movers`, `shocks` and the match rows carry exactly
  the shapes `recap.json` and `season_report.json` already use for the current
  round. The CI persist step commits this file alongside `history.json`.

## 4. `site/data/<slug>/rooting.json` — who you want to win

```json
{"generated": "...", "md": 4,
 "events": ["title", "top", "out"],
 "matches": [
   {"h": "chelsea", "a": "liverpool", "date": "2026-09-12",
    "effects": {"arsenal": {"title": [0.012, 0.001, -0.014],
                            "top": [0.0, 0.0, 0.0], "out": [0.0, 0.0, 0.0]}}}
 ]}
```

* Next matchweek only. For every club NOT playing in the match, the change in
  its probability of each event if the match ends home win / draw / away win,
  relative to the unconditional forecast. Clubs playing in the match are
  omitted (their own leverage already covers them). Values below 0.0005 in
  absolute size are dropped to keep the file small.

## 5. `forecast.json` team rows gain `clinch`

```json
"clinch": {"title": {"done": false, "need": 71}, "top": {"done": false, "need": 62},
           "safe": {"done": false, "need": 38}}
```

* `need` is the fewest further points that guarantee the outcome on the
  simple bound (a rival's maximum is its current points plus three per
  remaining match; rivals playing each other is ignored, so the bound is
  conservative and the site says so). `done` is true once the outcome is
  mathematically settled; `need` is null when it can no longer be reached.
  Cups: `title` means first in the league phase, `top` the direct places,
  `safe` avoiding elimination.

## 6. `site/data/global_history.json` — the ranking, week by week

```json
{"generated": "...", "weeks": ["2026-08-31", "2026-09-07"],
 "clubs": {"arsenal": [91.1, 91.4], "bodo-glimt": [58.2, null]}}
```

* One column per ISO week (the Monday), appended by `build_rankings` when the
  week has no column yet; a club absent from that week's ranking gets null.
  Persisted by CI like `history.json`. Kept under 100KB per season.

## 7. `site/match/*.html` — static fixture stubs

Generated by `model/seo.py` for every unplayed fixture in the next 14 days
across all ready competitions: `site/match/<slug>/<date>-<h>-<a>.html`, with
title, description, share card and a redirect to the live match modal deep
link the site already supports. Listed in `sitemap.xml`; pruned when the
fixture is played or the window moves on.

## Site-only features (no contract needed)

* "How the forecast has changed" per club from `history.json`.
* Club character tiles from `gamestate.json` `profile` / `profile_average`.
* League strength table from `global.json`.
* Alternate tables (actual / form / home / away / since MW) from `matches.json`,
  loaded lazily on first toggle.
* Fixture congestion strip from `clubindex.json` once it exists.
