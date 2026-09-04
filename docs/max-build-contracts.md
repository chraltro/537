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


## 8. `sharpen` — the per-league calibration exponent

Added by the model agent in the September round. One number per competition,
fitted on that league's own past seasons and applied to every published match
probability, so anything that recomputes a probability has to apply it too or
it will disagree with the page it is drawn on.

### The formula, exactly

Given the model's three outcome probabilities `p = [pH, pD, pA]` (which sum to
1) and the exponent `k`:

```
q_i = p_i ** k                     for i in (home, draw, away)
out_i = q_i / (q_H + q_D + q_A)
```

That is the whole of it. `k > 1` sharpens (pushes away from the base rate),
`k < 1` softens, **`k == 1` is exactly the identity and must be skipped** — and
a **missing `sharpen` field means 1.0**, i.e. do nothing. Never apply it twice:
every probability the pipeline publishes (`matches.json` `ph`/`pd`/`pa` and
`grid`, `forecast.json` team probabilities, `predictions.json`) already has it
applied.

**Worked example**, `sharpen = 1.15`, input `[0.55, 0.25, 0.20]`:

```
0.55 ** 1.15 = 0.50282527
0.25 ** 1.15 = 0.20306310
0.20 ** 1.15 = 0.15710301
sum          = 0.86299137
out = [0.582654, 0.235301, 0.182045]      (sums to 1)
```

### What it is applied to

The exponent is measured on match outcomes, so it is applied to the **1X2
triple**, and the score matrix is then reweighted to match it rather than being
sharpened cell by cell. Each cell is scaled by its own outcome class's factor:

```
class(i, j) = home if i > j, draw if i == j, away if i < j
p_class     = sum of grid cells in that class          (the three numbers above)
q_class     = sharpen_probs(p_class, k)
grid'[i,j]  = grid[i,j] * q_class(i,j) / p_class(i,j)   then renormalise
```

So the distribution of scorelines *within* a result is untouched and only the
weight of the three results moves. A simulator that samples scorelines (the
what-if worker) should do the same thing to its per-fixture grid before taking
the CDF; a page that only needs 1X2 can use the three-number form. Both give
the same answer.

Example on a real grid (λ 1.62 / 1.05, ρ 0.03), `k = 1.15`:
`[0.5099, 0.2406, 0.2495]` → `[0.5373, 0.2265, 0.2362]`; every home-win cell is
multiplied by 1.0537, every draw cell by 0.9415.

### Where the number comes from and where it is published

* Fitted by `backtest.fit_sharpen` on the league's whole walk-forward, and
  evaluated honestly on a chronological 60/40 split that the shipped number
  never sees. Clamped to `[0.70, 1.40]`; not fitted at all below 750 scored
  matches, which is why the Champions League ships `1.0`; and **not shipped at
  all unless that held-out split gains** — a league whose exponent does not
  generalise publishes `sharpen: 1.0` with `backtest.json` → `sharpen.applied:
  false`, `sharpen.measured_k` and a reason. So `sharpen` is 1.0 for more
  competitions than not, and a page that handles 1.0 as "do nothing" is right
  most of the time by construction.
* Published in three places, all the same number:
  `sim_input.json` → `sharpen`, `forecast.json` → `sharpen`,
  `backtest.json` → `params.sharpen` (with the evidence under `sharpen`).

---

## 9. Other fields the model round added or changed

Everything below is additive: no existing field changed name, type or meaning.

**`forecast.json`**

| Field | Where | What |
|---|---|---|
| `sharpen` | top level | §8. Present for leagues and cups. |
| `clinch` | each team row | §5. `{"title": {...}, "top": {...}, "safe": {...}}`, each `{"done": bool, "need": int\|null}`. Cups map to first in the league phase / the direct places / avoiding elimination. `need` is null when no attainable points total guarantees it (which is most of the table in August — the bound is conservative and the page should say so). |
| `played`, `w`, `d`, `l`, `gf`, `ga`, `cur_pts` | each **cup** team row | Were hardcoded to 0 and are now the real league-phase table. Domestic rows are unchanged. |

**`matches.json`** — unchanged in shape. `ph`/`pd`/`pa`/`grid`/`alt`/`o25`/
`btts`/`csh`/`csa` now carry the calibrated probabilities (§8); `xgh`/`xga` are
the model's λ and are *not* calibrated.

**`sim_input.json`** — one new top-level field, `sharpen` (§8). Everything else
unchanged.

**`backtest.json`**

| Field | What |
|---|---|
| `model` | **now the model the site publishes**: the same walk-forward, predicted through the measured preseason correction (`priors.preseason_net`) exactly as the live build applies it. |
| `model_ratings_only` | the previous `model`: the same walk-forward straight off the ratings, no correction. |
| `held_out_ratings_only` | the same split for the ratings-only model. |
| `adjusted`, `adjustment_note` | whether the correction was applied, and the note explaining that the *market* anchor is not in the backtest (no historical odds snapshot exists to walk forward over). |
| `sharpen` | `{k, n, band, in_sample, held_out{k, n, train_n, log_loss, calibrated, gain}, note}` — the evidence behind §8. |
| `params.sharpen`, `params.ridge` | the shipped exponent and the ridge, so the method page can quote numbers rather than describe them. |
| `calibration`, `by_outcome` | unchanged shape; now measured on the published (adjusted) model. |

**`<slug>/rooting.json`** — new file, §4, with one deliberate deviation from
the example there. Each club's effects are keyed by **the competition's own
event keys** — `["title","ucl","releg"]` for a league, `["top8","qualify","out"]`
for a cup — which are exactly the keys `matches.json`'s `swings` already uses,
so the site labels them with `eventLabels()` and no new mapping. `events` at the
top of the file names them, and `contract_events` gives §4's positional names
`["title","top","out"]` in the same order for a reader that wants them.

The reason for the deviation is a real collision: in a cup's league-phase
simulation the first event is *finishing in the top eight*, not winning the
trophy — and "title" in the same competition's `forecast.json` is the trophy.
Publishing that as `title` would have made two files disagree about one word.

`md` is the matchweek the file covers (next unplayed matchday only). Effects
below 0.0005 in absolute size are dropped, a club with no surviving effect is
absent from `effects`, and a club playing in the match is never in its own
`effects`. The file is deleted when a competition has no fixtures left.
