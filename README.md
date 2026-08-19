# 2026/27 Football Forecast

A rebuild of what FiveThirtyEight's Soccer Power Index did for club football: a rating for
every club, a probability for every match, and a simulated final table — for the 2026/27
Premier League, La Liga, Serie A, Bundesliga, Ligue 1, Eredivisie, Primeira Liga and
Championship seasons, plus the Champions League. One model, nine competitions, one site
with a switcher — and one cross-league rating that puts all of their clubs in a single
ordering.

**→ [chraltro.github.io/537](https://chraltro.github.io/537/)**

Everything runs from public match data, with no API keys, no paid feeds and no hand-tuned
opinions. It rebuilds itself every six hours on GitHub Actions and publishes to GitHub Pages.

## What it does

- **Ratings.** A time-weighted Dixon–Coles model gives each club an attack and a defence
  rating — goals expected scored and conceded against an average team on neutral ground,
  which is exactly what SPI was built from.
- **Shots, not just goals.** The model is fitted twice, once on goals and once on the goals
  a club's shot profile implies, then blended. Over a handful of matches goals are mostly
  finishing luck; shot volume repeats.
- **Promoted clubs.** Coventry, Ipswich and Hull have no Premier League record, so the model
  is fitted across the Premier League *and* the Championship together, stitched into one
  scale by clubs that have played in both. The promotion penalty is then measured from every
  promoted club since 2013 rather than assumed.
- **A market anchor that expires** (Premier League only). A results model cannot see a new
  manager or a £90m signing, so the PL preseason forecast is anchored to a checked-in
  snapshot of bookmaker odds whose weight decays to zero over the first ten matchweeks.
  The other leagues run on results alone, and the method page says so.
- **50,000 season simulations**, redrawing club ratings from their uncertainty between
  scenarios — without that, the forecast claims far more confidence than it has earned.
- **A walk-forward backtest** against five baselines, published on the site — including the
  **closing bookmaker odds** and **ClubElo**, the two that are actually hard to beat. Every
  prediction is made using only matches played before it, and each external baseline is
  scored against the model over exactly the matches it covers.
- **A global club ranking.** One pooled Dixon–Coles fit over ~66,000 matches and ~836 clubs —
  fifteen seasons of UEFA competition, the big five, every non-big-five top flight with a
  feed, and the Championship — puts Bodø/Glimt and Brentford on one scale. Published on its
  own page and deliberately kept out of the league forecasts, which keep their own.
- **Match importance.** For every remaining fixture, how far a home win versus an away win
  moves each club's title, top-five and relegation chances — counted inside the same
  simulation, so it costs almost nothing and answers the only question that matters about a
  fixture: does it matter?
- **Exact-score distributions.** Click any match for the full grid of plausible scorelines.
  The most likely single score is usually only a 10-15% shot, which is the honest shape of a
  football match.
- **Strength of schedule.** Average opponent rating for what is left, adjusted for venue,
  plus a shaded run of the next six fixtures per club.
- **The forecast's own history.** A daily snapshot archive, charted, so "the model liked them
  in August" stays checkable in April.
- **In-season scoring that cannot cheat.** Each match's probabilities are frozen before
  kick-off and never overwritten, so the running log-loss is measured against what the model
  actually said beforehand — and the same frozen numbers give every club an **expected
  points** total, so "lucky" and "unlucky" become measurable rather than rhetorical.
- **Rating history back to 2003-04.** One fit per season, each using only matches played
  before it, so every club has a decade-and-a-half trajectory rather than a shape.
- **Half time.** The results feed has always carried a half-time score and this project never
  read it. A second fit on those goals gives interval probabilities per match and, per club,
  how often a lead survives and a deficit is rescued.
- **What they need.** For every club, how often each outcome happened in the simulated seasons
  where it finished on a given points total.
- **Things that leave the site.** A share card per competition and per club (so a shared link
  is no longer a blank grey box), an iCalendar feed per competition and per club, a JSON Feed
  and RSS of forecast movement, an embeddable widget, and an offline-capable PWA.

## Running it

```bash
pip install numpy scipy pytest Pillow
python -m pytest tests/ -q       # parser, name mapping, simulation invariants
python -m model.run              # all eight leagues -> site/data/<league>/*.json
python -m model.run --league la-liga   # just one
python -m model.run --league champions-league --replay 2025-26   # cup staging data
cd site && python -m http.server # then open http://localhost:8000
```

`Pillow` only draws the share cards; without it the build prints a line and skips them.
`python -m tools.extract_baselines` refreshes the committed odds/ClubElo backtest baselines
and is run by hand, not by CI.

`SKIP_BACKTEST=1` skips the walk-forward evaluation, which is the slow part.

## Layout

```
model/       fetch, parse, ratings, priors, simulate, insight, backtest, run
             europe, knockout   — the pooled corpus and the Champions League
             rankings           — the cross-league rating and head-to-head
             gamestate          — half time, discipline, referees
             feeds, social      — calendars, change feed, share cards
data/        team_meta.json (323 clubs: aliases + colours), market_priors/ (odds snapshots;
             Premier League only — other leagues run without a market anchor),
             baselines/ (frozen closing odds + ClubElo, for the backtest),
             europe/ (Champions League participants and the committed fixture file)
tools/       extract_baselines.py — one-off, run by hand
site/        the static site; site/data/*.json is generated, as are site/cal/*.ics,
             site/og/*.png, site/feed.json and site/feed.xml
             (press `/` anywhere on the site to jump to a club or matchweek)
tests/       parser, club-name mapping, simulation and leverage invariants
```

## Data sources

| What | Where |
|---|---|
| Match results, shots, cards, half-time, referee (big five only) | [datasets/football-datasets](https://github.com/datasets/football-datasets) — a mirror of football-data.co.uk. Note it does not create a season's file until months in: it added 2025-26 on 2026-02-17. |
| Fixtures, second tiers, and every competition outside the big five | [openfootball](https://github.com/openfootball) — england, espana, italy, deutschland, europe, champions-league |
| Preseason odds | Hand-captured snapshot in `data/market_priors/`, with sources and date |
| Backtest baselines (closing odds + ClubElo) | [xgabora/Club-Football-Match-Data-2000-2025](https://github.com/xgabora/Club-Football-Match-Data-2000-2025), extracted once into `data/baselines/` — frozen on purpose |

## Known limits

No injuries, suspensions or lineups. No true expected goals — free feeds have no shot
locations, so shot quality is approximated by whether a shot was on target. No fixture
congestion or European commitments. Promoted-club estimates rest on 39 historical cases.
No live or in-play anything: the build runs every six hours. Referee names are published as
a record, never as a model input — no reachable source names the official before kick-off,
and La Liga and Ligue 1 do not carry the column at all. The Eredivisie, Primeira Liga and
Championship run on goals alone from a single feed with no fallback.

Not affiliated with the Premier League. Not betting advice.
