# Big Five 2026/27 Forecast

A rebuild of what FiveThirtyEight's Soccer Power Index did for club football: a rating for
every club, a probability for every match, and a simulated final table — for the 2026/27
Premier League, La Liga, Serie A, Bundesliga and Ligue 1 seasons. One model, five leagues,
one site with a league switcher.

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
- **A walk-forward backtest** against three baselines, published on the site. Every
  prediction is made using only matches played before it.
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
  actually said beforehand.

## Running it

```bash
pip install numpy scipy pytest
python -m pytest tests/ -q       # parser, name mapping, simulation invariants
python -m model.run              # all five leagues -> site/data/<league>/*.json
python -m model.run --league la-liga   # just one
cd site && python -m http.server # then open http://localhost:8000
```

`SKIP_BACKTEST=1` skips the walk-forward evaluation, which is the slow part.

## Layout

```
model/       fetch, parse, ratings, priors, simulate, insight, backtest, run
data/        team_meta.json (220 clubs: aliases + colours), market_priors/ (odds snapshots;
             Premier League only — other leagues run without a market anchor)
site/        the static site; site/data/*.json is generated
             (press `/` anywhere on the site to jump to a club or matchweek)
tests/       parser, club-name mapping, simulation and leverage invariants
```

## Data sources

| What | Where |
|---|---|
| Match results, shots, cards | [datasets/football-datasets](https://github.com/datasets/football-datasets) — a daily mirror of football-data.co.uk |
| Fixtures, Championship results | [openfootball/england](https://github.com/openfootball/england) |
| Preseason odds | Hand-captured snapshot in `data/market_priors.json`, with sources and date |

## Known limits

No injuries, suspensions or lineups. No true expected goals — free feeds have no shot
locations, so shot quality is approximated by whether a shot was on target. No fixture
congestion or European commitments. Promoted-club estimates rest on 39 historical cases.

Not affiliated with the Premier League. Not betting advice.
