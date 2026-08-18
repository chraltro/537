# Premier League 2026/27 Forecast

A rebuild of what FiveThirtyEight's Soccer Power Index did for club football: a rating for
every club, a probability for all 380 matches, and a simulated final table — for the
2026/27 Premier League season.

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
- **A market anchor that expires.** A results model cannot see a new manager or a £90m
  signing, so the preseason forecast is anchored to a checked-in snapshot of bookmaker odds.
  Its weight decays to zero over the first ten matchweeks.
- **50,000 season simulations**, redrawing club ratings from their uncertainty between
  scenarios — without that, the forecast claims far more confidence than it has earned.
- **A walk-forward backtest** against three baselines, published on the site. Every
  prediction is made using only matches played before it.

## Running it

```bash
pip install numpy scipy pytest
python -m pytest tests/ -q       # parser, name mapping, simulation invariants
python -m model.run              # writes site/data/*.json
cd site && python -m http.server # then open http://localhost:8000
```

`SKIP_BACKTEST=1` skips the walk-forward evaluation, which is the slow part.

## Layout

```
model/       fetch, parse, ratings, priors, simulate, backtest, run
data/        team_meta.json (club aliases + colours), market_priors.json (odds snapshot)
site/        the static site; site/data/*.json is generated
tests/       parser, club-name mapping and simulation invariants
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
