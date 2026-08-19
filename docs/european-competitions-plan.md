# European club competitions — research findings and implementation plan

Owner: session lead. Build agents implement from this; nobody edits it but the lead.
All data claims below were probed on 2026-08-19 from this machine. All format claims are cited.

---

## Executive summary

**Yes for the Champions League, with high confidence on the model and low confidence on the
fixture feed. No for the Europa and Conference Leagues this season.**

**Modelling is the easy half.** A single Dixon-Coles fit over the pooled corpus — our five
domestic leagues plus **4,174 European matches parsed from `openfootball/champions-league`,
2011-12 through 2025-26** — is a connected graph: the European matches are the edges tying the
league sub-graphs together, and 272 clubs appear in them with a median of 7 matches each over
the last two seasons alone. No UEFA coefficient, no market data and no new maths is needed to
put a Slavia Praha rating on the same scale as an Arsenal rating. Two engine changes carry it:
a per-competition home term (measured below at **log-HA 0.284 in Europe vs 0.192 in the
Premier League**) and ridge shrinkage toward a *league* mean rather than toward zero.

**Data is the hard half, and it is a supply problem, not a parsing problem.** `openfootball` is
the only maintained GitHub source for European fixtures — a search across 141 UEFA-data repos
found no alternative — and its publication of a new season's league-phase file is wildly
unreliable. Measured lag from draw to first commit: **2024-25 CL +3 days; 2025-26 CL +68 days
(after matchday 4); 2024-25 EL and UECL +208 days; 2025-26 EL and UECL never appeared and are
still absent today.** We cannot build a product on a feed that behaves like that.

**The critical path is therefore not code, it is a 60-line fixture file committed into our own
repo.** The 2026-27 UCL draw is **Thursday 27 August 2026, 18:00 CET, Nyon** — eight days from
now. Ship everything that does not need the draw before 27 August; on 28 August the lead
transcribes the 144-fixture pairing list into `data/europe/` as the *primary* source with
openfootball as an opportunistic override; the forecast goes live the same day. UEL and UECL
are out of scope until openfootball shows it will carry them — a manual 36-team transcription
times three is not maintainable.

Confidence: **high** that a UCL league-phase forecast ships by ~1 September 2026; **medium**
that it stays fresh in-season without manual intervention (Risk 1); **low** that UEL/UECL are
feasible at all this cycle.

---

## 1. DATA — what actually exists

### 1.1 Repository identity (probed)

Four repo names, two distinct repositories — `diff -rq --exclude=.git` returns empty for both
pairs. `openfootball/champions-league` ≡ `openfootball/europe-champions-league` (the European
competitions repo); `openfootball/europe` ≡ `openfootball/france` (the 50-country domestic
collection, 204 `.txt` files). The `/tmp/of-france` lead was correct but mis-named: it is the
whole `europe` collection. `openfootball/uefa`, `/at`, `/be`, `/scotland` do not exist — clone
returns `could not read Username for 'https://github.com'`, which is what a 404 looks like
here; easy to misread as an auth problem.

### 1.2 (a) 2026-27 UCL/UEL/UECL participants and fixtures — **DOES NOT EXIST YET**

`git ls-tree --name-only HEAD` on `openfootball/champions-league` lists seasons
`2011-12 … 2025-26`. **There is no `2026-27` directory.** There is no GitHub source anywhere
for the 2026-27 pairings, because they do not exist yet — the draw has not happened (§2).

Publication lag, measured with `git log --diff-filter=A` on the full-history clone:

| File | Draw date | First commit | Lag |
|---|---|---|---|
| `2024-25/cl.txt` | Thu 29 Aug 2024 | 2024-09-01 | **+3 days** |
| `2025-26/cl.txt` | Thu 28 Aug 2025 | 2025-11-04 | **+68 days** |
| `2024-25/el.txt`, `2024-25/conf.txt` | Aug 2024 | 2025-03-25 | **+208 days** |
| `2025-26/el.txt`, `2025-26/conf.txt` | Aug 2025 | — | **never; still absent** |

`2025-26/` contains only `cl.txt clq.txt elq.txt confq.txt`. The Europa and Conference League
league phases of the season that finished three months ago were never committed.

### 1.3 (b) Historical European results for backtesting — **STRONG**

15 seasons. Parsed with our own `model.parse.parse_openfootball`; counts are *played* matches
it actually returned, not line counts:

| Competition | Seasons present | Parsed |
|---|---|---|
| Champions League `cl.txt` | 2011-12 … 2025-26 (15) | 1,985 |
| Europa League `el.txt` | 2020-21 … 2024-25 (5) | 804 |
| Conference League `conf.txt` | 2021-22 … 2024-25 (4) | 566 |
| Qualifying `clq`/`elq`/`confq` | 2024-25, 2025-26 | 176 / 152 / 491 |
| **Total** | | **4,174** |

Two complete Swiss-model seasons exist (2024-25 and 2025-26, 189 matches each; the 32-team
group seasons are 125). **2025-26 `cl.txt` is complete through the final** — PSG beat Arsenal
4-3 on penalties on 30 May 2026. So European results are *fresher* than most domestic
non-big-five feeds (§1.5). That inverts the usual assumption and drives the modelling
recommendation in §3.

File format (`2025-26/cl.txt`):
```
▪ League, Matchday 1
  Tue Sep 16 2025
    18:45  Athletic Club (ESP)     v Arsenal FC (ENG)         0-2 (0-0)
▪ Finals, Final
  Sat May 30
    18:00  Paris Saint-Germain FC (FRA) v Arsenal FC (ENG)  4-3 pen. 1-1 a.e.t. (1-1, 0-1)
```
Stage headers are `▪ League, Matchday N` / `▪ Playoffs, Matchday N` /
`▪ Finals, {Round of 16|Quarterfinals|Semifinals|Final}` in Swiss seasons, and
`▪ Group A` / `▪ League phase` / `▪ Playoffs` / `▪ Round of 16` … in older or non-CL files.
Club names carry a three-letter country suffix in parentheses.

### 1.4 Parser probe — **4 concrete defects, all in `model/parse.py`**

Running the *current* `parse_openfootball` against `2025-26/cl.txt`:

```
parsed matches: 176 of 189 expected      unique teams: 35 of 36      matchday: None for all 176
resolve("Arsenal FC (ENG)") -> arsenal-eng        resolve("Arsenal") -> arsenal
```

1. **Digit guard drops clubs.** `if any(c.isdigit() for c in home + away): continue` silently
   discards every line containing `Bayer 04 Leverkusen` — 12 lines here, and the reason only
   35 teams appear. Also affected in the corpus: `Bologna FC 1909`, `FC Basel 1893`,
   `FC Heidenheim 1846`, `FC Schalke 04`, `SK Dnipro-1`, `Stade Brestois 29`. Move the guard
   *after* the score is stripped, or match only a standalone numeric token.
2. **Compound `a.e.t.` / `pen.` results unmatched.** `_TRAIL_SCORE` handles `2-1 a.e.t. (1-0)`
   but not `4-3 pen. 1-1 a.e.t. (1-1, 0-1)` — 3 lines here. The 90-minute score is the *second*
   pair; ingest that one, not the shootout.
3. **Matchday never captured.** `_MD_RE` requires `Matchday` immediately after `▪`; the file
   says `▪ League, Matchday 1`. All 176 rows get `matchday=None`.
4. **Country suffix forks club identity.** `arsenal-eng` ≠ `arsenal`. Every European club must
   resolve to the same id as its domestic feed or the §3 bridge silently disconnects. Strip a
   trailing `(XXX)` before `resolve()` **and keep the code** — it is the association, which the
   §3.5 draw constraints need anyway.

Agent A is mid-edit on `parse.py` (accent folding, `import unicodedata`) — that fixes
`bayern-m-nchen` / `club-atl-tico-de-madrid` but none of the four above.

### 1.5 The rating base for non-big-five participants — **thin AND stale**

Country codes were extracted from the 2024-25 CL/EL/UECL and 2025-26 CL files and mapped to
`openfootball/europe` directories. Coverage, and — measured by parsing the newest top-flight
file in each — the date of the **last actual result available**:

| Countries | Seasons in repo | Last result on disk | Staleness |
|---|---|---|---|
| ENG ESP ITA GER FRA | big-five repos | current | — (in pipeline) |
| NED POR | 2018-19 … 2026-27 (9) | 2026-27 fixtures present | fine |
| BEL | separate repo `openfootball/belgium`, **2026-27 present** | fine | fine |
| AUT | separate repo `openfootball/austria`, → 2025-26 | 2025-26 | ~3 months |
| SCO GRE TUR | 6 seasons, newest 2025-26 | **2025-11-03** | **~9 mo, ⅓ season only** |
| NOR SWE FIN ISL IRL LVA LTU EST BLR | 2023–2025 | 2025 season, ~20% played | **~15 months** |
| CZE SUI HUN | 4-6 seasons, newest 2024-25 | **2025-05-25** | **~15 months** |
| DEN POL CRO SRB UKR CYP SVN ROU BUL SVK BIH AZE ARM MDA MKD ALB KOS MNE WAL NIR FRO LUX MLT GIB AND SMR | 2 seasons, newest 2024-25 | **2025-05-25** | **~15 months** |
| ISR KAZ | **no source at all** | — | — |

(`MCO` resolves to Ligue 1 — Monaco is covered by the France feed, not missing.) Per-country
last-commit dates confirm upstream neglect rather than a stale clone: the whole non-big-five
set was last touched `2026-05-28`, `netherlands`/`portugal` on `2026-07-08`.

**Conclusion: domestic data cannot be the rating base for non-big-five clubs.** For a Croatian
or Cypriot participant it ends 15 months before kickoff and misses a whole transfer cycle.
The European corpus, which *is* current to May 2026, has to carry them (§3).

`datasets/football-datasets` was cloned and `ls datasets` returned: `bundesliga la-liga ligue-1
premier-league serie-a worldcup` plus 11 editorial article directories. **No non-big-five
leagues. It is useless here.** Also probed and rejected: `jokecamp/FootballData`
(`UEFA_CHAMPIONS_LEAGUE/` contains one HTML bracket), `footballcsv/cache.soccerdata`
(England-centric, stops 2018-19), `engsoccerdata` (404). `openfootball/clubs` **is** useful —
56 European country directories of club metadata for the alias work in §5.

### 1.6 (c) Current-season results as they happen

Only `openfootball/champions-league`, and only when it feels like it. 18 commits in 2026, at
irregular multi-week intervals (`auto-update week 8, 9, 10, 11, 12, 13, 16, 19, 20, 22, 27`),
most recent `2026-07-02`. Against a 6-hourly build this is a feed that can be **weeks** behind.
Mitigation is in §5 Risk 1; there is no second source to fall back to.

---

## 2. FORMAT — 2026-27, verified

**Champions League league phase.** 36 clubs in one table; each plays **8 matches against 8
different opponents, 4 home and 4 away** — 144 fixtures. Four pots of nine ordered by **UEFA
club coefficient**, title holder (PSG) placed in pot 1; each club draws **two opponents from
each pot, one home and one away in each pair**. Constraints: **same association cannot meet**,
and **no club faces more than two clubs from any one other association**.
[SI, Aug 2026](https://www.si.com/soccer/when-2026-27-champions-league-draw-date-how-it-works-league-phase);
[ESPN](https://www.espn.com/soccer/story/_/id/41040476/champions-league-draw-how-new-format-works);
[BBC](https://feeds.bbci.co.uk/sport/football/articles/ce35d5n5wzdo);
[kassiesa seeding](https://kassiesa.net/uefa/seedcl2026.html).

**Draw date — the gate on this whole project.** The 2026-27 league-phase draw is **Thursday
27 August 2026, 18:00 CET, UEFA HQ Nyon**, covering 29 automatic qualifiers plus 7 play-off
winners (play-off second legs 25–26 Aug).
[BeSoccer, Jul 2026](https://www.besoccer.com/new/uefa-confirms-date-for-the-202627-champions-league-league-phase-draw-1421299);
[Vanguard, Jul 2026](https://www.vanguardngr.com/2026/07/2026-27-champions-league-full-list-of-teams-qualified-date-of-draw-venue/).
**Pairings are therefore unavailable to every build cycle before 27 Aug and available to every
one after it** — provided we source them ourselves (§1.2).

**Knockout.** **1–8 straight to the round of 16**; **9–24 into a two-legged knockout play-off**
seeded strictly (9v24, 10v23, …), higher seed home in the second leg; **25–36 eliminated**.
R16 pairs the eight direct qualifiers with the eight play-off winners, higher seed home second.
Same-association protection ends with the league phase.
[Sky Sports](https://www.skysports.com/football/news/11095/13428134/champions-league-2025-26-how-does-league-phase-and-knockout-qualification-work);
[UEFA](https://www.uefa.com/uefachampionsleague/news/02a0-1f5779647b95-29ad8ef754a8-1000--champions-league-round-of-16-and-knockout-phase-play-off/).
**Away goals: abolished** across all UEFA club competitions from 2021-22; a level aggregate
goes to **30 minutes extra time, then penalties**.
[UEFA, Jun 2021](https://www.uefa.com/uefachampionsleague/news/026a-1298aeb73a7a-5b64cb68d920-1000--abolition-of-away-goals-rule-in-all-uefa-club-competitions).

**Calendar.** MD1 8–10 Sep 2026; MD2 13–14 Oct; MD3 20–21 Oct; MD4 3–4 Nov; MD5 24–25 Nov;
MD6 8–9 Dec; MD7 19–20 Jan 2027; MD8 27 Jan (simultaneous). Knockout play-off 16–17 and
23–24 Feb 2027. Final **5 June 2027, Estadio Metropolitano, Madrid**.
[Groundhopper](https://groundhopperguides.com/2026-27-champions-league-schedule/);
[UEFA hub](https://www.uefa.com/uefachampionsleague/news/02a6-20d57cfcd03e-407c22a7f465-1000--2026-27-champions-league-teams-dates-draws-format-final/).

**UEL and UECL.** Same skeleton — 36 clubs, single table, top 8 direct, 9–24 play-off, 25–36
out — but **UEL plays 8 league-phase matches, UECL 6**. Both draws **Friday 28 August 2026,
13:00 CET, Nyon**.
[UEFA EL draws](https://www.uefa.com/uefaeuropaleague/draws/);
[UEFA UECL hub](https://www.uefa.com/uefaconferenceleague/news/02a6-20d57d15f093-a90cf54c928f-1000--2026-27-conference-league-teams-dates-draws-format-final/).
Corroborated on disk: `2024-25/el.txt` = 189 matches (36×8/2 = 144 league + knockouts),
`2024-25/conf.txt` = 153 (36×6/2 = 108 + knockouts).

**Extra time frequency** (measured, `grep -c`): 2–7 a.e.t. and 2 shootouts per league-phase
season; far commoner in two-legged qualifying (`2025-26/confq.txt`: 26 a.e.t., 6 shootouts in
256). Extra time matters for the trophy simulation, not the league phase.

---

## 3. MODELLING

### 3.1 Recommendation: one pooled fit, bridged by European matches, hierarchical ridge

**Fit a single Dixon-Coles model over the union of all matches** — five domestic leagues,
every reachable non-big-five domestic league, and all 4,174 European matches — with club
attack/defence free, and let the European matches identify the league offsets. Option (a) in
the brief, right for three reasons.

*Identified.* The 272 clubs in European matches over 2024-25 and 2025-26 alone have a median
of 7 such matches each (p25 = 4, p75 = 12, max = 35); 170 have ≥ 6. Adding 13 prior CL seasons
and 5 EL / 4 UECL seasons makes the cross-league graph densely connected. League strength is
not a parameter we invent — it falls out as the mean club rating within each league.

*Shape-preserving.* `ratings.fit` already takes an arbitrary match list and team pool;
`Fit.lambdas`, `score_matrix` and `simulate_season` need no rewrite.

*No unreachable dependency.* Option (b), UEFA coefficients as the anchor, needs a feed that is
not on GitHub — the canonical source (kassiesa.net) is not a repo and the build machine cannot
reach it — so it would be hand-committed and hand-updated annually, and it is a
**backward-looking five-year sum that already lags what the match data says**. Use coefficients
for pot display only, plus a weak fallback prior for any participant with < 3 European matches
*and* stale domestic data. Not as the calibration mechanism.

**Three engine changes are required:**

1. **Per-competition home advantage.** Measured over the whole European corpus
   (n = 4,174, schedule balanced home/away by construction): home 1.683 goals/match,
   away 1.267, **log-HA = 0.2837**. Premier League 2022-23…2025-26 (n = 1,520): home 1.618,
   away 1.336, **log-HA = 0.1921**. European home advantage is ~48% larger and this is far too
   big to absorb. `_fit_core` must carry a `home` coefficient **per competition group**
   (one per domestic league + one for Europe) rather than a single scalar, i.e. replace
   `x[2n+1]` with a small vector indexed by a per-match group id. `Fit.lambdas` gains a
   `group=` argument defaulting to the club's domestic league.
2. **Ridge toward the league mean, not toward zero.** Today `ridge * sum(att**2)` pulls every
   club to the *global* average. In a pooled fit that would drag Bodø/Glimt up and Real Madrid
   down. Parameterise `att_club = league_mean_att[L(club)] + dev_club` and ridge only `dev`.
   The league means are free parameters identified by the European edges. This is the single
   most important change and the one most likely to be got subtly wrong — gate it (§5).
3. **Extra-time and penalties.** For knockout ties, simulate 90 minutes from the existing
   score matrix; if aggregate is level, simulate 30 minutes of extra time as an independent
   draw with **λ scaled by 30/90 × 0.85** (extra time is measurably lower-scoring than open
   play; the 0.85 is a judgement call, flag it in the method page), then a coin-flip-ish
   shootout at **P(home) = 0.5** — there is no credible shootout skill signal in 24 observed
   shootouts. Away goals do **not** apply (§2).

### 3.2 What FiveThirtyEight's SPI actually did

Worth being precise, because the honest answer changes our recommendation. SPI's club version
ran on a **single global pool of matches** (ESPN's database plus the `engsoccerdata` GitHub
repo, back to 1888, ~550k matches) with per-club offensive and defensive expected-goals
ratings — architecturally very close to what we run. Cross-league comparability came from that
shared pool: continental matches and clubs moving between divisions are the links, exactly the
mechanism in §3.1.

But SPI did **not** rely on match results alone where cross-league history was thin. Roughly
**one third of the SPI rating came from a market-valuation-implied rating derived from
Transfermarkt squad values** — that is what let it rate a promoted or newly-continental club
before it had played anyone comparable. We cannot copy that piece: Transfermarkt is not a
GitHub repo and is unreachable at build time.

Sourcing caveat: `fivethirtyeight.com` is **egress-blocked from this machine**, so this rests
on the [soccer-spi README](https://github.com/fivethirtyeight/data/blob/master/soccer-spi/README.md)
(fetched successfully), [dadmetrics](https://dadmetrics.com/2019/06/14/a-bit-of-a-deeper-understanding-of-the-538-soccer-predictions-numbers/)
and [Yazman](https://joshyazman.github.io/spi-ratings-analysis/). Treat "one third market
value" as approximately right, not exact, and say so on the method page.

**What we take:** the match-pool bridge. For the market anchor we already have the mechanism
(`priors.load_market`, `MARKET_WEIGHT`, decaying over 10 matchweeks) — the right home for a
hand-curated preseason anchor on the ~10 participants with the weakest match evidence. Reuse
it; do not build a second one.

### 3.3 Clubs with no European history

A first-time European entrant gets its rating from its domestic league, which the pooled fit
has already placed on the global scale via its *league-mates'* European results — a first-time
Norwegian entrant inherits Bodø/Glimt's league offset. Two guards: (i) reuse `PROMOTED_SHRINK`
in the new form, shrinking the club's *deviation* toward its league mean in proportion to how
little recent data it has; (ii) where the domestic feed is 15 months stale (§1.5), inflate
`RATING_SD` for that club — a stale rating is not wrong, it is *uncertain*, and the scenario
resampling in `simulate_season` is exactly where that belongs. Propose
`rating_sd × (1 + months_stale / 12)`, capped at 2×.

### 3.4 Backtesting

The walk-forward harness in `model/backtest.py` extends directly: score the model on the
1,985 CL matches (and 1,370 EL/UECL) it has never seen, refitting before each matchday.
The two Swiss seasons (2024-25, 2025-26 — 378 matches) are the honest holdout for the format
we are actually forecasting. **Gate: pooled-fit log-loss on those 378 must beat a
league-average baseline and must not degrade Premier League log-loss by more than 0.002.**

### 3.5 Simulating the draw

**After the draw (from 28 Aug 2026):** trivial — the 144 pairings are fixed, and
`simulate_season` runs on them as it does on a 380-match league season. The league phase is
literally a 36-team, 8-match league table. Tiebreakers are points, then goal difference, then
goals scored — the existing `key` expression in `simulate_season` works unmodified.

**Before the draw:** sample a valid pairing set per scenario under the four constraints (two
opponents per pot, one home one away in each pair, no same-association pairing, ≤ 2 from any
one other association). Rejection sampling from a random matching is the honest cheap
implementation; UEFA runs a constrained *sequential* draw, so our marginal pairing
probabilities will differ slightly — say so. **Only worth building if it ships before
27 August**, which it will not (§5), so pre-draw the product degrades instead (§4).

**Knockout draws** are the same problem in miniature and *are* worth building, since the R16
draw is unknown when the league phase starts: sample the play-off bracket (seeded 9v24,
10v23, …) and the R16 pairing among the eight winners, then run the bracket.

---

## 4. PRODUCT — what ships

A sixth entry in the league switcher, `slug: champions-league`, reusing Agent B's `?lg=`
plumbing and Agent A's `site/data/<slug>/` layout so nothing new is invented. It is a league
with 36 teams and 8 matchdays; that is the whole trick.

**Manifest.** `site/data/leagues.json` gains an entry with `"ready": false` until the draw,
`n_teams: 36`, a `kind: "cup"` discriminator and `advance_direct: 8` / `advance_playoff: 16`
in place of `ucl_places` / `releg_places`. Agent B's rule that all league-dependent wording
comes from the manifest is what makes this work — **the "top 5" / "bottom 3" wording must
already be manifest-driven before this lands**; that is a dependency on Agent B's existing
phase, not a new requirement.

**JSONs** (`site/data/champions-league/`), same names, same shapes where possible:

| File | Change from the league contract |
|---|---|
| `forecast.json` | per-club: `spi`, `off`, `def`, projected points, `pos[36]`, plus `p_top8`, `p_playoff` (9-24), `p_out` (25-36), and round-by-round `p_r16`, `p_qf`, `p_sf`, `p_final`, `p_win`. `league` block gains `pot`, `country`, `advance_direct`, `advance_playoff`. |
| `matches.json` | unchanged shape; `md` is 1-8 for the league phase, then `"KPO"`, `"R16"`, `"QF"`, `"SF"`, `"F"`. Leverage carries over — swap the three events from title/UCL/relegation to **top-8 / qualification / elimination**, which is `simulate._leverage`'s `EVENTS` tuple and nothing else. |
| `sim_input.json` | unchanged; add `advance_direct`, `advance_playoff`, `n_teams: 36`. The browser "what if" page then works on the league phase for free. |
| `schedule.json`, `history.json`, `predictions.json`, `backtest.json` | unchanged shapes. |
| `season_report.json`, `recap.json` | unchanged. |

**Pages.** No new templates. `index.html` shows the 36-row league-phase table with the top-8
and 24th-place lines instead of UCL/relegation lines; `races.html` becomes "the race for the
top 8" and "the race to survive the cut". A **bracket strip** showing P(reach each round) is
the one genuinely new component — keep it to an SVG the existing `pos[]` heat-map colours drive.

**Degradation before the draw.** With `"ready": false` the switcher shows the entry disabled
with a "draw 27 August" tooltip — Agent B's existing not-yet-ready mechanism, used as-is.
Optionally, if Phase 1 lands early, publish a **participants-and-power page**: the 36 qualified
clubs with pooled-fit SPI and pot, plus P(win the trophy) averaged over sampled draws (§3.5).
That is the only thing needing the draw-sampler, so the sampler's priority is exactly as high
as our appetite for that page and no higher.

---

## 5. PLAN

### Phase boundaries

| Phase | Content | Blocked by draw? | Gate |
|---|---|---|---|
| **0 — lead, now** | Pre-wire: extend `leagues.League` with `kind`/`advance_direct`/`advance_playoff`; create `data/europe/` with `participants-2026-27.json` schema and an empty `fixtures-2026-27.txt`; add `champions-league` to the manifest as `ready:false`; write the alias list for all 36 likely participants into `data/team_meta.json` from `openfootball/clubs`. | No | Manifest loads; site shows a disabled sixth entry; no 404s. |
| **1 — parser + corpus** | Fix the four `parse.py` defects (§1.4); add `parse_openfootball_euro` returning stage/leg/aet metadata; ingest the 4,174-match corpus; add non-big-five domestic loaders. | No | **`2025-26/cl.txt` parses 189/189 with 36 teams and matchdays 1-8**; `2024-25` cl/el/conf parse 189/189/153; a test pins each count. |
| **2 — pooled ratings** | Per-competition home term; hierarchical ridge; staleness-inflated `RATING_SD`; extend `backtest.py` to European matches. | No | §3.4 log-loss gates. PL forecast must not move by more than noise — **diff `site/data/premier-league/forecast.json` before/after and justify every SPI change > 0.5**. |
| **3 — competition sim** | Two-legged ties, extra time, shootouts; bracket simulation; league-phase `simulate_season` on 36 teams; top-8 / play-off / round-by-round probabilities. | No | Replay 2024-25 and 2025-26 from the league-phase table: P(champion) for the actual winner must be sane, bracket probabilities must sum to 1 per round. |
| **4 — fixtures** | Lead transcribes the draw into `data/europe/fixtures-2026-27.txt` in openfootball format; loader prefers our file, overrides from openfootball only when it has ≥ as many played matches. | **YES — 27 Aug 2026** | 144 fixtures, 36 clubs, each 4H/4A, no same-association pair, ≤2 per other association — assert all five in `validate()`. |
| **5 — site** | Bracket strip; top-8/cut lines; `races.html` wording; sim worker on 36 teams. | Needs Phase 4 for real data; buildable against a fixture from Phase 3. | Page renders from a synthetic 2025-26 replay before the real draw exists. |

Phases 0–3 and most of 5 ship **before 27 August**. Phase 4 is a ~2-hour task on 28 August.
Target: live forecast by **1 September 2026**, ahead of MD1 on 8 September.

### File ownership — two parallel agents, no overlap

**Agent A (model)** owns `model/parse.py`, `model/leagues.py`, `model/data.py`,
`model/ratings.py`, `model/backtest.py`, `model/priors.py`, and a **new** `model/knockout.py`
(two-legged ties, extra time, shootouts, bracket simulation — new file precisely so it does not
collide with `simulate.py`, which the multi-league work is still touching).

**Agent B (site)** owns `site/*.html`, `site/assets/*`, and the bracket component. Agent B
consumes the JSONs and never reads `model/`.

**Contested files the LEAD owns and must pre-wire in Phase 0**, because both agents need them
and neither may edit them: `model/config.py`, `data/team_meta.json`, `data/europe/*`,
`site/data/leagues.json` schema, and this document. `model/simulate.py` is **shared** — Agent A
may add a `groups=` argument and swap `EVENTS`, but the multi-league agents are mid-edit there,
so any change lands as a lead-reviewed patch, not an agent free-for-all.

### Risk register

**Risk 1 — the fixture/results feed stalls. Likelihood high, impact fatal.** Measured:
EL/UECL 2025-26 never published, CL 2025-26 published 68 days late, most recent commit
2026-07-02 after multi-week gaps. Depending on openfootball for 2026-27 pairings could mean no
product until November. *Mitigation:* our committed `data/europe/fixtures-2026-27.txt` is the
**primary** source from day one; openfootball is a results *override* only, applied when it has
strictly more played matches than we do. Add a CI check that fails loudly if the openfootball
CL file has been static > 21 days during the league phase, so the lead hand-enters results
rather than the site quietly showing a frozen table. **This is the difference between shipping
and not shipping — do not let an agent treat it as optional.**

**Risk 2 — the pooled fit corrupts the Premier League forecast. Likelihood medium, impact
high.** Adding thousands of matches against much weaker opposition, plus a hierarchical
reparameterisation, can shift ratings the site already publishes; a subtly wrong ridge centre
would inflate every big-five club at once and look entirely plausible. *Mitigation:* the
Phase 2 gate is a **forecast diff, not just a log-loss number** — regenerate all five leagues,
compare club-by-club, explain any SPI move > 0.5. Keep the pooled fit behind a flag until it
passes. Fallback if it fails: fit Europe separately and map league offsets from European
results only — worse, but it cannot damage the existing product.

**Risk 3 — non-big-five ratings are ~15 months stale in a way users can see. Likelihood high,
impact medium.** Twenty-nine participating countries have no domestic data after May 2025. A
club since rebuilt or newly rich is rated on a squad that no longer exists, and it shows when
they lose 6-1. *Mitigation:* (i) the European corpus, current to May 2026, dominates their
rating (§3.1); (ii) inflated `RATING_SD` widens their intervals honestly (§3.3); (iii) the
market-anchor mechanism is a documented place for the lead to hand-correct the handful of clubs
where this bites, decaying over 10 matchdays as real results arrive; (iv) the method page
states staleness per country rather than hiding it. **Do not paper over this with a coefficient
prior** — that swaps a stale signal for a lagging one and buys nothing.

### Out of scope

Europa League and Conference League: 2025-26 league phases **never published**, 2024-25 ones
208 days late. Transcribing 36×8 and 36×6 fixtures across three competitions twice a year is
not a maintainable commitment. Revisit when `openfootball/champions-league` ships a
`2026-27/el.txt` — the Phase 1–3 machinery is competition-agnostic and needs only a fixtures file.
