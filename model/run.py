"""Build every JSON file the site reads, for one league or for all five.

    python -m model.run                     # all five domestic leagues
    python -m model.run --league la-liga    # just one
    SKIP_BACKTEST=1 python -m model.run     # skip the walk-forward evaluation

    python -m model.run --league champions-league
        The live cup build. Reads data/europe/fixtures-2026-27.txt first and
        openfootball only as a results override; while that file is empty it
        reports 'awaiting draw', writes nothing and leaves the league not-ready.

    python -m model.run --league champions-league --replay 2025-26
        The same pipeline over a finished season's real league phase, forecast
        from the day before its first matchday. Staging data for the site, stamped
        `"replay"` in forecast.json so the manifest can never call it live.

    python -m model.run --pooled            # or POOLED_FIT=1
        Fit the domestic leagues on the pooled European corpus instead of on
        their own history. Off by default and deliberately so -- see the Phase 2
        gate: SPI is defined against a league-average opponent, and pooling
        redefines 'average' as the average of nine hundred clubs in fifty-two
        leagues, which moves every published number for no new evidence.

Each league gets its own directory under site/data/<slug>/, and the manifest at
site/data/leagues.json is regenerated from `model.leagues` every run so the
site's switcher can never disagree with what was actually built. The Premier
League's files are additionally copied to the legacy flat site/data/*.json paths
the current pages still read; that is a copy of finished output, not a second
computation, and it goes away when the site refactor lands.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import time

import numpy as np

from . import (backtest, clubmeta, config, europe, feeds, gamestate, insight, knockout,
               leagues, priors, rankings, ratings, scale, seo, simulate, social)
from .data import Dataset

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "site", "data")

#: Nine files that used to be copied from site/data/premier-league/ up to
#: site/data/, from when the site read one league from flat paths. Every page
#: has read `site/data/<slug>/<name>.json` since the multi-league layout landed,
#: so the copies were a second, silently-diverging Premier League that nothing
#: fetched and every build rewrote. They are deleted rather than left in place:
#: a stale duplicate of a forecast is worse than no duplicate.
RETIRED_FLAT_FILES = ("forecast.json", "matches.json", "schedule.json",
                      "history.json", "predictions.json", "season_report.json",
                      "recap.json", "sim_input.json", "backtest.json")


def spi(fit: ratings.Fit, team: str, adj: float = 0.0) -> float:
    """Expected share of points against an average team, home and away.

    This is FiveThirtyEight's definition, kept deliberately: 100 means winning
    every game against a league-average side, 0 means losing every one.
    """
    o = np.log(fit.offence(team)) + adj / 2
    d = np.log(fit.defence(team)) - adj / 2
    pts = 0.0
    for home in (True, False):
        lh = np.exp(o + (fit.home if home else 0.0))
        la = np.exp(d + (0.0 if home else fit.home))
        m = simulate.score_matrix(float(lh), float(la), fit.rho)
        w, dr, _ = simulate.outcome_probs(m)
        pts += 3 * w + dr
    return float(pts / 6.0 * 100.0)


def _rating_history(teams) -> dict[str, list]:
    """SPI at each July, so a club has a trajectory and not one number.

    Read straight off the pooled trajectory rather than fitted here. It used to
    be fitted here, on this league's own matches, which made every point mean
    "how strong in this division" and cost the chart two things: a club that
    spent a season one division down left a hole in its line, and two clubs from
    different leagues drawn on one axis said whatever their divisions' averages
    happened to say.

    The window is shorter for it. A pooled rating needs the UEFA matches that
    join one league to another, and `openfootball/champions-league` begins at
    2011-12, so the line starts there instead of in 2003-04. Every point on it
    now means the same thing as every other, which the long version could not
    say about any two of its points from different divisions.
    """
    tr = global_trajectory()
    return {t: list(tr.get(t, [])) for t in teams}


def _second_tier_rating(ds: Dataset, fit: ratings.Fit, teams: list[str],
                        league: leagues.League) -> float | None:
    """Net rating of the club likely to come up through the play-off.

    Germany's and France's play-off opponent is the third-placed second-tier
    side, which this pipeline does not project -- that division has no fixture
    list here. What it does have is the division's recent results, and the fit
    already contains those clubs, so the opponent is represented by the
    third-strongest current second-tier club's net rating. An approximation, and
    the method page says so.
    """
    recent = sorted({m.season for m in ds.second})[-1:]
    if not recent:
        return None
    pool = {m.home for m in ds.second if m.season in recent}
    nets = sorted((float(np.log(fit.offence(t)) - np.log(fit.defence(t))))
                  for t in pool if t in fit.index)
    if len(nets) < 6:
        return None
    # Third best of that division, centred on this division's own average.
    third = nets[-3]
    here = float(np.mean([np.log(fit.offence(t)) - np.log(fit.defence(t))
                          for t in teams if t in fit.index]))
    return third - here


def _recap_words(league: leagues.League) -> dict:
    """How the weekly narrative names this competition's three stakes."""
    if league.kind == "cup":
        return {"title": "chance of winning it", "ucl": "chance of a top-eight finish",
                "releg": "risk of going out", "win": "the trophy", "down": "go out"}
    if league.kind == "promotion":
        return {"title": "title chance", "ucl": "chance of automatic promotion",
                "releg": "relegation risk", "win": "the title", "down": "go down"}
    return {"title": "title chance", "ucl": "chance of qualifying for the "
            "Champions League", "releg": "relegation risk",
            "win": "the title", "down": "go down"}


def _card_words(league: leagues.League) -> dict:
    """Column headings and captions for the share card."""
    name = league.name
    if league.kind == "cup":
        return {"question": f"Who wins the {name}?", "win": "TROPHY",
                "top": "TOP 8", "down": "Out",
                "finish": "Projected phase finish", "topnote": "straight to the last 16",
                "posnote": "Chance of finishing in each league-phase place"}
    if league.kind == "promotion":
        return {"question": f"Who goes up from the {name}?", "win": "TITLE",
                "top": "AUTO", "down": "Relegated",
                "finish": "Projected finish", "topnote": "automatic promotion",
                "posnote": "Chance of finishing in each place"}
    article = "" if name.split()[-1][:1].isdigit() or len(name.split()[-1]) == 1 \
        else "the "
    return {"question": f"Who wins {article}{name}?", "win": "TITLE",
            "top": "TOP", "down": "Relegated",
            "finish": "Projected finish",
            "topnote": f"top {league.ucl_places} qualifies",
            "posnote": "Chance of finishing in each place"}


def write_calendars(league: leagues.League, ms: list[dict], meta: dict,
                    teams: list[str]) -> None:
    """One .ics for the competition and one per club, into site/cal/.

    The only push a static site has. A calendar client refetches the URL on its
    own schedule, so a subscriber's fixtures carry whatever the last build
    believed and update themselves as the season moves.
    """
    base = os.path.join(HERE, "site", "cal")

    def label(md):
        if md is None or str(md).strip() == "":
            return league.name
        return f"{'Matchday' if league.kind == 'cup' else 'Matchweek'} {md}" \
            if str(md).isdigit() else str(md)

    feeds.write(os.path.join(base, f"{league.slug}.ics"),
                feeds.calendar(ms, meta, title=f"{league.name} forecast",
                               uid_ns=f"{league.slug}.537", round_label=label))
    for t in teams:
        feeds.write(
            os.path.join(base, league.slug, f"{t}.ics"),
            feeds.calendar(ms, meta, title=f"{meta[t]['name']}, {league.name}",
                           uid_ns=f"{league.slug}.537", team=t, round_label=label))
    print(f"  → cal/{league.slug}.ics and {len(teams)} club calendars")


def write_cards(league: leagues.League, fc: dict) -> None:
    """Share cards for the competition and every club. Never fatal."""
    words = _card_words(league)
    base = os.path.join(HERE, "site", "og")
    try:
        n = int(social.save(social.league_card(fc, words),
                            os.path.join(base, f"{league.slug}.png")))
        # The card prints the club's rating, and the published rating is the
        # pooled one. The trajectory's last point is fitted at the same date as
        # the ranking, so it is that number and not one near it.
        tr = global_trajectory()
        for rank, t in enumerate(fc["teams"], 1):
            line = tr.get(t["id"]) or []
            if line:
                t = {**t, "spi": line[-1]["spi"]}
            n += int(social.save(social.club_card(fc, t, rank, words),
                                 os.path.join(base, league.slug, f"{t['id']}.png")))
        print(f"  → og/{league.slug}: {n} share card(s) redrawn")
    except social.Unavailable as exc:
        print(f"  ! share cards skipped: {exc}")
    except Exception as exc:                       # noqa: BLE001
        # A drawing bug must never cost the forecast. Loud, but not fatal.
        print(f"  ! share cards failed for {league.slug}: {exc}")


# --------------------------------------------------------------------------
# One league
# --------------------------------------------------------------------------
#: Pooled-fit switch for the five domestic leagues. Off, deliberately: the
#: pooled fit is the right rating for a competition whose clubs come from thirty
#: leagues, and a needless change to five forecasts the site already publishes.
#: See the Phase 2 gate in the build report -- turning it on moves Premier
#: League SPI by up to several points, none of it from evidence the domestic
#: fit lacked, so domestic builds stay on the path they were calibrated for.
POOLED_DOMESTIC = os.environ.get("POOLED_FIT") == "1"


#: The date every pooled number is quoted at. One constant, because the ranking
#: and the trajectory both fit at it and their answers have to be the same
#: answer: the last point of a club's line is the SPI its page prints.
RANK_REF = max(dt.date.today(), dt.date(2026, 8, 1))

_TRAJECTORY: list = []


def global_trajectory() -> dict[str, list]:
    """Every club's SPI at each July, on the European scale, computed once.

    This replaced a per-league walk-forward fit whose points meant "how strong
    in this division". That version put Sporting CP at 89 and Barcelona at 80 on
    the comparison page's shared axis, which is exactly backwards and is how the
    whole business was noticed.
    """
    if not _TRAJECTORY:
        print("Rating trajectories, on the European scale…")
        t0 = time.perf_counter()
        tr = rankings.trajectory(shared_corpus(), RANK_REF)
        print(f"  · {len(tr)} clubs, {time.perf_counter() - t0:.0f}s")
        _TRAJECTORY.append(tr)
    return _TRAJECTORY[0]


def pooled_fit_for(ds: Dataset, ref: dt.date, *, league: leagues.League,
                   quiet: bool = True) -> tuple[ratings.Fit, europe.Corpus]:
    """One Dixon-Coles fit over this league plus the whole European corpus.

    Shares the Dataset's registry, which is the only reason the bridge works:
    'Arsenal FC (ENG)' in a UEFA file and 'Arsenal' in the Premier League mirror
    have to be the same club id or the European matches connect nothing.
    """
    corpus = europe.Corpus(ds.reg).load(quiet=quiet)
    corpus.add(ds.top, league.slug)
    corpus.add(ds.second, f"{league.slug}-2")
    hist = corpus.before(ref)
    pool = sorted({m.home for m in hist} | {m.away for m in hist})
    fit = ratings.fit_pooled(hist, pool, ref, group_of=corpus.group_of,
                             club_league=corpus.club_leagues(),
                             default_group=league.slug)
    return fit, corpus


def build(league: leagues.League, *, skip_backtest: bool | None = None,
          pooled: bool | None = None) -> None:
    """Run the whole pipeline for one league and write its JSON directory."""
    if skip_backtest is None:
        skip_backtest = os.environ.get("SKIP_BACKTEST") == "1"
    if pooled is None:
        pooled = POOLED_DOMESTIC
    if league.kind == "cup":
        return build_cup(league)
    out = os.path.join(OUT, league.slug)
    os.makedirs(out, exist_ok=True)

    print(f"\n=== {league.name} ({league.country}) "
          f"— {league.n_teams} clubs, {league.n_matches} matches ===")
    ds = Dataset(league).load()
    teams = ds.teams
    meta = ds.reg.meta
    kickoff = ds.kickoff

    print("Fitting ratings…")
    shot_conv = ratings.fit_shot_conversion(ds.top)
    ref = max(dt.date.today(), kickoff)
    freshness = dict(ds.sources)
    hist = ds.before(ref)
    # Every club in this season's fixture list is in the pool even if it has
    # never played a match the corpus can see. Belgium has no second-tier feed
    # upstream, so its three promoted clubs arrive with no record at all; the
    # ridge then hands them the league-average rating, which is the honest prior
    # for a club nothing is known about, and `preseason_net` applies the
    # promoted-club correction on top. Without this they are simply absent from
    # the fit and the build dies on a KeyError.
    pool = sorted({m.home for m in hist} | {m.away for m in hist} | set(teams))
    unseen = [t for t in teams
              if not any(t in (m.home, m.away) for m in hist)]
    if unseen:
        print(f"  · {len(unseen)} club(s) with no match history at all, rated at "
              f"the division average: {', '.join(unseen)}")
    if pooled:
        print("  · pooled fit: adding the European corpus")
        fit, _ = pooled_fit_for(ds, ref, league=league, quiet=False)
    else:
        fit = ratings.fit(hist, pool, ref, shot_conv=shot_conv)

    print("Calibrating priors against history…")
    cal = priors.calibrate(ds, shot_conv)
    for key in ("continuing", "promoted", "relegated"):
        c = cal.get(key)
        if not c or (key == "relegated" and not league.above_slug):
            continue
        print(f"  · {key} slope {c['slope']:.3f} (n={c['n']}, "
              f"from {c.get('source', league.slug)}"
              + (f"; {c['reason']}" if c.get("reason") else "") + ")")
    raw_net = priors._centred_net(fit, teams)
    prev_season = sorted({m.season for m in ds.top if m.season != ds.season})[-1]
    prior_net = priors.preseason_net(ds, fit, cal, teams, prev_season)
    base_adj = {t: prior_net[t] - raw_net[t] for t in teams}

    played = sum(1 for f in ds.fixtures if f.played)
    market = priors.load_market(priors.market_path(league))
    w = priors.market_weight(played, league)
    if market and w > 0:
        print(f"Anchoring to the preseason market (weight {w:.2f})…")
        fitted = priors.fit_market_adjustment(fit, ds.fixtures, teams, market,
                                              league=league, base_adj=base_adj,
                                              verbose=True)
        adj = {t: base_adj[t] + w * (fitted[t] - base_adj[t]) for t in teams}
    else:
        if not market:
            print("No market anchor for this league (data/market_priors/"
                  f"{league.market_file} absent). Ratings alone.")
        adj = base_adj
        w = 0.0

    # How much evidence each club actually has. A promoted club with two
    # seasons behind it is not as well known as one with twenty, and the
    # simulation should say so by resampling its rating more widely.
    seen_n = {t: 0 for t in teams}
    for m in hist:
        if m.home in seen_n:
            seen_n[m.home] += 1
        if m.away in seen_n:
            seen_n[m.away] += 1
    typical = float(np.median([v for v in seen_n.values() if v] or [1]))
    rating_sd = np.array([
        config.RATING_SD * min(2.0, max(1.0, (typical / max(seen_n[t], 1)) ** 0.25))
        for t in teams])
    thin = [t for t in teams if seen_n[t] < typical * 0.25]
    if thin:
        print(f"  · widened intervals for {len(thin)} thinly-evidenced club(s)")

    print(f"Simulating the season {config.N_SIMS:,} times…")
    # A play-off at either end needs the finishing order of every simulated
    # season, not just the aggregate table: who a club would meet depends on
    # where both of them finished.
    wants_orders = bool(league.advance_playoff) or bool(league.releg_playoff_pos)
    sim = simulate.simulate_season(fit, ds.fixtures, teams, league=league,
                                   adj=adj, leverage=True, rating_sd=rating_sd,
                                   keep_orders=wants_orders)

    p_po_up = np.zeros(len(teams))
    p_po_down = np.zeros(len(teams))
    if league.kind == "promotion" and league.advance_playoff:
        print("Playing the promotion play-off…")
        p_po_up = knockout.promotion_playoff(
            fit, teams, sim["orders"], direct=league.advance_direct or 0,
            band=league.advance_playoff, adj=adj)
        print(f"  · {p_po_up.sum():.2f} of one promotion place distributed")
    if league.releg_playoff_pos:
        rating = _second_tier_rating(ds, fit, teams, league)
        if rating is not None:
            print(f"Playing the relegation play-off (opponent net {rating:+.2f})…")
            p_po_down = knockout.relegation_playoff(
                fit, teams, sim["orders"], position=league.releg_playoff_pos,
                opponent_rating=rating, adj=adj)
            print(f"  · {p_po_down.sum() * 100:.1f}% chance the play-off sends "
                  "someone down")

    # Half time is a second, cheaper model over the same corpus, and the only
    # thing the results feed carries that the forecast has never read.
    ht_fit = gamestate.half_time_fit(hist, ref)
    print("  · half-time model: "
          + ("fitted" if ht_fit else "not enough half-time data in this feed"))

    table = ds.season_table(ds.season)
    idx = {t: i for i, t in enumerate(teams)}
    history = _rating_history(teams)
    how = priors.arrivals(ds, teams, prev_season)

    pos = sim["position"]
    # A second tier's table is read against a promotion line and a play-off
    # band, exactly as a cup's is read against advancement lines. The band is
    # the positions between automatic promotion and the play-off cut.
    promo = league.kind == "promotion" and league.advance_playoff
    rows = []
    for t in teams:
        i = idx[t]
        m = meta[t]
        cur = table.get(t, {})
        a = adj.get(t, 0.0)
        if promo:
            lo = league.advance_direct or 0
            playoff = float(pos[i, lo:lo + league.advance_playoff].sum())
        else:
            playoff = None
        rows.append({
            "id": t, "name": m["name"], "short": m["short"],
            "primary": m["primary"], "secondary": m["secondary"],
            # This competition's own strength scale, not the site's. It ranks
            # the clubs in this division against each other and nothing else,
            # which is exactly what the schedule-difficulty numbers need and
            # exactly what a reader must never be shown next to a rating from
            # another league: quoted as "SPI" it made Sporting CP 89.3 and FC
            # Barcelona 80.8 on two club pages of the same site. Published SPI
            # comes from `ratings.json` and is the pooled European one.
            "lg_strength": round(spi(fit, t, a), 1),
            "off": round(fit.offence(t) * np.exp(a / 2), 2),
            "def": round(fit.defence(t) * np.exp(-a / 2), 2),
            "pts": round(float(sim["points_mean"][i]), 1),
            "pts_lo": round(float(sim["points_p10"][i])),
            "pts_hi": round(float(sim["points_p90"][i])),
            "pts_min": int(sim["points_min"][i]), "pts_max": int(sim["points_max"][i]),
            "gd": round(float(sim["gd_mean"][i])),
            "title": float(sim["title"][i]), "ucl": float(sim["ucl"][i]),
            "europa": float(sim["europa"][i]), "releg": float(sim["relegation"][i]),
            **({"p_playoff": playoff,
                # Reaching the play-off is not the same as going up through it:
                # four clubs get there and one of them is promoted.
                "p_playoff_won": round(float(p_po_up[i]), 5),
                "p_up": round(float(sim["ucl"][i]) + float(p_po_up[i]), 5)}
               if promo else {}),
            **({"p_releg_playoff": round(float(p_po_down[i]), 5),
                "p_down": round(float(sim["relegation"][i]) + float(p_po_down[i]), 5)}
               if league.releg_playoff_pos else {}),
            "pos": [round(float(x), 5) for x in sim["position"][i]],
            "played": cur.get("pld", 0), "w": cur.get("w", 0), "d": cur.get("d", 0),
            "l": cur.get("l", 0), "gf": cur.get("gf", 0), "ga": cur.get("ga", 0),
            "cur_pts": cur.get("pts", 0),
            "history": history.get(t, []),
            "promoted": how[t] == "up",
            # Which way they came. A top flight only ever sees "up"; a second
            # tier also receives clubs from the division above, and calling
            # those "promoted" was both wrong on the page and wrong in the fit.
            "arrived": how[t],
        })
    rows.sort(key=lambda r: (-r["pts"], -r["gd"]))

    # No ratings on a forecast row. They used to be written here, centred on this
    # competition's own average, and `ratings.json` wrote a second set centred on
    # Europe -- so the league table showed Arsenal's defence as 89 and the club
    # page showed 81, both correct, neither labelled. Every rating the site
    # publishes now comes from `build_ratings` and there is exactly one of each.
    # `off` and `def` stay: those are goals, they are what every probability is
    # computed from, and they are honestly this competition's own.

    print("Writing match forecasts…")
    lev_by_match = {}
    if sim.get("leverage"):
        for f, lv in zip([x for x in ds.fixtures if not x.played], sim["leverage"]):
            lev_by_match[(f.home, f.away)] = lv

    ms = []
    for f in sorted(ds.fixtures, key=lambda x: (x.matchday or 0, x.date, x.home)):
        rep = simulate.match_report(fit, f.home, f.away, adj)
        best = rep["top_scores"][0]
        lv = lev_by_match.get((f.home, f.away))
        ht = gamestate.half_time_report(ht_fit, f.home, f.away, adj)
        ms.append({
            "md": f.matchday, "date": f.date.isoformat(), "time": f.time,
            "h": f.home, "a": f.away,
            # Clean sheets were already being computed by `match_report` and
            # thrown away here; half-time is the new model. Four numbers each,
            # which is what keeps them out of the 7x7 grid's league of payload.
            "csh": round(rep["cs_home"], 3), "csa": round(rep["cs_away"], 3),
            **({"ht": [round(ht["ph"], 3), round(ht["pd"], 3), round(ht["pa"], 3)],
                "htsc": ht["sc"], "htscp": round(ht["scp"], 3)} if ht else {}),
            "ph": round(rep["home_win"], 4), "pd": round(rep["draw"], 4),
            "pa": round(rep["away_win"], 4),
            "xgh": round(rep["xg_home"], 2), "xga": round(rep["xg_away"], 2),
            "sc": [best["h"], best["a"]], "scp": round(best["p"], 4),
            "alt": [[s["h"], s["a"], round(s["p"], 4)] for s in rep["top_scores"][1:4]],
            "o25": round(rep["over25"], 3), "btts": round(rep["btts"], 3),
            # 0-6 goals covers 99.9% of the distribution and keeps the payload small
            "grid": [[round(float(v), 5) for v in row[:7]] for row in rep["grid"][:7]],
            "lev": round(lv["score"], 4) if lv else 0.0,
            "swings": lv["swings"] if lv else [],
            "played": f.played, "hg": f.hg, "ag": f.ag,
        })
    json.dump({"matches": ms}, open(os.path.join(out, "matches.json"), "w"),
              separators=(",", ":"))
    print(f"  → matches.json ({len(ms)} matches)")

    print("Deriving schedule strength, history and in-season scoring…")
    spi_by_team = {r["id"]: r["lg_strength"] for r in rows}
    sos = insight.strength_of_schedule(ds.fixtures, teams, spi_by_team, fit.home)
    for r in rows:
        s_ = sos[r["id"]]
        r["sos"] = s_["remaining"]
        r["sos_rank"] = s_.get("rank")
        r["sos_played"] = s_["played"]
        r["next"] = s_["next"]
    json.dump({"schedule": {t: {"remaining": sos[t]["remaining"],
                                "played": sos[t]["played"],
                                "rank": sos[t].get("rank"),
                                "fixtures": sos[t]["fixtures"]} for t in teams}},
              open(os.path.join(out, "schedule.json"), "w"), separators=(",", ":"))

    # Optional derived outputs; each module is owned by its feature and the
    # pipeline must keep working whether or not it exists yet.
    try:
        from . import siminput
        siminput.write_sim_input(fit, ds.fixtures, teams, adj, meta,
                                 os.path.join(out, "sim_input.json"), league=league)
        print("  → sim_input.json")
    except ImportError:
        pass
    names = {t: meta[t]["name"] for t in teams}
    try:
        from . import recap
        recap.write_recap(os.path.join(out, "recap.json"), rows, played,
                          names=names, words=_recap_words(league))
        print("  → recap.json")
    except ImportError:
        pass

    frozen = insight.freeze_predictions(ms, out)
    report = insight.season_report(ms, frozen, names)
    json.dump(report, open(os.path.join(out, "season_report.json"), "w"),
              separators=(",", ":"))
    # Points earned against points the pre-kick-off forecasts said each match
    # was worth. Computed after the freeze so it reads exactly the probabilities
    # that were on record before kick-off, never the ones written today.
    xp = insight.expected_points(ms, frozen, teams)
    for r in rows:
        r["xp"] = xp[r["id"]]["xp"]
        r["xp_diff"] = xp[r["id"]]["diff"]
        r["xp_played"] = xp[r["id"]]["played"]
    snaps = insight.append_history(rows, played, out)
    print(f"  → schedule.json, predictions.json, season_report.json "
          f"({report['n']} scored), history.json ({len(snaps)} snapshots)")

    if sim.get("curves"):
        json.dump({"generated": dt.datetime.now(dt.timezone.utc)
                                  .isoformat(timespec="seconds"),
                   "events": list(simulate.EVENTS),
                   "n_sims": int(sim["n_sims"]),
                   "min_seasons": simulate.MIN_CURVE_SEASONS,
                   "teams": sim["curves"]},
                  open(os.path.join(out, "scenarios.json"), "w"),
                  separators=(",", ":"))
        print("  → scenarios.json")

    gs = gamestate.build(ds.top, teams, ref)
    json.dump(gs, open(os.path.join(out, "gamestate.json"), "w"),
              separators=(",", ":"))
    print(f"  → gamestate.json ({len(gs['referees'])} referees, "
          f"{gs['average']['n']} matches of discipline)")

    payload = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "season": config.SEASON_LABEL,
        "league": league.public(),
        "matches_played": played,
        "matches_total": league.n_matches,
        "market_weight": round(w, 3),
        "home_advantage": round(float(np.exp(fit.home)), 3),
        "ucl_places": league.ucl_places,
        "n_sims": int(sim["n_sims"]),
        "lines": sim.get("lines"),
        "sources": freshness,
        "half_time": bool(ht_fit),
        "coverage": _coverage(ds),
        "teams": rows,
    }
    json.dump(payload, open(os.path.join(out, "forecast.json"), "w"),
              separators=(",", ":"))
    print(f"  → forecast.json ({len(rows)} teams)")
    write_calendars(league, ms, meta, teams)
    write_cards(league, payload)

    if not skip_backtest:
        print(f"Running the walk-forward backtest (from {league.backtest_from})…")
        bt = backtest.run(ds)
        bt["calibration_priors"] = cal
        bt["league"] = league.slug
        bt["table"] = backtest.table_accuracy(
            ds, [s for s in sorted({m.season for m in ds.top})
                 if s >= league.backtest_from and s != ds.season], shot_conv)
        if bt["table"]:
            ta = bt["table"]
            print(f"  · table: champion called {ta['champion_rate']:.0%} of "
                  f"{ta['n']} seasons, rank correlation {ta['rank_corr']:.2f}, "
                  f"mean points error {ta['mean_points_error']:.1f}")
        json.dump(bt, open(os.path.join(out, "backtest.json"), "w"), indent=1)
        m = bt["model"]
        print(f"  → backtest.json  log-loss {m['log_loss']:.4f} "
              f"rps {m['rps']:.4f} acc {m['accuracy'] * 100:.1f}% over {m['n']} matches")
    validate(league)


# --------------------------------------------------------------------------
# European competitions
# --------------------------------------------------------------------------
#: Stage label -> the `md` value the site contract uses for it.
KNOCKOUT_MD = {"playoff": "KPO", "r16": "R16", "qf": "QF", "sf": "SF",
               "final": "F"}


def _recentre(fit: ratings.Fit, teams: list[str]) -> ratings.Fit:
    """Re-express a pooled fit relative to the average club in `teams`.

    SPI means 'expected share of points against an average team', and in a
    pooled fit the average team is the average of nine hundred clubs across
    every league with a feed -- which would put every Champions League participant
    above 95 and say nothing. Shifting attack and defence onto the competition's
    own average, with the intercept absorbing the shift, leaves every match
    probability identical and makes the number mean what it says.
    """
    idx = [fit.index[t] for t in teams]
    a_bar = float(np.mean(fit.att[idx]))
    d_bar = float(np.mean(fit.dfn[idx]))
    out = ratings.Fit(fit.teams, fit.att - a_bar, fit.dfn - d_bar,
                      fit.mu + a_bar - d_bar, fit.home, fit.rho,
                      homes=fit.homes, default_group=fit.default_group,
                      club_league=fit.club_league)
    return out


def build_cup(league: leagues.League, *, replay: str | None = None,
              skip_backtest: bool | None = None,
              n_sims: int = config.N_SIMS) -> None:
    """Build one European competition's JSON directory.

    Two modes. Live: the pairings come from `data/europe/fixtures-{season}.txt`
    (ours, primary) with openfootball as a results override, and if that file is
    still empty the build reports 'awaiting draw' and writes nothing, leaving the
    league `ready: false`. Replay: a finished season's real league phase is
    forecast from the day before its first matchday, which is exactly the state
    the live build will be in on 1 September, and every file is stamped with the
    season it replays.
    """
    if skip_backtest is None:
        skip_backtest = os.environ.get("SKIP_BACKTEST") == "1"
    out = os.path.join(OUT, league.slug)
    season = replay or config.SEASON
    print(f"\n=== {league.name} — {league.n_teams} clubs, "
          f"{league.n_matches} league-phase matches "
          f"({'REPLAY ' + replay if replay else season}) ===")

    from .parse import TeamRegistry
    reg = TeamRegistry()
    knockouts: list = []
    if replay:
        fixtures, src = europe.load_replay_fixtures(reg, replay)
        text = europe.fetch.get(europe.euro_url(replay, "cl"), required=True)
        from .parse import parse_openfootball_euro
        knockouts = [m for m in parse_openfootball_euro(text, replay, reg, "cl")
                     if m.stage in KNOCKOUT_MD]
    else:
        try:
            fixtures, src = europe.load_cup_fixtures(reg, season)
        except europe.AwaitingDraw as exc:
            print(f"  · awaiting draw: {exc}")
            print(f"  · {league.slug} stays not-ready; nothing written.")
            return
    print(f"  · {len(fixtures)} league-phase fixtures from {src['source']} "
          f"({src['ours_played']} played in ours, "
          f"{src['openfootball_played']} upstream)")

    teams = sorted({m.home for m in fixtures} | {m.away for m in fixtures})
    validate_cup_fixtures(league, fixtures, teams, reg)

    kickoff = min(f.date for f in fixtures)
    ref = kickoff - dt.timedelta(days=1) if replay else max(dt.date.today(), kickoff)
    if replay:
        # A replay is a forecast, not a recital: everything from the season being
        # replayed is removed from the fit, and the fixtures go in unplayed.
        for f in fixtures:
            f.hg = f.ag = None
            f.played = False

    print("Fitting the pooled rating over the European corpus…")
    corpus = shared_corpus(reg, quiet=False)
    hist = corpus.before(ref)
    pool = sorted({m.home for m in hist} | {m.away for m in hist})
    missing = [t for t in teams if t not in set(pool)]
    if missing:
        raise ValueError(f"{league.slug}: no history at all for {missing}")
    fit = ratings.fit_pooled(hist, pool, ref, group_of=corpus.group_of,
                             club_league=corpus.club_leagues(),
                             default_group=europe.EUROPE)
    fit = _recentre(fit, teams)
    print(f"  · {len(hist):,} matches, {len(pool)} clubs, "
          f"European home advantage x{np.exp(fit.home_advantage(europe.EUROPE)):.3f}")

    # Staleness: a club whose freshest result is fifteen months old is not badly
    # rated, it is uncertainly rated (plan 3.3).
    seen = corpus.last_seen()
    rating_sd = ratings.staleness_sd(teams, seen, ref)
    stale = [(t, round((ref - seen[t]).days / 30.44, 1)) for t in teams
             if (ref - seen[t]).days > 200]
    if stale:
        print(f"  · widened intervals for {len(stale)} stale clubs: "
              + ", ".join(f"{t} ({m}mo)" for t, m in sorted(stale, key=lambda r: -r[1])[:6]))

    # The preseason anchor. Plan 3.2 is explicit that this mechanism already
    # exists and must not be rebuilt: it is where the lead hand-corrects the
    # handful of participants whose domestic feed is fifteen months stale, and
    # it decays to nothing over the first ten matchdays as real results arrive.
    # A cup's league phase has no title market to solve against, so what the
    # file carries here is a direct per-club net-rating nudge, not odds.
    adj = None
    market = priors.load_market(priors.market_path(league))
    played_now = sum(1 for f in fixtures if f.played)
    w = priors.market_weight(played_now, league)
    if market.get("net") and w > 0:
        adj = {t: w * float(market["net"].get(t, 0.0)) for t in teams}
        moved = {t: round(v, 3) for t, v in adj.items() if abs(v) > 1e-9}
        print(f"  · preseason anchor (weight {w:.2f}) moves {len(moved)} clubs: "
              f"{moved}")
    else:
        w = 0.0

    print(f"Simulating the league phase {n_sims:,} times…")
    sim = simulate.simulate_season(fit, fixtures, teams, league=league,
                                   adj=adj, n_sims=n_sims, rating_sd=rating_sd,
                                   leverage=True, events=simulate.CUP_EVENTS,
                                   keep_orders=True)
    print("Drawing and playing the knockout…")
    br = knockout.simulate_bracket(fit, teams, sim["orders"],
                                   group=europe.EUROPE, adj=adj)

    os.makedirs(out, exist_ok=True)
    meta = reg.meta
    parts = load_participants(season if not replay else None)
    pos = sim["position"]
    direct, playoff = league.advance_direct, league.advance_playoff
    idx = {t: i for i, t in enumerate(teams)}
    rows = []
    for t in teams:
        i = idx[t]
        m = meta[t]
        p = parts.get(t, {})
        rows.append({
            "id": t, "name": m["name"], "short": m["short"],
            "primary": m["primary"], "secondary": m["secondary"],
            # As above: this competition's own scale, for its own
            # schedule numbers. The published rating is in ratings.json.
            "lg_strength": round(spi(fit, t, (adj or {}).get(t, 0.0)), 1),
            "off": round(fit.offence(t) * np.exp((adj or {}).get(t, 0.0) / 2), 2),
            "def": round(fit.defence(t) * np.exp(-(adj or {}).get(t, 0.0) / 2), 2),
            "pot": p.get("pot"), "assoc": p.get("assoc"),
            "pts": round(float(sim["points_mean"][i]), 1),
            "pts_lo": round(float(sim["points_p10"][i])),
            "pts_hi": round(float(sim["points_p90"][i])),
            "pts_min": int(sim["points_min"][i]), "pts_max": int(sim["points_max"][i]),
            "gd": round(float(sim["gd_mean"][i])),
            "p_top8": float(pos[i, :direct].sum()),
            "p_playoff": float(pos[i, direct:direct + playoff].sum()),
            "p_out": float(pos[i, direct + playoff:].sum()),
            "p_r16": float(br["r16"][i]), "p_qf": float(br["qf"][i]),
            "p_sf": float(br["sf"][i]), "p_final": float(br["final"][i]),
            "p_win": float(br["win"][i]),
            # The three history/recap metric names are shared across every
            # competition so `insight.append_history` and `recap` need no cup
            # branch. For a cup they mean: won the trophy / finished top 8 /
            # eliminated in the league phase. The site labels them from the
            # manifest, exactly as it does the domestic ones.
            "title": float(br["win"][i]),
            "ucl": float(pos[i, :direct].sum()),
            "releg": float(pos[i, direct + playoff:].sum()),
            "pos": [round(float(x), 5) for x in pos[i]],
            "played": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "cur_pts": 0,
            # A cup club's trajectory used to be empty, because the points
            # were fitted from a league's own matches and a cup has none. They
            # come from the pooled fit now, which every one of these clubs is
            # in, so the Champions League club pages get the same chart as
            # everyone else: 35 of the 36 have a line.
            "history": _rating_history([t]).get(t, []),
            "promoted": False, "arrived": "stayed",
        })
    rows.sort(key=lambda r: (-r["pts"], -r["gd"]))

    lev_by_match = {}
    for f, lv in zip([x for x in fixtures if not x.played], sim["leverage"] or []):
        lev_by_match[(f.home, f.away)] = lv
    ms = []
    for f in sorted(fixtures, key=lambda x: (x.matchday or 0, x.date, x.home)):
        ms.append(_match_row(fit, f, f.matchday,
                             lev_by_match.get((f.home, f.away)), adj))
    for f in sorted(knockouts, key=lambda x: (x.date, x.home)):
        ms.append(_match_row(fit, f, KNOCKOUT_MD[f.stage], None, adj))
    json.dump({"matches": ms}, open(os.path.join(out, "matches.json"), "w"),
              separators=(",", ":"))
    print(f"  → matches.json ({len(ms)} rows, "
          f"{len(ms) - len(fixtures)} knockout)")

    spi_by_team = {r["id"]: r["lg_strength"] for r in rows}
    sos = insight.strength_of_schedule(fixtures, teams, spi_by_team,
                                       fit.home_advantage(europe.EUROPE))
    for r in rows:
        s_ = sos[r["id"]]
        r["sos"], r["sos_rank"] = s_["remaining"], s_.get("rank")
        r["sos_played"], r["next"] = s_["played"], s_["next"]
    json.dump({"schedule": {t: {"remaining": sos[t]["remaining"],
                                "played": sos[t]["played"],
                                "rank": sos[t].get("rank"),
                                "fixtures": sos[t]["fixtures"]} for t in teams}},
              open(os.path.join(out, "schedule.json"), "w"), separators=(",", ":"))

    from . import siminput
    siminput.write_sim_input(fit, fixtures, teams, adj, meta,
                             os.path.join(out, "sim_input.json"), league=league)
    # The European files carry a half-time score in parentheses on every line,
    # which the parser now keeps, so a cup gets the same game-state panel as a
    # league -- built from these clubs' continental record rather than from a
    # domestic season they did not all play in. No cards or referees: the UEFA
    # feed has neither.
    euro_hist = [m for m in corpus.matches
                 if m.comp in europe.EURO_COMPS and (m.home in set(teams)
                                                     or m.away in set(teams))]
    gs = gamestate.build(euro_hist, teams, ref)
    gs["note"] = ("From these clubs' matches in UEFA competition. The European feed "
                  "carries no cards, fouls or referees, so only the half-time "
                  "columns are populated.")
    json.dump(gs, open(os.path.join(out, "gamestate.json"), "w"),
              separators=(",", ":"))
    frozen = insight.freeze_predictions(ms, out)
    names = {t: meta[t]["name"] for t in teams}
    report = insight.season_report(ms, frozen, names)
    json.dump(report, open(os.path.join(out, "season_report.json"), "w"),
              separators=(",", ":"))
    xp = insight.expected_points(ms, frozen, teams)
    for r in rows:
        r["xp"] = xp[r["id"]]["xp"]
        r["xp_diff"] = xp[r["id"]]["diff"]
        r["xp_played"] = xp[r["id"]]["played"]
    snaps = insight.append_history(rows, 0, out)
    from . import recap
    recap.write_recap(os.path.join(out, "recap.json"), rows, 0,
                      names=names, words=_recap_words(league))
    if sim.get("curves"):
        json.dump({"generated": dt.datetime.now(dt.timezone.utc)
                                  .isoformat(timespec="seconds"),
                   "events": list(simulate.CUP_EVENTS),
                   "n_sims": int(sim["n_sims"]),
                   "min_seasons": simulate.MIN_CURVE_SEASONS,
                   "teams": sim["curves"]},
                  open(os.path.join(out, "scenarios.json"), "w"),
                  separators=(",", ":"))

    payload = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "season": season.replace("-", "/"),
        "league": league.public(),
        "matches_played": sum(1 for f in fixtures if f.played),
        "matches_total": league.n_matches,
        "market_weight": round(w, 3),
        "home_advantage": round(float(np.exp(fit.home_advantage(europe.EUROPE))), 3),
        "ucl_places": league.ucl_places,
        "advance_direct": direct, "advance_playoff": playoff,
        "n_sims": int(sim["n_sims"]),
        "bracket_sims": int(br["n_sims"]),
        "lines": sim.get("lines"),
        "source": src,
        "participants_status": parts.get("__status__"),
        "teams": rows,
    }
    if replay:
        payload["replay"] = replay
        payload["replay_note"] = (
            f"Staging data. The {replay} league phase, forecast from the day "
            "before its first matchday; knockout rows are that season's real "
            "ties. Not a live forecast: the 2026-27 draw is 27 August 2026.")
    json.dump(payload, open(os.path.join(out, "forecast.json"), "w"),
              separators=(",", ":"))
    print(f"  → forecast.json ({len(rows)} clubs), schedule.json, sim_input.json, "
          f"season_report.json, history.json ({len(snaps)} snapshots)")
    write_calendars(league, ms, meta, teams)
    write_cards(league, payload)

    if not skip_backtest:
        print("Running the European walk-forward (the two Swiss seasons)…")
        bt = backtest.run_european(corpus, ["2024-25", "2025-26"], quiet=True)
        bt["league"] = league.slug
        json.dump(bt, open(os.path.join(out, "backtest.json"), "w"), indent=1)
        m, base = bt["model"], bt["baselines"]["league_average"]
        print(f"  → backtest.json  log-loss {m['log_loss']:.4f} vs "
              f"{base['log_loss']:.4f} for the league-average baseline, "
              f"rps {m['rps']:.4f} vs {base['rps']:.4f}, over {m['n']} matches")
    validate_cup(league)


def _match_row(fit, f, md, lv, adj=None) -> dict:
    rep = simulate.match_report(fit, f.home, f.away, adj)
    best = rep["top_scores"][0]
    return {
        "md": md, "date": f.date.isoformat(), "time": f.time,
        "h": f.home, "a": f.away,
        "ph": round(rep["home_win"], 4), "pd": round(rep["draw"], 4),
        "pa": round(rep["away_win"], 4),
        "xgh": round(rep["xg_home"], 2), "xga": round(rep["xg_away"], 2),
        "sc": [best["h"], best["a"]], "scp": round(best["p"], 4),
        "alt": [[s["h"], s["a"], round(s["p"], 4)] for s in rep["top_scores"][1:4]],
        "o25": round(rep["over25"], 3), "btts": round(rep["btts"], 3),
        "grid": [[round(float(v), 5) for v in row[:7]] for row in rep["grid"][:7]],
        "lev": round(lv["score"], 4) if lv else 0.0,
        "swings": lv["swings"] if lv else [],
        "played": f.played, "hg": f.hg, "ag": f.ag,
    }


def load_participants(season: str | None) -> dict:
    """The committed participant list, keyed by club id. Empty when absent."""
    if season is None:
        return {}
    path = europe.participants_path(season)
    if not os.path.exists(path):
        return {}
    doc = json.load(open(path))
    out = {c["id"]: c for c in doc.get("clubs", []) if c.get("id")}
    out["__status__"] = doc.get("status")
    return out


def validate_cup_fixtures(league, fixtures, teams, reg) -> None:
    """The five things a league-phase fixture list has to be.

    Plan Phase 4's gate, asserted here so a hand-transcribed file cannot ship
    with a club playing five home games or two clubs from one association drawn
    against each other.
    """
    if len(fixtures) != league.n_matches:
        raise ValueError(f"{league.slug}: expected {league.n_matches} "
                         f"league-phase fixtures, got {len(fixtures)}")
    if len(teams) != league.n_teams:
        raise ValueError(f"{league.slug}: expected {league.n_teams} clubs, "
                         f"got {len(teams)}")
    per = league.n_matches * 2 // league.n_teams // 2
    for t in teams:
        h = sum(1 for m in fixtures if m.home == t)
        a = sum(1 for m in fixtures if m.away == t)
        if h != per or a != per:
            raise ValueError(f"{league.slug}: {t} has {h} home / {a} away")
    auto = [t for t in teams if reg.meta.get(t, {}).get("auto")]
    if auto:
        raise ValueError(f"{league.slug}: unmapped club names: {auto}")
    same = [(m.home, m.away) for m in fixtures
            if m.home_assoc and m.home_assoc == m.away_assoc]
    if same:
        raise ValueError(f"{league.slug}: same-association pairings: {same[:4]}")
    if any(m.home_assoc for m in fixtures):
        for t in teams:
            seen: dict[str, int] = {}
            for m in fixtures:
                if m.home == t:
                    seen[m.away_assoc] = seen.get(m.away_assoc, 0) + 1
                elif m.away == t:
                    seen[m.home_assoc] = seen.get(m.home_assoc, 0) + 1
            over = {k: v for k, v in seen.items() if v > 2}
            if over:
                raise ValueError(f"{league.slug}: {t} faces {over}, max 2 per "
                                 "association")


def validate_cup(league: leagues.League) -> None:
    """Refuse to ship an incoherent cup forecast."""
    out = os.path.join(OUT, league.slug)
    fc = json.load(open(os.path.join(out, "forecast.json")))
    rows = fc["teams"]
    assert len(rows) == league.n_teams, "wrong club count"
    for t in rows:
        s = sum(t["pos"])
        assert abs(s - 1) < 1e-3, f"{t['id']} position distribution sums to {s}"
        s = t["p_top8"] + t["p_playoff"] + t["p_out"]
        assert abs(s - 1) < 1e-3, f"{t['id']} advancement splits sum to {s}"
        assert t["p_win"] <= t["p_final"] <= t["p_sf"] <= t["p_qf"] <= t["p_r16"] + 1e-9
    for key, want in (("p_top8", league.advance_direct),
                      ("p_playoff", league.advance_playoff),
                      ("p_out", league.n_teams - league.advance_direct
                       - league.advance_playoff),
                      ("p_r16", 16), ("p_qf", 8), ("p_sf", 4),
                      ("p_final", 2), ("p_win", 1)):
        s = sum(t[key] for t in rows)
        assert abs(s - want) < 0.03, f"{key} sums to {s:.3f}, expected {want}"
    print(f"Validation passed. ({league.slug})")


# --------------------------------------------------------------------------
# Integrity and manifest
# --------------------------------------------------------------------------
def validate(league: leagues.League) -> None:
    """Refuse to ship a broken forecast."""
    out = os.path.join(OUT, league.slug)
    fc = json.load(open(os.path.join(out, "forecast.json")))
    ms = json.load(open(os.path.join(out, "matches.json")))["matches"]
    assert len(fc["teams"]) == league.n_teams, "wrong team count"
    assert len(ms) == league.n_matches, "wrong match count"
    assert fc["league"]["slug"] == league.slug, "forecast is labelled as another league"
    for t in fc["teams"]:
        s = sum(t["pos"])
        assert abs(s - 1) < 1e-3, f"{t['id']} position distribution sums to {s}"
    for k, want in (("title", 1), ("ucl", league.ucl_places),
                    ("releg", league.releg_places)):
        s = sum(t[k] for t in fc["teams"])
        assert abs(s - want) < 0.02, f"{k} probabilities sum to {s}, expected {want}"
    for m in ms:
        s = m["ph"] + m["pd"] + m["pa"]
        assert abs(s - 1) < 1e-3, f"{m['h']}-{m['a']} outcome probabilities sum to {s}"
    print(f"Validation passed. ({league.slug})")


#: The pooled corpus is expensive to assemble the first time (about seventy
#: seconds cold, nothing at all against a warm `.cache`) and both the cup build
#: and the global ranking want it. Built once per process.
_CORPUS: europe.Corpus | None = None


def shared_corpus(reg=None, *, quiet: bool = True) -> europe.Corpus:
    """The pooled corpus, loaded once and complete, for everything that fits it.

    Complete matters more than once. The rating trajectory and the global
    ranking are two fits of the same model, and their answers have to be the
    same answer where they overlap: the last point of a club's line is the SPI
    its page prints. They were fitted on corpora that differed by the whole
    Championship, because that competition used to be added here after the
    league builds had already drawn their trajectories, and RB Leipzig came out
    at 52.2 on one and 52.3 on the other.
    """
    global _CORPUS
    if _CORPUS is None:
        from .parse import TeamRegistry
        _CORPUS = europe.Corpus(reg or TeamRegistry()).load(quiet=quiet)
        # A second tier this site forecasts in its own right is not in
        # `load_second_tiers` -- that would read it twice -- so it joins here,
        # through the same registry, and a club that goes up keeps one id and
        # one unbroken line.
        for lg in leagues.LEAGUES:
            if lg.kind != "promotion":
                continue
            try:
                ds = Dataset(lg).load()
            except Exception as exc:               # noqa: BLE001
                print(f"  ! {lg.slug} not added to the pooled corpus: {exc}")
                continue
            _CORPUS.add([m for m in ds.top if m.played], lg.slug)
            if not quiet:
                print(f"  · added {len(ds.top)} {lg.name} matches to the corpus")
    return _CORPUS


#: Where a projected league's files go, keyed by association. A slug and not a
#: derived slugification, because these are URLs: once one is published it is
#: someone's bookmark, and a rename made by tweaking a helper is a dead link.
PROJECTION_SLUG: dict[str, str] = {
    "NOR": "eliteserien",
    "BLR": "belarusian-premier-league",
    "LUX": "luxembourg-national-division",
    "UKR": "ukrainian-premier-league",
    "POL": "ekstraklasa",
}


def _projection_season(src) -> str:
    """The season a projected league is currently playing.

    A winter league is labelled the way `config.SEASON` is; a summer league by
    the bare year, which is the year that season started and, in August, the one
    running now.
    """
    latest = max(src.seasons) if src.seasons else config.SEASON
    if "-" in latest:
        return config.SEASON
    return config.SEASON.split("-")[0]


def build_projections(_ready: set[str]) -> list[str]:
    """A projected final table for every league whose grid armed.

    These are the leagues with no fixture list anywhere: the club list and the
    results come from a Wikipedia results grid, and the matches left over are
    the pairs that have not met, which in a plain double round-robin is the
    whole of the rest of the season. See `model/projection.py` for what that
    does and does not entitle the site to say.

    The ratings are the pooled ones -- the same fit the global ranking
    publishes -- so a projection cannot drift onto a scale of its own.
    """
    from . import projection, roundrobin, wikifootball
    live = sorted(wikifootball.PROJECTED & wikifootball.CANDIDATES)
    if not live:
        return []
    corpus = shared_corpus()
    reg = corpus.reg
    hist = corpus.before(RANK_REF)
    pool = sorted({m.home for m in hist} | {m.away for m in hist})
    fit = ratings.fit_pooled(hist, pool, RANK_REF, group_of=corpus.group_of,
                             club_league=corpus.club_leagues())
    done: list[str] = []
    for assoc in live:
        src = europe.BY_ASSOC[assoc]
        slug = PROJECTION_SLUG[assoc]
        season = _projection_season(src)
        try:
            got = wikifootball.read(assoc, reg, season)
            if got is None:
                raise roundrobin.ShapeError(f"no {season} article for {assoc}")
            names, _rows = got
            clubs = [reg.known(n) for n in names]
            if any(c is None for c in clubs):
                missing = [n for n, c in zip(names, clubs) if c is None]
                raise roundrobin.ShapeError(
                    f"{len(missing)} club(s) in the {season} grid resolve to "
                    f"nothing: {', '.join(missing)}")
            played = [m for m in corpus.matches
                      if m.comp == src.group and m.season == season and m.played]
            proj = projection.Projection(
                slug=slug, name=src.name, country=src.country or src.assoc,
                season=season, source="wikipedia", clubs=clubs, played=played)
            out = projection.run(proj, fit)
        except Exception as exc:                       # noqa: BLE001
            print(f"  ! {slug}: no projection ({exc})")
            continue
        out["generated"] = dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds")
        path = os.path.join(OUT, slug)
        os.makedirs(path, exist_ok=True)
        json.dump(out, open(os.path.join(path, "projection.json"), "w"), indent=1)
        print(f"  → {slug}/projection.json ({out['matches_played']}/"
              f"{out['matches_total']} played, {len(out['teams'])} clubs)")
        done.append(slug)
    return done


def build_rankings(ready: set[str]) -> None:
    """The cross-league rating, written to site/data/global.json.

    Deliberately a separate file on a separate scale. The five domestic
    forecasts are built by `build()` from each league's own fit and are not
    touched here — see `POOLED_DOMESTIC` for why that matters.
    """
    print("\n=== Global club rankings (pooled cross-league fit) ===")
    featured: set[str] = set()
    for slug in sorted(ready):
        path = os.path.join(OUT, slug, "forecast.json")
        try:
            featured |= {t["id"] for t in json.load(open(path))["teams"]}
        except (OSError, ValueError, KeyError):
            continue
    corpus = shared_corpus(quiet=False)
    payload = rankings.build(corpus, RANK_REF, featured=featured)
    json.dump(payload, open(os.path.join(OUT, "global.json"), "w"),
              separators=(",", ":"))
    print(f"  → global.json ({payload['n_clubs']} clubs across "
          f"{payload['n_leagues']} leagues, from {payload['n_matches']:,} matches)")
    # And the same lines as their own file, for the comparison page. It offers
    # every club in the ranking and could draw a trajectory only for the ones
    # with a forecast page, because that is where the points lived; the rest got
    # a sentence explaining that this site does not build their competition,
    # which was true and not what anybody wanted to read.
    # Only the clubs the ranking holds. The trajectory has points for a hundred
    # more -- sides that fell below the match floor or went quiet by the final
    # July -- and the comparison page cannot offer any of them, so they are rows
    # nothing can address in a file every visit to that page downloads.
    rated = {c["id"] for c in payload["clubs"]}
    tr = {k: v for k, v in global_trajectory().items() if k in rated}
    json.dump({"generated": payload["generated"],
               "note": ("SPI each July, on the pooled European scale. A line "
                        "starts when the corpus can first see that club's "
                        "league, so they are not all the same length."),
               "clubs": tr},
              open(os.path.join(OUT, "trajectory.json"), "w"),
              separators=(",", ":"))
    kb = os.path.getsize(os.path.join(OUT, "trajectory.json")) / 1024
    print(f"  → trajectory.json ({len(tr)} clubs, {kb:.0f}KB)")
    try:
        social.save(social.global_card(payload),
                    os.path.join(HERE, "site", "og", "global.png"))
        print("  → og/global.png")
    except Exception as exc:                       # noqa: BLE001
        print(f"  ! global share card skipped: {exc}")
    h2h = rankings.head_to_head(corpus, featured)
    json.dump({"generated": payload["generated"], "pairs": h2h},
              open(os.path.join(OUT, "h2h.json"), "w"), separators=(",", ":"))
    print(f"  → h2h.json ({len(h2h)} pairs among {len(featured)} featured clubs)")
    top = payload["clubs"][:5]
    for r in top:
        print(f"    {r['rank']:>3}. {r['name']:<24} {r['spi']:>5.1f}  {r['league']}")


HOME = "https://chraltro.github.io/537/"


def build_feeds(ready: set[str]) -> None:
    """The JSON Feed and RSS of what the forecast changed its mind about."""
    rows = []
    for lg in leagues.LEAGUES + leagues.EUROPEAN:
        if lg.slug not in ready:
            continue
        d = os.path.join(OUT, lg.slug)
        try:
            recap_doc = json.load(open(os.path.join(d, "recap.json")))
            fc = json.load(open(os.path.join(d, "forecast.json")))
        except (OSError, ValueError):
            continue
        rows.append({
            "slug": lg.slug, "name": lg.name, "recap": recap_doc,
            # The same vocabulary the weekly narrative uses, so a Championship
            # item says "chance of automatic promotion" rather than naming a
            # European competition the club is not in.
            "words": _recap_words(lg),
            "meta": {t["id"]: t for t in fc["teams"]},
            "url": f"{HOME}races.html"
                   + (f"?lg={lg.slug}" if lg.slug != leagues.DEFAULT.slug else ""),
        })
    items = feeds.feed_items(rows)
    title = "537: forecast movement"
    site = os.path.join(HERE, "site")
    feeds.write(os.path.join(site, "feed.json"),
                json.dumps(feeds.json_feed(items, home=HOME, title=title),
                           separators=(",", ":")))
    feeds.write(os.path.join(site, "feed.xml"),
                feeds.rss(items, home=HOME, title=title))
    print(f"  → feed.json and feed.xml ({len(items)} item(s) with real movement)")


def _coverage(ds) -> dict:
    """What this competition's forecast is actually fitted on.

    Nine competitions do not all rest on the same amount of evidence, and until
    now the site said nothing about which. The Premier League has twenty-five
    seasons from a feed that carries shots and cards; Belgium has six seasons of
    goals from a feed that skipped two of them. Both produce a table of
    probabilities that looks identical. This is the difference, published.
    """
    lg = ds.league
    seasons = sorted({m.season for m in ds.top if m.season != ds.season})
    per = {}
    for m in ds.top:
        if m.season != ds.season:
            per[m.season] = per.get(m.season, 0) + 1
    # A season the feed left half-finished is worth naming: it is weighted as
    # heavily as a complete one and carries a fraction of the evidence.
    full = lg.n_teams * (lg.n_teams - 1)
    partial = [s for s in seasons if per[s] < full * 0.75]
    return {
        "source": lg.source,
        "seasons": len(seasons),
        "first_season": seasons[0] if seasons else "",
        "last_season": seasons[-1] if seasons else "",
        "matches": sum(per.values()),
        "second_tier": len(ds.second),
        "above": len(ds.above),
        "missing_seasons": [s for s in _expected_seasons(lg, seasons)
                            if s not in per],
        "partial_seasons": partial,
        "has_shots": lg.source == "mirror",
    }


def _expected_seasons(lg, seen: list[str]) -> list[str]:
    """Every season between the first and last the feed has, inclusive.

    A gap in the middle is a hole in the feed; a short run at the start is just
    where the feed begins. Only the first is worth reporting.
    """
    if not seen:
        return []
    a, b = int(seen[0][:4]), int(seen[-1][:4])
    return [f"{y}-{str(y + 1)[2:]}" for y in range(a, b + 1)]


def build_coverage(ready: set[str]) -> None:
    """One file describing what every competition is built from."""
    rows = []
    for lg in leagues.LEAGUES:
        if lg.slug not in ready:
            continue
        try:
            fc = json.load(open(os.path.join(OUT, lg.slug, "forecast.json")))
        except (OSError, ValueError):
            continue
        cov = fc.get("coverage")
        if cov:
            rows.append({"slug": lg.slug, "name": lg.name,
                         "country": lg.country, **cov})
    json.dump({"generated": dt.datetime.now(dt.timezone.utc)
               .isoformat(timespec="seconds"), "leagues": rows},
              open(os.path.join(OUT, "coverage.json"), "w"), indent=1)
    thin = [r["slug"] for r in rows if r["missing_seasons"] or r["partial_seasons"]]
    print(f"  → coverage.json ({len(rows)} competitions"
          + (f", {len(thin)} with feed gaps: {', '.join(thin)}" if thin else "")
          + ")")
    # The changelog is written by hand and lives with the source, not with the
    # generated output; the build copies it so the site fetches one directory.
    src = os.path.join(HERE, "data", "changelog.json")
    if os.path.exists(src):
        with open(src) as fh:
            log = json.load(fh)
        json.dump(log, open(os.path.join(OUT, "changelog.json"), "w"), indent=1)
        print(f"  → changelog.json ({len(log.get('entries', []))} entries)")


def probe_external(quiet: bool = False) -> list[dict]:
    """Fetch and check every second feed, and say what happened.

    This is the half of the verification that cannot run where the code is
    written. The development sandbox reaches GitHub and nothing else, so the
    readers in `footballdata.py` and `wikifootball.py` ship unfetched; here, on a
    runner with open egress, they are fetched, parsed, lined up against a season
    the GitHub feed already gave us, and either armed or refused with a reason.

    Safe to run on its own -- `python -m model.run --probe` -- which is how to
    find out whether a new league's club names resolve before wiring it in.
    """
    from .parse import TeamRegistry
    reg = TeamRegistry()
    dom = europe.load_domestic(reg, quiet=True)
    _, verdicts = europe.load_external(reg, dom, quiet=quiet)
    return [v.as_json() for v in verdicts]


def build_sources(_ready: set[str]) -> None:
    """`sources.json`: which competitions are running on a second feed.

    Written from whatever the last probe found rather than from the source list,
    so the file describes what happened and not what was hoped for. A build
    whose second feeds were all unreachable publishes that fact.
    """
    verdicts = [v.as_json() for v in europe.LAST_VERDICTS]
    if not verdicts:
        verdicts = probe_external(quiet=True)
    used = [v for v in verdicts if not v.get("watching")]
    armed = [v for v in used if v["ok"]]
    json.dump({"generated": dt.datetime.now(dt.timezone.utc)
               .isoformat(timespec="seconds"),
               "note": ("Second feeds, off GitHub, for competitions whose "
                        "GitHub source stopped publishing. A feed is used only "
                        "if it reproduces a season this site already holds."),
               "sources": verdicts},
              open(os.path.join(OUT, "sources.json"), "w"), indent=1)
    watched = [v for v in verdicts if v.get("watching")]
    print(f"  → sources.json ({len(armed)} of {len(used)} second feed(s) armed"
          + (f", {len(watched)} watched)" if watched else ")"))
    for v in verdicts:
        if not v["ok"]:
            print(f"    · {v['source']}/{v['league']}: {v['reason']}"
                  + (" -- " + ", ".join(v["unresolved"]) if v["unresolved"] else ""))
        elif v.get("watching"):
            print(f"    · {v['source']}/{v['league']}: would arm, "
                  f"{v['agreed']}/{v['compared']} of {v['overlap_season']}")


#: How stale a competition's newest result may be, in days, before the build
#: says so. Chosen from what the feeds actually do rather than from a round
#: number: openfootball's in-season commits come roughly weekly with multi-week
#: gaps, and the results mirror does not create a season's file until months in.
#: Three weeks is past both of those and into "somebody stopped updating this".
STALE_DAYS = 21


def check_feeds(ready: set[str]) -> None:
    """Say out loud when a competition's feed has gone quiet.

    Belgium is why this exists. openfootball's `2025-26/be1.txt` stopped at 121
    of 240 matches when the maintainer moved on to the Wallonian provincial
    leagues, and nothing in the build noticed: a feed that stops updating
    produces exactly the same output as a feed with nothing new to say, so a
    frozen competition looked like a quiet one for months.

    This does not fail the build. A stale feed is still the best evidence
    available and a forecast built on it is still the right forecast; the
    failure mode is not knowing. So it prints, and the site says the same thing
    on its own method page.
    """
    today = dt.date.today()
    quiet, missing = [], []
    for lg in leagues.LEAGUES:
        if lg.slug not in ready:
            continue
        try:
            fc = json.load(open(os.path.join(OUT, lg.slug, "forecast.json")))
        except (OSError, ValueError):
            continue
        src = fc.get("sources", {})
        cov = fc.get("coverage", {})
        # A feed with no results yet is a season that has not started, not a
        # feed that has stopped. Only a season already under way can go quiet.
        to = src.get("results_to")
        if to and src.get("played"):
            age = (today - dt.date.fromisoformat(to)).days
            if age > STALE_DAYS:
                quiet.append((lg.slug, to, age))
        for key in ("mirror", "fixtures"):
            block = src.get(key) or {}
            if block.get("url") and not block.get("available"):
                missing.append((lg.slug, key, block["url"]))
        if cov.get("missing_seasons") or cov.get("partial_seasons"):
            gaps = ", ".join(cov.get("missing_seasons", [])
                             + [f"{s} (partial)" for s in cov.get("partial_seasons", [])])
            print(f"  · {lg.slug}: feed has holes -- {gaps}")
    for slug, key, url in missing:
        print(f"!! {slug}: the {key} feed did not resolve ({url})")
    for slug, to, age in quiet:
        print(f"!! {slug}: newest result is {to}, {age} days ago -- feed may have "
              "stopped updating")
    if not quiet and not missing:
        print("  → every feed is current")


def build_clubs(ready: set[str]) -> None:
    """Founded year and city per club, from `openfootball/clubs`.

    The one thing a club page can say that no amount of results will tell you.
    Matching is on this repository's own alias table rather than on display
    names, and a club the register does not have is left out rather than given
    a placeholder -- see `model/clubmeta` for what is deliberately not read.
    """
    path = os.path.join(HERE, "data", "team_meta.json")
    try:
        with open(path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        print("  · team_meta.json unreadable, skipping club metadata")
        return
    got = clubmeta.build(OUT, meta)
    print(f"  → clubs.json ({got['matched']} of {got['of']} clubs matched "
          "to the openfootball register)")


def build_shooting(ready: set[str]) -> None:
    """One file of per-club shooting, merged from the leagues that have it.

    Each league already writes its own into `gamestate.json`; the comparison
    page is not scoped to a league and cannot fetch nine of those to answer one
    question. So this reads the files the build has just written -- no second
    pass over the corpus -- and keys them by club.

    Big five only. Four of the nine competitions here are read from openfootball,
    which carries goals and nothing else, and a club from one of them is absent
    rather than present with zeroes.
    """
    clubs, leagues_with = {}, {}
    for lg in leagues.LEAGUES:
        if lg.slug not in ready:
            continue
        try:
            gs = json.load(open(os.path.join(OUT, lg.slug, "gamestate.json")))
        except (OSError, ValueError):
            continue
        shot = gs.get("shooting") or {}
        if not shot:
            continue
        leagues_with[lg.slug] = gs.get("shooting_average") or {}
        # Discipline rides along, because it comes from the same five feeds and
        # is wanted for the same reason: one file the comparison page can read
        # to rate a club on the same scale as every other club that has a shot
        # feed. Assembled here rather than stored six ways -- a yellow, three
        # for a red, and a sixth of a foul, which puts the three on a comparable
        # footing without pretending a foul is a booking.
        disc = gs.get("discipline") or {}
        for cid, row in shot.items():
            d = disc.get(cid) or {}
            extra = {}
            if d.get("n"):
                extra = {"yellow_pm": d.get("yellow_pm"), "red_pm": d.get("red_pm"),
                         "fouls_pm": d.get("fouls_pm"),
                         "foul_index": round(d["yellow_pm"] + 3 * d["red_pm"]
                                             + d["fouls_pm"] / 6, 3)}
            clubs[cid] = {**row, **extra, "league": lg.slug}
    json.dump({"generated": dt.datetime.now(dt.timezone.utc)
               .isoformat(timespec="seconds"),
               "note": ("Shots, shots on target and discipline, five seasons "
                        "per club. The results mirror carries them for the big "
                        "five only; no other feed this build can reach has a "
                        "shot or a card in it."),
               "averages": leagues_with, "clubs": clubs},
              open(os.path.join(OUT, "shooting.json"), "w"), indent=1)
    print(f"  → shooting.json ({len(clubs)} clubs from "
          f"{len(leagues_with)} competition(s) with a shot feed)")


def build_ratings(ready: set[str]) -> None:
    """Every club rating, on one scale, in one file. The only source of them.

    Attack and defence on a competition page used to be measured against that
    competition's own average, which made them useless for the comparison people
    most want to make: Club Brugge's attack against Manchester City's. Both read
    somewhere in the sixties or seventies, and neither number knew about the
    other. A rating that is only meaningful next to its own neighbours is a rank
    with extra steps.

    Fixing those two and leaving the other six alone half-solved it, and the
    half that was left was worse than obvious: the comparison page drew one
    radar shape out of two axes on a shared scale and five that were each
    relative to a different division. So all eight are global now, and this file
    is the only place any of them exists. `forecast.json` carries none, which is
    what stops a page picking up a league-scale rating by accident -- the way the
    league table did for months while the club page had it right.

    Four come from the pooled fit by way of `global.json`, SPI among them: a
    league page used to quote its own, which put Sporting CP at 89.3 and FC
    Barcelona at 80.8 on two club pages of the same site. Three more need a shot
    or a card and so exist only for the big five, and are measured against the mean
    of that whole population rather than against each of its five leagues.

    There were eight for about an hour. Two of them, home advantage and big
    games, turned out to be 7% and 4% signal: see `model/scale.py`, which is
    where the reliability of each of these is written down and where the four
    that survive are shrunk by it.

    A club with no entry here gets no rating rather than a rating on some other
    scale. That is sixteen of the two hundred and ten clubs with a forecast page:
    sides projected up from a division the pooled corpus does not carry, whose
    only top-flight record is too old to rate from. Blank is the honest answer.
    """
    try:
        with open(os.path.join(OUT, "global.json"), encoding="utf-8") as fh:
            g = json.load(fh)
    except (OSError, ValueError):
        print("  · global.json unreadable, skipping ratings.json")
        return
    keep = ("spi", "spi_lo", "spi_hi", "att_r", "def_r", "consistency_r")
    clubs: dict[str, dict] = {}
    for c in g.get("clubs", []):
        row = {k: c[k] for k in keep if c.get(k) is not None}
        if row:
            clubs[c["id"]] = row

    # The three that need a shot feed. One reference for all five leagues, which
    # is what makes a Serie A creation rating and a Bundesliga one the same
    # number; the population that has the data is the big five entire.
    shot_fields = (("creation_r", "creation", "sot_pm", True),
                   ("finishing_r", "finishing", "conversion", True),
                   ("discipline_r", "discipline", "foul_index", False))
    n_shot = 0
    try:
        with open(os.path.join(OUT, "shooting.json"), encoding="utf-8") as fh:
            sh = (json.load(fh).get("clubs") or {})
    except (OSError, ValueError):
        sh = {}
    if sh:
        refs = {}
        for _f, dim, key, log in shot_fields:
            vals = [r[key] for r in sh.values()
                    if isinstance(r, dict) and r.get(key) is not None]
            if vals:
                refs[key] = (math.exp(sum(math.log(v) for v in vals if v > 0)
                                      / len([v for v in vals if v > 0]))
                             if log else sum(vals) / len(vals))
        for cid, r in sh.items():
            got = {}
            for field, dim, key, log in shot_fields:
                v = scale.dimension(dim, r.get(key), refs.get(key), log=log)
                if v is not None:
                    got[field] = v
            if got:
                clubs.setdefault(cid, {}).update(got)
                n_shot += 1

    json.dump({"generated": g.get("generated"),
               "scale": {"global": ["spi", "spi_lo", "spi_hi",
                                    "att_r", "def_r", "consistency_r",
                                    "creation_r", "finishing_r",
                                    "discipline_r"],
                         "league": []},
               "note": ("Every rating here is measured against one reference, "
                        "so a 70 means the same thing in any league. SPI, "
                        "attack, defence and consistency come from the pooled "
                        "European "
                        "fit; creation, finishing and discipline need a shot or "
                        "a card and exist for the big five only, measured "
                        "against the mean of all five together. Each is shrunk "
                        "toward the middle by how much of it a club's matches "
                        "can actually resolve."),
               "clubs": clubs},
              open(os.path.join(OUT, "ratings.json"), "w"), separators=(",", ":"))
    print(f"  → ratings.json ({len(clubs)} clubs, {n_shot} with a shot feed, "
          "six ratings on one scale)")


def build_seo(ready: set[str]) -> None:
    """robots.txt, the sitemap, and a static stub per club."""
    manifest = {"default": leagues.DEFAULT.slug,
                "leagues": [lg.manifest_entry(lg.slug in ready)
                            for lg in leagues.LEAGUES + leagues.EUROPEAN]}
    forecasts = {}
    for slug in sorted(ready):
        try:
            forecasts[slug] = json.load(
                open(os.path.join(OUT, slug, "forecast.json")))
        except (OSError, ValueError):
            continue
    got = seo.build(OUT, manifest, forecasts)
    print(f"  → robots.txt, sitemap.xml ({got['leagues']} competitions), "
          f"{got['stubs']} club pages")


#: What the Pages artefact may weigh before the build complains. Not a hard
#: failure -- a forecast that is a megabyte over is still a forecast -- but a
#: number somebody has to look at, because 8 MB of share cards arrived without
#: anybody deciding to spend it.
SIZE_BUDGET_MB = 26


def check_size(ready: set[str]) -> None:
    site = os.path.join(HERE, "site")
    by_dir: dict[str, int] = {}
    total = 0
    for root, _dirs, files in os.walk(site):
        for f in files:
            try:
                sz = os.path.getsize(os.path.join(root, f))
            except OSError:
                continue
            total += sz
            top = os.path.relpath(root, site).split(os.sep)[0]
            by_dir[top] = by_dir.get(top, 0) + sz
    mb = total / 1e6
    parts = ", ".join(f"{k} {v / 1e6:.1f}MB" for k, v in
                      sorted(by_dir.items(), key=lambda kv: -kv[1])[:4])
    print(f"  → site is {mb:.1f}MB ({parts})")
    if mb > SIZE_BUDGET_MB:
        print(f"  ! over the {SIZE_BUDGET_MB}MB budget by {mb - SIZE_BUDGET_MB:.1f}MB "
              "- see SIZE_BUDGET_MB in model/run.py")


def _rated_only() -> list[dict]:
    """Every league in the global ranking that has no forecast page, by size.

    Read off `global.json` rather than from a registry, because the set is
    whatever the corpus turned out to contain and a hand-kept copy of that would
    be wrong by the next feed.
    """
    try:
        with open(os.path.join(OUT, "global.json"), encoding="utf-8") as fh:
            clubs = json.load(fh)["clubs"]
    except (OSError, ValueError, KeyError):
        return []
    counts: dict[tuple[str, str], int] = {}
    for c in clubs:
        if c.get("slug"):
            continue                       # a competition with its own page
        key = (c["league"], c.get("country") or "")
        counts[key] = counts.get(key, 0) + 1
    return [{"name": name, "country": country, "n": n}
            for (name, country), n in sorted(counts.items(), key=lambda kv: -kv[1])]


def _projected() -> list[dict]:
    """Leagues with a projection written on disk, in the order the picker lists."""
    out = []
    for assoc, slug in sorted(PROJECTION_SLUG.items(), key=lambda kv: kv[1]):
        path = os.path.join(OUT, slug, "projection.json")
        try:
            with open(path, encoding="utf-8") as fh:
                p = json.load(fh)
        except (OSError, ValueError):
            continue
        out.append({"slug": slug, "name": p["name"], "country": p["country"],
                    "season": p["season"], "n_teams": p["n_teams"]})
    return out


def write_manifest(ready: set[str]) -> dict:
    """Regenerate site/data/leagues.json from the registry.

    'ready' means the directory on disk actually has a forecast in it, so a
    league that failed to build this run drops back to false and the site shows
    it as coming soon instead of 404ing.
    """
    payload = {
        "default": leagues.DEFAULT.slug,
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "leagues": [lg.manifest_entry(lg.slug in ready)
                    for lg in leagues.LEAGUES + leagues.EUROPEAN],
        "note": "ready=false means no live forecast; the directory may still "
                "hold replay staging data, stamped with a 'replay' key.",
        # Leagues the pooled fit rates but this site does not forecast: the
        # second tiers and the fifty-odd domestic divisions the corpus carries
        # for the ratings. They have no table, no fixture list and no page of
        # their own, and until now no way to reach them either -- the league
        # picker offered nine competitions while the ranking held sixty, and the
        # only door to the other fifty-one was a filter on one page that did not
        # even survive being linked to. The picker sends these to that filter.
        "rated": _rated_only(),
        # And the ones in between: no fixture list anywhere, so no forecast
        # page, but a results grid this site has checked, which is enough to
        # project the final table. Read off what was actually written, so a
        # league whose article went missing this run leaves the picker rather
        # than sending a reader to a file that is not there.
        "projected": _projected(),
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(payload, open(os.path.join(OUT, "leagues.json"), "w"), indent=1)
    return payload


def has_forecast(league: leagues.League) -> bool:
    """A forecast on disk that is actually a forecast of the season ahead.

    Replay staging data lives in the same directory and has the same shape, so
    the manifest asks the file itself: anything stamped `replay` is development
    material and must never make the switcher entry live."""
    path = os.path.join(OUT, league.slug, "forecast.json")
    if not os.path.exists(path):
        return False
    try:
        return json.load(open(path)).get("replay") is None
    except (ValueError, OSError):
        return False


def drop_legacy_flat() -> None:
    """Remove the flat duplicates of the default league's output.

    Kept as a build step rather than a one-off deletion because the files are
    committed: a checkout from before this change still has them, and a build
    that leaves them lying around leaves a Premier League forecast frozen at
    whatever date the copies stopped being written.
    """
    gone = []
    for name in RETIRED_FLAT_FILES:
        p = os.path.join(OUT, name)
        if os.path.exists(p):
            os.remove(p)
            gone.append(name)
    if gone:
        print(f"  → removed {len(gone)} retired flat file(s) from site/data/")


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--league", "-l", action="append", metavar="SLUG",
                    help="build only this league (repeatable); default is all five")
    ap.add_argument("--skip-backtest", action="store_true",
                    help="same as SKIP_BACKTEST=1")
    ap.add_argument("--replay", metavar="SEASON",
                    help="build a cup league from a finished season's real "
                         "league phase, as staging data (e.g. --replay 2025-26)")
    ap.add_argument("--pooled", action="store_true",
                    help="fit domestic leagues on the pooled European corpus "
                         "(off by default; see the Phase 2 gate)")
    ap.add_argument("--probe", action="store_true",
                    help="check the second feeds outside GitHub and exit; the "
                         "only place that check can run is a machine with open "
                         "egress, which is a runner and not the sandbox")
    args = ap.parse_args(argv)

    if args.probe:
        rows = probe_external()
        # A watched feed is not expected to arm: it is being probed precisely to
        # find out what it would do, and it contributes nothing either way. Only
        # a feed the build actually relies on can fail this.
        used = [r for r in rows if not r.get("watching")]
        bad = sum(1 for r in used if not r["ok"])
        raise SystemExit(0 if not bad else
                         f"{bad} second feed(s) did not arm")

    try:
        todo = ([leagues.get(s) for s in args.league] if args.league
                else list(leagues.LEAGUES))
    except KeyError as exc:
        raise SystemExit(str(exc).strip('"')) from None
    skip = args.skip_backtest or os.environ.get("SKIP_BACKTEST") == "1"
    if args.replay and not any(lg.kind == "cup" for lg in todo):
        raise SystemExit("--replay only applies to a cup; pass "
                         "--league champions-league")

    t0 = time.perf_counter()
    timings: list[tuple[str, float]] = []
    failures: list[tuple[str, Exception]] = []
    for lg in todo:
        t1 = time.perf_counter()
        try:
            if lg.kind == "cup":
                build_cup(lg, replay=args.replay, skip_backtest=skip)
            else:
                build(lg, skip_backtest=skip, pooled=args.pooled or None)
        except Exception as exc:                      # noqa: BLE001
            # One league's source going missing must not take the other four
            # down: the manifest marks it not-ready and the run exits non-zero.
            failures.append((lg.slug, exc))
            print(f"!! {lg.slug} failed: {exc}")
        timings.append((lg.slug, time.perf_counter() - t1))

    ready = {lg.slug for lg in leagues.LEAGUES + leagues.EUROPEAN
             if has_forecast(lg)}
    # Cross-competition outputs, once, after every league has been written.
    # Each is optional: a failure here must not cost the forecasts that already
    # landed, so it is reported and the run carries on.
    for name, fn in (("projections", build_projections),
                     ("global rankings", build_rankings), ("feeds", build_feeds),
                     ("coverage", build_coverage),
                     ("second feeds", build_sources),
                     ("club register", build_clubs),
                     ("pooled ratings", build_ratings),
                     ("shooting", build_shooting),
                     ("retired flat files", lambda _r: drop_legacy_flat()),
                     ("feed freshness", check_feeds),
                     ("sitemap and club pages", build_seo),
                     ("size budget", check_size)):
        try:
            fn(ready)
        except Exception as exc:                   # noqa: BLE001
            print(f"!! {name} failed: {exc}")
            failures.append((name, exc))
    write_manifest(ready)
    total = time.perf_counter() - t0
    print("\n--- timings ---")
    for slug, secs in timings:
        print(f"  {slug:15s} {secs:7.1f}s")
    print(f"  {'TOTAL':15s} {total:7.1f}s")
    print(f"Manifest: {len(ready)}/{len(leagues.LEAGUES)} leagues ready "
          f"({', '.join(sorted(ready))}).")
    if failures:
        raise SystemExit(f"{len(failures)} league(s) failed: "
                         + ", ".join(s for s, _ in failures))


if __name__ == "__main__":
    main()
