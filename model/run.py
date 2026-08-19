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
import os
import shutil
import time

import numpy as np

from . import (backtest, config, europe, feeds, gamestate, insight, knockout,
               leagues, priors, rankings, ratings, simulate, social)
from .data import Dataset

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "site", "data")

#: Files copied from site/data/premier-league/ to site/data/ for the pages that
#: have not moved to the per-league layout yet.
LEGACY_FILES = ("forecast.json", "matches.json", "schedule.json", "history.json",
                "predictions.json", "season_report.json", "recap.json",
                "sim_input.json", "backtest.json")


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


def _rating_history(ds: Dataset, teams, shot_conv, adj: dict[str, float],
                    kickoff: dt.date) -> dict[str, list]:
    """SPI at the start of every season in the record, so each club has a
    trajectory rather than a single number with no context.

    This used to stop at four seasons, which is enough to see a trend and far
    too little to see a club. The mirror carries all five big-five leagues back
    to 1993-94, and one fit costs about four tenths of a second on the full
    corpus, so the whole run is a handful of seconds per league — a cheap price
    for the one chart on the site that shows a decade instead of a season.

    Every point is fitted only on matches played before that season started, so
    the line is a walk-forward record and not a curve drawn through hindsight.
    The final point carries the same prior and market adjustment as the headline
    rating, so the trajectory ends exactly where the club's SPI is quoted.
    """
    hist: dict[str, list] = {t: [] for t in teams}
    seasons = sorted({m.season for m in ds.top if m.season != ds.season})
    points = [(s, dt.date(int(s.split("-")[0]), 7, 1)) for s in seasons]
    points.append((ds.season, kickoff))
    for label, ref in points:
        past = ds.before(ref)
        if len(past) < 1000:
            continue                   # too little history to fit anything honest
        pool = sorted({m.home for m in past} | {m.away for m in past})
        f = ratings.fit(past, pool, ref, shot_conv=shot_conv)
        live = label == ds.season
        for t in teams:
            if t in f.index:
                a = adj.get(t, 0.0) if live else 0.0
                hist[t].append({"season": label, "spi": round(spi(f, t, a), 1)})
    return hist


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
        for rank, t in enumerate(fc["teams"], 1):
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
    pool = sorted({m.home for m in hist} | {m.away for m in hist})
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

    print(f"Simulating the season {config.N_SIMS:,} times…")
    sim = simulate.simulate_season(fit, ds.fixtures, teams, league=league,
                                   adj=adj, leverage=True)

    # Half time is a second, cheaper model over the same corpus, and the only
    # thing the results feed carries that the forecast has never read.
    ht_fit = gamestate.half_time_fit(hist, ref)
    print("  · half-time model: "
          + ("fitted" if ht_fit else "not enough half-time data in this feed"))

    table = ds.season_table(ds.season)
    idx = {t: i for i, t in enumerate(teams)}
    history = _rating_history(ds, teams, shot_conv, adj, kickoff)
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
            "spi": round(spi(fit, t, a), 1),
            "off": round(fit.offence(t) * np.exp(a / 2), 2),
            "def": round(fit.defence(t) * np.exp(-a / 2), 2),
            "pts": round(float(sim["points_mean"][i]), 1),
            "pts_lo": round(float(sim["points_p10"][i])),
            "pts_hi": round(float(sim["points_p90"][i])),
            "pts_min": int(sim["points_min"][i]), "pts_max": int(sim["points_max"][i]),
            "gd": round(float(sim["gd_mean"][i])),
            "title": float(sim["title"][i]), "ucl": float(sim["ucl"][i]),
            "europa": float(sim["europa"][i]), "releg": float(sim["relegation"][i]),
            **({"p_playoff": playoff} if promo else {}),
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
    spi_by_team = {r["id"]: r["spi"] for r in rows}
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
    fifty-two leagues -- which would put every Champions League participant
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
            "spi": round(spi(fit, t, (adj or {}).get(t, 0.0)), 1),
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
            "history": [], "promoted": False, "arrived": "stayed",
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

    spi_by_team = {r["id"]: r["spi"] for r in rows}
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
#: Competitions already folded into the shared corpus, so a second call does not
#: add the same matches twice and double their weight in the fit.
_RANKED_EXTRA: set[str] = set()


def shared_corpus(reg=None, *, quiet: bool = True) -> europe.Corpus:
    global _CORPUS
    if _CORPUS is None:
        from .parse import TeamRegistry
        _CORPUS = europe.Corpus(reg or TeamRegistry()).load(quiet=quiet)
    return _CORPUS


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
    # Second tiers this site forecasts in their own right belong in the pooled
    # fit too, or the Championship would have a projected table on one page and
    # no rating on the other. Loaded through the same registry, so a club that
    # goes up keeps one id and one rating history.
    for lg in leagues.LEAGUES:
        if lg.kind != "promotion" or lg.slug in _RANKED_EXTRA:
            continue
        try:
            ds = Dataset(lg).load()
        except Exception as exc:                   # noqa: BLE001
            print(f"  ! {lg.slug} not added to the pooled corpus: {exc}")
            continue
        corpus.add([m for m in ds.top if m.played], lg.slug)
        _RANKED_EXTRA.add(lg.slug)
        print(f"  · added {len(ds.top)} {lg.name} matches to the pooled corpus")
    payload = rankings.build(corpus, featured=featured)
    json.dump(payload, open(os.path.join(OUT, "global.json"), "w"),
              separators=(",", ":"))
    print(f"  → global.json ({payload['n_clubs']} clubs across "
          f"{payload['n_leagues']} leagues, from {payload['n_matches']:,} matches)")
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
            "meta": {t["id"]: t for t in fc["teams"]},
            "url": f"{HOME}races.html"
                   + (f"?lg={lg.slug}" if lg.slug != leagues.DEFAULT.slug else ""),
        })
    items = feeds.feed_items(rows)
    title = "Ninety: forecast movement"
    site = os.path.join(HERE, "site")
    feeds.write(os.path.join(site, "feed.json"),
                json.dumps(feeds.json_feed(items, home=HOME, title=title),
                           separators=(",", ":")))
    feeds.write(os.path.join(site, "feed.xml"),
                feeds.rss(items, home=HOME, title=title))
    print(f"  → feed.json and feed.xml ({len(items)} item(s) with real movement)")


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


def copy_legacy(league: leagues.League) -> None:
    """Mirror one league's output to the flat site/data/*.json the site still reads."""
    src = os.path.join(OUT, league.slug)
    n = 0
    for name in LEGACY_FILES:
        p = os.path.join(src, name)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(OUT, name))
            n += 1
    print(f"Copied {n} {league.slug} files to the legacy flat site/data/ paths.")


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
    args = ap.parse_args(argv)

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

    if leagues.DEFAULT in todo and has_forecast(leagues.DEFAULT):
        copy_legacy(leagues.DEFAULT)

    ready = {lg.slug for lg in leagues.LEAGUES + leagues.EUROPEAN
             if has_forecast(lg)}
    # Cross-competition outputs, once, after every league has been written.
    # Each is optional: a failure here must not cost the forecasts that already
    # landed, so it is reported and the run carries on.
    for name, fn in (("global rankings", build_rankings), ("feeds", build_feeds)):
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
