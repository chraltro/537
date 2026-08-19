"""Build every JSON file the site reads, for one league or for all five.

    python -m model.run                     # all five leagues
    python -m model.run --league la-liga    # just one
    SKIP_BACKTEST=1 python -m model.run     # skip the walk-forward evaluation

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

from . import backtest, config, insight, leagues, priors, ratings, simulate
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
    """SPI at the start of each of the last few seasons, so each club has a
    trajectory to show rather than a single number with no context.

    The final point carries the same prior and market adjustment as the headline
    rating, so the trajectory ends exactly where the club's SPI is quoted.
    """
    hist: dict[str, list] = {t: [] for t in teams}
    seasons = sorted({m.season for m in ds.top if m.season != ds.season})[-4:]
    points = [(s, dt.date(int(s.split("-")[0]), 8, 1)) for s in seasons]
    points.append((ds.season, kickoff))
    for label, ref in points:
        past = [m for m in ds.top if m.date < ref] + [m for m in ds.second if m.date < ref]
        if len(past) < 1000:
            continue
        pool = sorted({m.home for m in past} | {m.away for m in past})
        f = ratings.fit(past, pool, ref, shot_conv=shot_conv)
        live = label == ds.season
        for t in teams:
            if t in f.index:
                a = adj.get(t, 0.0) if live else 0.0
                hist[t].append({"season": label, "spi": round(spi(f, t, a), 1)})
    return hist


# --------------------------------------------------------------------------
# One league
# --------------------------------------------------------------------------
def build(league: leagues.League, *, skip_backtest: bool | None = None) -> None:
    """Run the whole pipeline for one league and write its JSON directory."""
    if skip_backtest is None:
        skip_backtest = os.environ.get("SKIP_BACKTEST") == "1"
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
    hist = [m for m in ds.top if m.date < ref] + [m for m in ds.second if m.date < ref]
    pool = sorted({m.home for m in hist} | {m.away for m in hist})
    fit = ratings.fit(hist, pool, ref, shot_conv=shot_conv)

    print("Calibrating priors against history…")
    cal = priors.calibrate(ds, shot_conv)
    print(f"  · continuing slope {cal['continuing']['slope']:.3f} "
          f"(n={cal['continuing']['n']}, from {cal['continuing'].get('source', league.slug)})"
          f", promoted slope {cal['promoted']['slope']:.3f} "
          f"(n={cal['promoted']['n']}, from {cal['promoted'].get('source', league.slug)})")
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
                  f"{league.market_file} absent) — ratings alone.")
        adj = base_adj
        w = 0.0

    print(f"Simulating the season {config.N_SIMS:,} times…")
    sim = simulate.simulate_season(fit, ds.fixtures, teams, league=league,
                                   adj=adj, leverage=True)

    table = ds.season_table(ds.season)
    idx = {t: i for i, t in enumerate(teams)}
    history = _rating_history(ds, teams, shot_conv, adj, kickoff)
    returning = {m.home for m in ds.top if m.season == prev_season}

    rows = []
    for t in teams:
        i = idx[t]
        m = meta[t]
        cur = table.get(t, {})
        a = adj.get(t, 0.0)
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
            "pos": [round(float(x), 5) for x in sim["position"][i]],
            "played": cur.get("pld", 0), "w": cur.get("w", 0), "d": cur.get("d", 0),
            "l": cur.get("l", 0), "gf": cur.get("gf", 0), "ga": cur.get("ga", 0),
            "cur_pts": cur.get("pts", 0),
            "history": history.get(t, []),
            "promoted": t not in returning,
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
        ms.append({
            "md": f.matchday, "date": f.date.isoformat(), "time": f.time,
            "h": f.home, "a": f.away,
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
    try:
        from . import recap
        recap.write_recap(os.path.join(out, "recap.json"), rows, played)
        print("  → recap.json")
    except ImportError:
        pass

    frozen = insight.freeze_predictions(ms, out)
    report = insight.season_report(ms, frozen, {t: meta[t]["name"] for t in teams})
    json.dump(report, open(os.path.join(out, "season_report.json"), "w"),
              separators=(",", ":"))
    snaps = insight.append_history(rows, played, out)
    print(f"  → schedule.json, predictions.json, season_report.json "
          f"({report['n']} scored), history.json ({len(snaps)} snapshots)")

    json.dump({
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
        "teams": rows,
    }, open(os.path.join(out, "forecast.json"), "w"), separators=(",", ":"))
    print(f"  → forecast.json ({len(rows)} teams)")

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


def write_manifest(ready: set[str]) -> dict:
    """Regenerate site/data/leagues.json from the registry.

    'ready' means the directory on disk actually has a forecast in it, so a
    league that failed to build this run drops back to false and the site shows
    it as coming soon instead of 404ing.
    """
    payload = {
        "default": leagues.DEFAULT.slug,
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "leagues": [lg.manifest_entry(lg.slug in ready) for lg in leagues.LEAGUES],
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(payload, open(os.path.join(OUT, "leagues.json"), "w"), indent=1)
    return payload


def has_forecast(league: leagues.League) -> bool:
    return os.path.exists(os.path.join(OUT, league.slug, "forecast.json"))


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
    args = ap.parse_args(argv)

    try:
        todo = ([leagues.get(s) for s in args.league] if args.league
                else list(leagues.LEAGUES))
    except KeyError as exc:
        raise SystemExit(str(exc).strip('"')) from None
    skip = args.skip_backtest or os.environ.get("SKIP_BACKTEST") == "1"

    t0 = time.perf_counter()
    timings: list[tuple[str, float]] = []
    failures: list[tuple[str, Exception]] = []
    for lg in todo:
        t1 = time.perf_counter()
        try:
            build(lg, skip_backtest=skip)
        except Exception as exc:                      # noqa: BLE001
            # One league's source going missing must not take the other four
            # down: the manifest marks it not-ready and the run exits non-zero.
            failures.append((lg.slug, exc))
            print(f"!! {lg.slug} failed: {exc}")
        timings.append((lg.slug, time.perf_counter() - t1))

    if leagues.DEFAULT in todo and has_forecast(leagues.DEFAULT):
        copy_legacy(leagues.DEFAULT)

    ready = {lg.slug for lg in leagues.LEAGUES if has_forecast(lg)}
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
