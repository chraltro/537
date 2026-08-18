"""Build every JSON file the site reads."""
from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np

from . import backtest, config, priors, ratings, simulate
from .data import Dataset

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "site", "data")
KICKOFF = dt.date(2026, 8, 21)


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


def _rating_history(ds: Dataset, teams, shot_conv, adj: dict[str, float]) -> dict[str, list]:
    """SPI at the start of each of the last few seasons, so each club has a
    trajectory to show rather than a single number with no context.

    The final point carries the same prior and market adjustment as the headline
    rating, so the trajectory ends exactly where the club's SPI is quoted.
    """
    hist: dict[str, list] = {t: [] for t in teams}
    seasons = sorted({m.season for m in ds.pl if m.season != config.SEASON})[-4:]
    points = [(s, dt.date(int(s.split("-")[0]), 8, 1)) for s in seasons]
    points.append((config.SEASON, KICKOFF))
    for label, ref in points:
        past = [m for m in ds.pl if m.date < ref] + [m for m in ds.ch if m.date < ref]
        if len(past) < 1000:
            continue
        pool = sorted({m.home for m in past} | {m.away for m in past})
        f = ratings.fit(past, pool, ref, shot_conv=shot_conv)
        live = label == config.SEASON
        for t in teams:
            if t in f.index:
                a = adj.get(t, 0.0) if live else 0.0
                hist[t].append({"season": label, "spi": round(spi(f, t, a), 1)})
    return hist


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    ds = Dataset().load()
    teams = ds.teams
    meta = ds.reg.meta

    print("Fitting ratings…")
    shot_conv = ratings.fit_shot_conversion(ds.pl)
    today = dt.date.today()
    ref = max(today, KICKOFF)
    hist = [m for m in ds.pl if m.date < ref] + [m for m in ds.ch if m.date < ref]
    pool = sorted({m.home for m in hist} | {m.away for m in hist})
    fit = ratings.fit(hist, pool, ref, shot_conv=shot_conv)

    print("Calibrating priors against history…")
    cal = priors.calibrate(ds, shot_conv)
    raw_net = priors._centred_net(fit, teams)
    prior_net = priors.preseason_net(ds, fit, cal, teams, sorted(
        {m.season for m in ds.pl if m.season != config.SEASON})[-1])
    base_adj = {t: prior_net[t] - raw_net[t] for t in teams}

    played = sum(1 for f in ds.fixtures if f.played)
    market = priors.load_market(os.path.join(HERE, "data", "market_priors.json"))
    w = priors.market_weight(played)
    if market and w > 0:
        print(f"Anchoring to the preseason market (weight {w:.2f})…")
        fitted = priors.fit_market_adjustment(fit, ds.fixtures, teams, market,
                                              base_adj=base_adj, verbose=True)
        adj = {t: base_adj[t] + w * (fitted[t] - base_adj[t]) for t in teams}
    else:
        adj = base_adj

    print(f"Simulating the season {config.N_SIMS:,} times…")
    sim = simulate.simulate_season(fit, ds.fixtures, teams, adj=adj)

    table = ds.season_table(config.SEASON)
    idx = {t: i for i, t in enumerate(teams)}
    history = _rating_history(ds, teams, shot_conv, adj)

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
            "promoted": t not in {mm.home for mm in ds.pl
                                  if mm.season == sorted({x.season for x in ds.pl
                                                          if x.season != config.SEASON})[-1]},
        })
    rows.sort(key=lambda r: (-r["pts"], -r["gd"]))

    json.dump({
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "season": config.SEASON_LABEL,
        "matches_played": played,
        "matches_total": config.N_MATCHES,
        "market_weight": round(w, 3),
        "home_advantage": round(float(np.exp(fit.home)), 3),
        "ucl_places": config.UCL_PLACES,
        "n_sims": int(sim["n_sims"]),
        "teams": rows,
    }, open(os.path.join(OUT, "forecast.json"), "w"), separators=(",", ":"))
    print(f"  → forecast.json ({len(rows)} teams)")

    print("Writing match forecasts…")
    ms = []
    for f in sorted(ds.fixtures, key=lambda x: (x.matchday or 0, x.date, x.home)):
        rep = simulate.match_report(fit, f.home, f.away, adj)
        best = rep["top_scores"][0]
        ms.append({
            "md": f.matchday, "date": f.date.isoformat(), "time": f.time,
            "h": f.home, "a": f.away,
            "ph": round(rep["home_win"], 4), "pd": round(rep["draw"], 4),
            "pa": round(rep["away_win"], 4),
            "xgh": round(rep["xg_home"], 2), "xga": round(rep["xg_away"], 2),
            "sc": [best["h"], best["a"]], "scp": round(best["p"], 4),
            "alt": [[s["h"], s["a"], round(s["p"], 4)] for s in rep["top_scores"][1:4]],
            "o25": round(rep["over25"], 3), "btts": round(rep["btts"], 3),
            "played": f.played, "hg": f.hg, "ag": f.ag,
        })
    json.dump({"matches": ms}, open(os.path.join(OUT, "matches.json"), "w"),
              separators=(",", ":"))
    print(f"  → matches.json ({len(ms)} matches)")

    if os.environ.get("SKIP_BACKTEST") != "1":
        print("Running the walk-forward backtest…")
        bt = backtest.run(ds)
        bt["calibration_priors"] = cal
        json.dump(bt, open(os.path.join(OUT, "backtest.json"), "w"), indent=1)
        m = bt["model"]
        print(f"  → backtest.json  log-loss {m['log_loss']:.4f} "
              f"rps {m['rps']:.4f} acc {m['accuracy'] * 100:.1f}% over {m['n']} matches")
    validate()


def validate() -> None:
    """Refuse to ship a broken forecast."""
    fc = json.load(open(os.path.join(OUT, "forecast.json")))
    ms = json.load(open(os.path.join(OUT, "matches.json")))["matches"]
    assert len(fc["teams"]) == config.N_TEAMS, "wrong team count"
    assert len(ms) == config.N_MATCHES, "wrong match count"
    for t in fc["teams"]:
        s = sum(t["pos"])
        assert abs(s - 1) < 1e-3, f"{t['id']} position distribution sums to {s}"
    for k, want in (("title", 1), ("ucl", config.UCL_PLACES),
                    ("releg", config.RELEGATION_PLACES)):
        s = sum(t[k] for t in fc["teams"])
        assert abs(s - want) < 0.02, f"{k} probabilities sum to {s}, expected {want}"
    for m in ms:
        s = m["ph"] + m["pd"] + m["pa"]
        assert abs(s - 1) < 1e-3, f"{m['h']}-{m['a']} outcome probabilities sum to {s}"
    print("Validation passed.")


if __name__ == "__main__":
    main()
