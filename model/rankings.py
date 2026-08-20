"""One rating for every club in Europe, on one scale.

This is the thing a league table cannot do and the thing FiveThirtyEight's
Global Club Soccer Rankings existed to do: put Real Madrid, Brentford and
Bodø/Glimt in one ordering and mean it.

The mechanism is already in the repository and is not new maths. `model.europe`
assembles a corpus of roughly fifty-four thousand matches -- fifteen seasons of
UEFA competition, forty-six non-big-five domestic leagues and the big five --
and `ratings.fit_pooled` fits one Dixon-Coles model over all of it with a
per-competition home term and a ridge that shrinks each club toward its own
league's mean rather than toward the global one. The European matches are the
only edges joining one league to another, and they are what identify the league
offsets. The Champions League build has run exactly this fit every cycle since
the cup landed.

Two things this module is careful about.

*It does not touch the league forecasts.* `run.POOLED_DOMESTIC` stays off. SPI
is defined against an average opponent, and a pooled fit redefines "average" as
the average of nine hundred clubs in fifty-two leagues; publishing that number
in place of the Premier League one would move every figure on the site for no
new evidence. So the pooled rating is written to its own file, on its own
stated scale, and the five domestic forecasts are built exactly as before.

*It says how old each rating is.* Twenty-nine of the associations in the corpus
have no domestic feed newer than May 2025. `ratings.staleness_sd` already turns
that into widened uncertainty for the cup simulation; here the same staleness is
published per club as a months-since-last-match figure, so a rating nobody has
tested in a year is visibly that rather than quietly wrong.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from . import config, europe, leagues, ratings, scale

#: SPI means "expected share of points against an average team", so the scale
#: needs an average worth naming. The pooled corpus average is nine hundred
#: clubs from San Marino upwards, against which every club anyone has heard of
#: scores above ninety and the number stops saying anything. So the scale is
#: re-centred on the big five specifically -- not on "every league this site
#: forecasts", which would move the meaning of 50 every time a competition was
#: added. 50 is an average Premier League / La Liga / Serie A / Bundesliga /
#: Ligue 1 club, and the page says so in those words.
SCALE_NOTE = ("100 means taking every point against an average club of the big "
              "five leagues; 50 means an even split.")

#: A club needs a real recent record before it belongs in a ranking. Below this
#: the rating is mostly its league's mean wearing a club's name.
MIN_MATCHES = 12
#: How stale a club's newest result may be and still be ranked. Two years is
#: generous, and it is published per club rather than hidden.
MAX_STALE_DAYS = 900


#: Corpus group id -> the league slug that now forecasts those clubs. The
#: corpus labels Dutch clubs `dom-ned` because that is the feed they came from;
#: the site forecasts them as `eredivisie`, and without this mapping a club
#: would rank on this page with no page of its own to link to.
#: Competition-group ids in the pooled corpus that are the *same competition* as
#: one this site forecasts, and so should carry its slug -- which is what makes a
#: club on the global ranking a link to its own page, and what lets the
#: comparison page find its season-by-season line.
#:
#: This is a hand-kept map and it silently rotted the moment Belgium was added
#: as a forecast competition: sixteen Belgian clubs sat in the ranking with no
#: link and no trajectory, and the comparison page told the reader this site
#: does not forecast the Belgian Pro League, which it plainly does.
#: `tests/test_leagues.py` now fails if a forecast competition appears in the
#: pooled corpus without a mapping, so the next one cannot rot the same way.
GROUP_SLUG = {"dom-ned": "eredivisie", "dom-por": "primeira-liga",
              "dom-bel": "pro-league", "premier-league-2": "championship"}


def _league_names() -> dict[str, dict]:
    """Human names for the competition-group ids the corpus tags clubs with."""
    out: dict[str, dict] = {}
    for lg in leagues.LEAGUES:
        out[lg.slug] = {"name": lg.name, "country": lg.country, "slug": lg.slug}
        out[f"{lg.slug}-2"] = {"name": f"{lg.country}, second tier",
                               "country": lg.country, "slug": None}
    for src in europe.DOMESTIC:
        out[src.group] = {"name": src.name, "country": src.assoc, "slug": None}
    for grp, slug in GROUP_SLUG.items():
        lg = leagues.BY_SLUG.get(slug)
        if lg:
            out[grp] = {"name": lg.name, "country": lg.country, "slug": lg.slug}
    out[europe.EUROPE] = {"name": "European competition", "country": "", "slug": None}
    out["other"] = {"name": "No domestic feed", "country": "", "slug": None}
    return out


def spi_from(off: float, dfn: float, home: float, rho: float) -> float:
    """Expected share of points against an average club, home and away.

    Deliberately the same definition and the same arithmetic as `run.spi`, taken
    from offence and defence rather than from a `Fit`, so the published `off`
    and `def` and the published `spi` can never disagree.
    """
    from .simulate import outcome_probs, score_matrix
    pts = 0.0
    for at_home in (True, False):
        lh = off * (np.exp(home) if at_home else 1.0)
        la = dfn * (1.0 if at_home else np.exp(home))
        w, d, _ = outcome_probs(score_matrix(float(lh), float(la), rho))
        pts += 3 * w + d
    return float(pts / 6.0 * 100.0)


def build(corpus: europe.Corpus, ref_date: dt.date | None = None, *,
          featured: set[str] | None = None,
          quiet: bool = True) -> dict:
    """Fit the pooled model and turn it into the global ranking payload."""
    ref = ref_date or max(dt.date.today(), dt.date(2026, 8, 1))
    hist = corpus.before(ref)
    pool = sorted({m.home for m in hist} | {m.away for m in hist})
    club_league = corpus.club_leagues()
    fit = ratings.fit_pooled(hist, pool, ref, group_of=corpus.group_of,
                             club_league=club_league,
                             default_group=europe.EUROPE)

    # Counts and recency, from the same match list the fit saw.
    played: dict[str, int] = {}
    last: dict[str, dt.date] = {}
    recent: dict[str, list] = {}
    for m in hist:
        for t in (m.home, m.away):
            played[t] = played.get(t, 0) + 1
            if t not in last or m.date > last[t]:
                last[t] = m.date
            recent.setdefault(t, []).append(m)

    # The scale: an average club of the five leagues we forecast. Those are the
    # clubs in the current fixture lists, which the caller passes as `featured`;
    # with none passed, fall back to every club the big-five groups contain.
    big5 = {lg.slug for lg in leagues.BIG_FIVE}
    scale_set = [t for t in pool if club_league.get(t) in big5
                 and played.get(t, 0) >= MIN_MATCHES
                 and (ref - last[t]).days <= 400]
    if len(scale_set) < 40:                       # corpus too thin; use everything
        scale_set = list(pool)
    idx = [fit.index[t] for t in scale_set]
    a_bar = float(np.mean(fit.att[idx]))
    d_bar = float(np.mean(fit.dfn[idx]))
    home = fit.home_advantage(europe.EUROPE)

    # Goals an average scale-set club scores, and concedes, against another of
    # them: `exp(mu + a_bar - d_bar)` by construction, and the same number for
    # both because the scale set is its own average opponent.
    ref_goals = float(np.exp(fit.mu + a_bar - d_bar))

    names = _league_names()
    meta = corpus.reg.meta
    rows = []
    for t in pool:
        if played.get(t, 0) < MIN_MATCHES:
            continue
        seen = last.get(t)
        stale = (ref - seen).days if seen else 10_000
        if stale > MAX_STALE_DAYS:
            continue
        i = fit.index[t]
        # Re-centred on the scale set: every match probability is unchanged,
        # only the club the number is quoted against moves.
        off = float(np.exp(fit.mu + a_bar - d_bar + (fit.att[i] - a_bar)))
        dfn = float(np.exp(fit.mu + a_bar - d_bar - (fit.dfn[i] - d_bar)))
        grp = club_league.get(t, "other")
        lab = names.get(grp, {"name": grp, "country": "", "slug": None})
        m = meta.get(t, {})
        rows.append({
            "id": t,
            "name": m.get("name", t),
            "short": m.get("short", t[:3].upper()),
            "primary": m.get("primary", "#7A8290"),
            "league": lab["name"],
            "country": lab["country"],
            "slug": lab["slug"],
            "spi": round(spi_from(off, dfn, home, fit.rho), 1),
            "off": round(off, 2),
            "def": round(dfn, 2),
            # The same two as a rating out of 100, higher better both times.
            # Centred on the scale set -- an average big-five club -- which is
            # the reference SPI is already quoted against on this page, so all
            # three numbers in a row mean "against the same opponent". Most of
            # Europe therefore sits below 50, which is the point of a ranking
            # that spans San Marino and the Premier League.
            "att_r": scale.attack(off, ref_goals, scale.SD_EUROPE),
            "def_r": scale.defence(dfn, ref_goals, scale.SD_EUROPE),
            "n": played.get(t, 0),
            "last": seen.isoformat() if seen else None,
            "stale": round(stale / 30.44, 1),
            "featured": bool(featured and t in featured),
            "form": _form(recent.get(t, []), t),
        })
    rows.sort(key=lambda r: -r["spi"])
    for k, r in enumerate(rows, 1):
        r["rank"] = k
    # Rank within a club's own league as well: "third best club in Portugal" is
    # a different and often more interesting fact than "sixty-first in Europe".
    seen_league: dict[str, int] = {}
    for r in rows:
        seen_league[r["league"]] = seen_league.get(r["league"], 0) + 1
        r["league_rank"] = seen_league[r["league"]]

    return {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "asof": ref.isoformat(),
        "note": SCALE_NOTE,
        "scale_clubs": len(scale_set),
        "n_clubs": len(rows),
        "n_matches": len(hist),
        "n_leagues": len({r["league"] for r in rows}),
        # Enough of the fit for a browser to work out a neutral-ground match
        # between any two clubs from `off` and `def` alone: lambda_home =
        # off_h * def_a * exp(home - mu), and lambda_away the mirror of it.
        # That is arithmetic on published parameters, not a second model.
        "mu": round(float(fit.mu + a_bar - d_bar), 6),
        "home": round(float(home), 6),
        "rho": round(float(fit.rho), 6),
        "rating_sd": config.RATING_SD,
        "clubs": rows,
    }


#: How many recent matches a club's form is read over. Twenty is about half a
#: season: long enough that one thrashing does not define it, short enough that
#: it is describing this squad rather than the last three.
FORM_MATCHES = 20


def _form(matches: list, club: str) -> dict:
    """What actually happened to one club lately, as opposed to what it is rated.

    None of this is an input to any forecast -- the fit reads the same matches
    and reads them better, with time weighting and opponent strength. It is here
    because a rating tells you how good a club is and says nothing about what
    watching it has been like, and the two are different questions.
    """
    ms = sorted(matches, key=lambda m: m.date)[-FORM_MATCHES:]
    if not ms:
        return {}
    w = d = l = gf = ga = cs = fail = 0
    for m in ms:
        home = m.home == club
        f, a = (m.hg, m.ag) if home else (m.ag, m.hg)
        gf += f
        ga += a
        if a == 0:
            cs += 1
        if f == 0:
            fail += 1
        if f > a:
            w += 1
        elif f == a:
            d += 1
        else:
            l += 1
    n = len(ms)
    return {
        "n": n,
        "w": w, "d": d, "l": l,
        "ppg": round((3 * w + d) / n, 2),
        "gf_pm": round(gf / n, 2),
        "ga_pm": round(ga / n, 2),
        "clean_pct": round(cs / n, 3),
        "blank_pct": round(fail / n, 3),
        "from": ms[0].date.isoformat(),
        "to": ms[-1].date.isoformat(),
    }


def head_to_head(corpus: europe.Corpus, clubs: set[str]) -> dict:
    """Every meeting between two clubs in `clubs`, aggregated per pair.

    Bounded on purpose: the corpus has tens of thousands of distinct pairings
    and shipping all of them would be a megabyte of JSON to answer a question
    nobody asked. `clubs` is the set with a forecast page, which is exactly the
    set the comparison tool can offer.
    """
    pairs: dict[str, dict] = {}
    for m in corpus.matches:
        if m.home not in clubs or m.away not in clubs:
            continue
        if m.hg is None or m.ag is None:
            continue
        a, b = sorted((m.home, m.away))
        key = f"{a}|{b}"
        r = pairs.get(key)
        if r is None:
            r = pairs[key] = {"n": 0, "aw": 0, "d": 0, "bw": 0,
                              "agf": 0, "bgf": 0, "last": None, "lastr": None}
        # `a` is the alphabetically first club, not the home side.
        agf, bgf = (m.hg, m.ag) if m.home == a else (m.ag, m.hg)
        r["n"] += 1
        r["agf"] += agf
        r["bgf"] += bgf
        r["aw"] += agf > bgf
        r["d"] += agf == bgf
        r["bw"] += agf < bgf
        iso = m.date.isoformat()
        if r["last"] is None or iso > r["last"]:
            r["last"] = iso
            r["lastr"] = (f"{corpus.reg.display(m.home)} {m.hg}-{m.ag} "
                          f"{corpus.reg.display(m.away)}")
    return pairs
