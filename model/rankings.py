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
import statistics

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


def _recentre(fit, pool, club_league, played, last, ref):
    """Where the scale sits: an average club of the big-five top flights.

    Returned rather than inlined because two things need it and they must agree
    exactly. `build` quotes today's SPI against it; `trajectory` quotes each
    past season's against the same definition applied to that season, which is
    what lets a line be read across clubs and across years at once.
    """
    big5 = {lg.slug for lg in leagues.BIG_FIVE}
    scale_set = [t for t in pool if club_league.get(t) in big5
                 and played.get(t, 0) >= MIN_MATCHES
                 and (ref - last[t]).days <= 400]
    if len(scale_set) < 40:                       # corpus too thin; use everything
        scale_set = list(pool)
    idx = [fit.index[t] for t in scale_set]
    return (float(np.mean(fit.att[idx])), float(np.mean(fit.dfn[idx])), scale_set)


def _spi_at(fit, t, a_bar, d_bar, home, adj: float = 0.0) -> float:
    off = float(np.exp(fit.mu + a_bar - d_bar + (fit.att[fit.index[t]] - a_bar) + adj))
    dfn = float(np.exp(fit.mu + a_bar - d_bar - (fit.dfn[fit.index[t]] - d_bar)))
    return spi_from(off, dfn, home, fit.rho)


#: The first season the pooled corpus can produce a rating anyone should read.
#: `openfootball/champions-league` starts at 2011-12 and those matches are the
#: only edges joining one league to another, so before it the corpus is five or
#: fifty disconnected leagues and a number quoted "against an average big-five
#: club" would be quoting against nothing. Probed, not assumed: every season
#: from 2000-01 to 2010-11 returns 404 upstream.
FIRST_BRIDGED_SEASON = "2012-13"


def trajectory(corpus: europe.Corpus, ref_date: dt.date | None = None,
               *, quiet: bool = True) -> dict[str, list]:
    """Every club's SPI at the start of every season, on one scale.

    The club pages used to draw this from each league's own fit, which made the
    line mean "how strong in this division" and put a gap in it for any season
    the club spent somewhere else. Two clubs from different leagues drawn on one
    axis then said whatever their divisions' averages happened to say: Sporting
    at 89 above Barcelona at 80, which is how this was noticed.

    So each point is a walk-forward pooled fit on everything played before that
    season began, recentred on that season's big-five average. A club that went
    down a division stays on the line, because the division below is in the
    corpus too, and it drops rather than disappearing -- which is the whole
    point of a trajectory.

    Costs about fifteen fits, three seconds each at full size, once per build.
    """
    # One point per year, not per season label. The corpus holds both spellings
    # -- a winter league writes 2025-26 and a summer league writes 2025 -- and
    # both cut at the same 1 July, so iterating labels drew every point twice
    # with identical values. Years, labelled in the winter convention the chart
    # already renders; each point is the fit as it stood that July.
    years = sorted({int(m.season[:4]) for m in corpus.matches if m.season[:4].isdigit()})
    first = int(FIRST_BRIDGED_SEASON[:4])
    out: dict[str, list] = {}
    # The newest point is fitted at exactly the date `build` uses, so the end of
    # a club's line is the SPI its page quotes rather than a number half a month
    # away from it. A chart whose last point disagrees with the headline above it
    # is the same class of bug as the one this whole change is about.
    today = ref_date or max(dt.date.today(), dt.date(2026, 8, 1))
    live = max(y for y in years) if years else None
    for year in [y for y in years if y >= first]:
        label = f"{year}-{(year + 1) % 100:02d}"
        ref = today if year == live else dt.date(year, 7, 1)
        past = corpus.before(ref)
        if len(past) < 2000:
            continue
        pool = sorted({m.home for m in past} | {m.away for m in past})
        played: dict[str, int] = {}
        domestic: dict[str, int] = {}
        last: dict[str, dt.date] = {}
        for m in past:
            league_match = m.comp not in europe.EURO_COMPS
            for t in (m.home, m.away):
                played[t] = played.get(t, 0) + 1
                if league_match:
                    domestic[t] = domestic.get(t, 0) + 1
                if t not in last or m.date > last[t]:
                    last[t] = m.date
        fit = ratings.fit_pooled(past, pool, ref, group_of=corpus.group_of,
                                 club_league=corpus.club_leagues(),
                                 default_group=europe.EUROPE)
        a_bar, d_bar, _ = _recentre(fit, pool, corpus.club_leagues(),
                                    played, last, ref)
        home = fit.home_advantage(europe.EUROPE)
        n = 0
        for t in pool:
            # The same floor the ranking uses. A club with four matches in the
            # corpus has a rating the fit invented from its ridge, and a line
            # that starts on one is a line that starts on a prior.
            # Domestic matches, not any matches. A club whose only appearances
            # that year were a Champions League group stage has been seen twelve
            # times against six opponents, and a rating off that lands wherever
            # those twelve fell: it put FC Barcelona at 52 in 2013-14, between
            # two seasons in the high eighties. A line should start when the
            # corpus can see the club's league, and not before.
            if (domestic.get(t, 0) < MIN_MATCHES
                    or (ref - last[t]).days > MAX_STALE_DAYS):
                continue
            out.setdefault(t, []).append(
                {"season": label, "spi": round(_spi_at(fit, t, a_bar, d_bar, home), 1)})
            n += 1
        if not quiet:
            print(f"    {label}: {n} clubs")
    return out


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
    a_bar, d_bar, scale_set = _recentre(fit, pool, club_league, played, last, ref)
    home = fit.home_advantage(europe.EUROPE)

    # Goals an average scale-set club scores, and concedes, against another of
    # them: `exp(mu + a_bar - d_bar)` by construction, and the same number for
    # both because the scale set is its own average opponent.
    ref_goals = float(np.exp(fit.mu + a_bar - d_bar))

    # The top quarter of every competition in every season, for the big-game
    # measure, and the raw profile of each club.
    prof = {t: _profile(ms, t, fit) for t, ms in recent.items()}
    # Each of these three is quoted against the club's *own* competition, not
    # against Europe as a whole -- unlike attack and defence, which are on the
    # big-five scale because goals are goals wherever they are scored.
    #
    # "Points against the top quarter of the division" has no cross-border
    # meaning: a club that dominates a weak league beats a weak top quarter, and
    # scored against a European average it comes out looking like the best
    # big-game side in Europe. The first attempt did exactly that and rated
    # Galatasaray 95 and Manchester City 88. Home advantage and consistency vary
    # by country for the same sort of reason. Measured against its own division
    # each says what it should, and matches the number the club's own forecast
    # page already shows.
    # One reference per measure, over every club in the corpus that has it.
    #
    # It used to be one reference per competition, which made a rating mean
    # "compared with your own division" and made the comparison page a liar: two
    # clubs from different leagues drew one shape out of axes that were not the
    # same axes. The three measures here are quantities a club carries with it
    # -- points gained at home over away, how much its goal difference moves,
    # how far it beats expectation against the best sides it meets -- and none
    # of them needs a division to be meaningful.
    prefs: dict[str, float] = {}
    for key in ("gd_sd",):
        vals = [r[key] for r in prof.values() if r.get(key) is not None]
        if vals:
            prefs[key] = sum(vals) / len(vals)

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
            # One standard deviation of the rating either way, in the units the
            # rating is quoted in. The league forecasts published this and the
            # ranking did not, so moving SPI here would have lost it.
            "spi_lo": round(_spi_at(fit, t, a_bar, d_bar, home,
                                    -config.RATING_SD), 1),
            "spi_hi": round(_spi_at(fit, t, a_bar, d_bar, home,
                                    config.RATING_SD), 1),
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
        # One more rating on the same 35-95 scale as attack and defence, so the
        # comparison page can draw a shape for any club in Europe rather than
        # only for the ones this site forecasts.
        p = prof.get(t) or {}
        for field, dim, key in (("consistency_r", "consistency", "gd_sd"),):
            got = scale.dimension(dim, p.get(key), prefs.get(key), europe=True)
            if got is not None:
                rows[-1][field] = got
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


#: How many recent seasons the profile ratings are read over in the pooled
#: corpus. The domestic feeds outside the big five are shallower and several
#: are stale, so this is the window rather than a season count.
PROFILE_MATCHES = 90


def _profile(matches: list, club: str, fit) -> dict:
    """Consistency, for one club, from the pooled corpus.

    Computed here rather than on a league forecast so that every club in the
    ranking has it and not only the 174 this site forecasts. A comparison page
    that offers 836 clubs and can rate 174 of them is offering something it
    mostly cannot do.

    This used to return three measures. Two of them are gone, and the reason is
    worth keeping: `tools/measure_scale.py` splits the spread of each measure
    into what clubs genuinely differ by and what is the luck of the matches we
    happened to see, and *home advantage* came back at 7% real, *big games* at
    4%. Both had a decent-looking spread across clubs and almost none of it
    survived. They were published as ratings out of 100 for a while, which means
    this site drew two axes of a radar out of noise. See `model/scale.py`.

    Consistency came back at 65%, which is why it is still here.
    """
    ms = sorted(matches, key=lambda m: m.date)[-PROFILE_MATCHES:]
    if len(ms) < 30:
        return {}
    hp, ap, gd = [], [], []
    for m in ms:
        home = m.home == club
        gf, ga = (m.hg, m.ag) if home else (m.ag, m.hg)
        if gf is None or ga is None:
            continue
        (hp if home else ap).append(1)
        gd.append(gf - ga)
    if len(hp) < 10 or len(ap) < 10 or len(gd) < 30:
        return {}
    return {"gd_sd": round(statistics.pstdev(gd), 3)}


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
