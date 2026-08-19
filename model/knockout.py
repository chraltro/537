"""Two-legged ties, extra time, shootouts, and the Champions League bracket.

A new file rather than an addition to `simulate.py`, because none of this is
season simulation: a knockout tie is a small exact calculation, and the bracket
on top of it is a bookkeeping problem.

Three rules, all from docs/european-competitions-plan.md 2 and 3.1:

* **No away goals.** Abolished across UEFA club competitions from 2021-22. A tie
  is decided on aggregate, full stop.
* **Extra time is a different game.** Thirty minutes, so lambda scales by 30/90,
  and it is measurably lower-scoring than open play, so by a further 0.85. That
  0.85 is a judgement call and is flagged as one on the method page.
* **A shootout is a coin flip.** Twenty-four observed shootouts in the corpus
  carry no credible skill signal, so P(home) = 0.5 and nothing is pretended.

Ties are computed exactly rather than sampled. The aggregate score difference of
two independent legs is the convolution of each leg's difference distribution,
which is a hundred-element numpy operation -- far cheaper and far less noisy
than simulating two matches fifty thousand times.
"""
from __future__ import annotations

import numpy as np

from . import config
from .ratings import Fit
from .simulate import score_matrix

#: Extra time: half an hour instead of ninety minutes, and quieter than open
#: play. The 30/90 is arithmetic; the 0.85 is a judgement call.
ET_FRACTION = 30.0 / 90.0
ET_TEMPO = 0.85


def _diff_dist(m: np.ndarray) -> np.ndarray:
    """P(home goals - away goals = d) for d in -maxg..maxg, as one vector."""
    n = m.shape[0]
    out = np.zeros(2 * n - 1)
    for d in range(-(n - 1), n):
        out[d + n - 1] = np.trace(m, offset=-d)
    return out


def _extra_time(lh: float, la: float, rho: float) -> tuple[float, float, float]:
    """P(home wins / draw / away wins) over thirty minutes of extra time."""
    k = ET_FRACTION * ET_TEMPO
    m = score_matrix(lh * k, la * k, rho)
    return (float(np.tril(m, -1).sum()), float(np.trace(m)),
            float(np.triu(m, 1).sum()))


def one_off(fit: Fit, home: str, away: str, *, neutral: bool = False,
            group: str | None = None, adj: dict[str, float] | None = None,
            extra_time: bool = True) -> float:
    """P(`home` advances) from a single match that must produce a winner.

    The final, in other words: ninety minutes on neutral ground, then extra
    time, then a coin flip.
    """
    lh, la = _lambdas(fit, home, away, adj, group, neutral=neutral)
    m = score_matrix(lh, la, fit.rho)
    w = float(np.tril(m, -1).sum())
    d = float(np.trace(m))
    if not extra_time:
        return w + d / 2
    ew, ed, _ = _extra_time(lh, la, fit.rho)
    return w + d * (ew + ed * 0.5)


def _lambdas(fit: Fit, home: str, away: str, adj, group, neutral: bool = False):
    lh, la = fit.lambdas(home, away, neutral=neutral, group=group)
    if adj:
        h = adj.get(home, 0.0) / 2
        a = adj.get(away, 0.0) / 2
        lh *= float(np.exp(h - a))
        la *= float(np.exp(a - h))
    return float(lh), float(la)


def two_legged(fit: Fit, first_away: str, first_home: str, *,
               group: str | None = None,
               adj: dict[str, float] | None = None) -> float:
    """P(`first_away` advances) from a two-legged tie.

    `first_away` is the club that plays the FIRST leg away and therefore the
    second leg at home -- the higher seed, under every UEFA bracket rule since
    the format changed. Legs are independent draws from the exact Dixon-Coles
    grid; the aggregate is their sum; a level aggregate goes to thirty minutes
    of extra time in the second leg's stadium, then to penalties.
    """
    a = first_away
    lh1, la1 = _lambdas(fit, first_home, a, adj, group)      # leg 1 at b's ground
    lh2, la2 = _lambdas(fit, a, first_home, adj, group)      # leg 2 at a's ground
    m1 = score_matrix(lh1, la1, fit.rho)
    m2 = score_matrix(lh2, la2, fit.rho)
    # d1 is (a - b) in leg one, which is the negative of leg one's home margin.
    d1 = _diff_dist(m1)[::-1]
    d2 = _diff_dist(m2)
    agg = np.convolve(d1, d2)
    mid = (len(agg) - 1) // 2
    p_win = float(agg[mid + 1:].sum())
    p_level = float(agg[mid])
    ew, ed, _ = _extra_time(lh2, la2, fit.rho)
    return p_win + p_level * (ew + ed * 0.5)


def tie_matrix(fit: Fit, teams: list[str], *, group: str | None = None,
               adj: dict[str, float] | None = None) -> np.ndarray:
    """`M[i, j]` = P(club i advances past club j) with i home in the second leg.

    Computed once for all 36 clubs and then indexed; the bracket simulation
    below is fifty thousand table lookups rather than fifty thousand fits.
    """
    n = len(teams)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            out[i, j] = two_legged(fit, teams[i], teams[j], group=group, adj=adj)
    return out


def neutral_matrix(fit: Fit, teams: list[str], *, group: str | None = None,
                   adj: dict[str, float] | None = None) -> np.ndarray:
    """`M[i, j]` = P(i beats j in a one-off match on neutral ground)."""
    n = len(teams)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = one_off(fit, teams[i], teams[j], neutral=True, group=group, adj=adj)
            out[i, j], out[j, i] = p, 1 - p
    return out


# --------------------------------------------------------------------------
# The bracket
# --------------------------------------------------------------------------
#: Knockout play-off pairings, by league-phase finishing position (1-based):
#: 9v24, 10v23 ... 16v17, higher seed at home in the second leg.
PLAYOFF_TIES = tuple((9 + k, 24 - k) for k in range(8))

#: Which play-off ties each pair of direct qualifiers can meet in the round of
#: 16. UEFA fixes the shape and draws only the two ways round inside each group:
#: seeds 1 and 2 get the winners of the two ties between the lowest-ranked
#: play-off clubs, and so on down. Indices are into PLAYOFF_TIES.
R16_GROUPS = (
    ((1, 2), (6, 7)),      # seeds 1,2  <- winners of 15v18 and 16v17
    ((3, 4), (4, 5)),      # seeds 3,4  <- winners of 13v20 and 14v19
    ((5, 6), (2, 3)),      # seeds 5,6  <- winners of 11v22 and 12v21
    ((7, 8), (0, 1)),      # seeds 7,8  <- winners of 9v24  and 10v23
)

#: Quarter-final bracket, by seed group. Standard seeded shape: the two top
#: seeds sit in opposite halves and can only meet in the final.
#: Each entry names the two R16 slots that feed one quarter-final; slots are
#: indexed in the order the groups above produce them (seed 1, 2, 3, ... 8).
QF_PAIRS = ((0, 7), (3, 4), (2, 5), (1, 6))


def simulate_bracket(fit: Fit, teams: list[str], orders: np.ndarray, *,
                     group: str | None = None,
                     adj: dict[str, float] | None = None,
                     seed: int = config.SEED,
                     max_sims: int = 20000) -> dict:
    """Round-by-round probabilities, given simulated league-phase finishes.

    `orders[s, r]` is the index of the club that finished r-th in simulated
    season s -- so the bracket is redrawn on top of each simulated table, and a
    club's chance of winning the trophy already contains its chance of finishing
    high enough to get an easy half.
    """
    n = len(teams)
    rng = np.random.default_rng(seed + 1)
    if len(orders) > max_sims:
        orders = orders[rng.choice(len(orders), max_sims, replace=False)]
    S = len(orders)
    tie = tie_matrix(fit, teams, group=group, adj=adj)
    neu = neutral_matrix(fit, teams, group=group, adj=adj)

    def resolve(high: np.ndarray, low: np.ndarray, table: np.ndarray) -> np.ndarray:
        """Play `high` (home in the second leg) against `low`, vectorised."""
        p = table[high, low]
        hit = rng.random(S) < p
        return np.where(hit, high, low)

    reach = {k: np.zeros(n) for k in ("r16", "qf", "sf", "final", "win")}
    # Play-off round: the eight ties are strictly seeded.
    po_winners = []
    for hi_pos, lo_pos in PLAYOFF_TIES:
        hi = orders[:, hi_pos - 1]
        lo = orders[:, lo_pos - 1]
        po_winners.append(resolve(hi, lo, tie))
    po = np.stack(po_winners, axis=1)                       # (S, 8)

    # Round of 16. Direct qualifiers 1-8 meet play-off winners; inside each
    # group of two seeds the pairing is a coin toss, which is the only part of
    # the R16 draw that is actually random.
    slots: list[tuple[np.ndarray, np.ndarray]] = [None] * 8   # type: ignore[list-item]
    for seeds, ties in R16_GROUPS:
        swap = rng.random(S) < 0.5
        a_tie = np.where(swap, po[:, ties[0]], po[:, ties[1]])
        b_tie = np.where(swap, po[:, ties[1]], po[:, ties[0]])
        slots[seeds[0] - 1] = (orders[:, seeds[0] - 1], a_tie)
        slots[seeds[1] - 1] = (orders[:, seeds[1] - 1], b_tie)

    for k in range(8):
        hi, lo = slots[k]
        for arr in (hi, lo):
            np.add.at(reach["r16"], arr, 1)
    r16_winners = np.stack([resolve(hi, lo, tie) for hi, lo in slots], axis=1)

    # From here the bracket is fixed and the higher league-phase finisher is at
    # home in the second leg, which `rank` decides.
    rank = np.empty((S, n), dtype=np.int64)
    np.put_along_axis(rank, orders, np.arange(n)[None, :].repeat(S, 0), axis=1)

    def play_round(entrants: np.ndarray, pairs) -> np.ndarray:
        outs = []
        for i, j in pairs:
            a, b = entrants[:, i], entrants[:, j]
            a_high = rank[np.arange(S), a] < rank[np.arange(S), b]
            hi = np.where(a_high, a, b)
            lo = np.where(a_high, b, a)
            outs.append(resolve(hi, lo, tie))
        return np.stack(outs, axis=1)

    for k in range(8):
        np.add.at(reach["qf"], r16_winners[:, k], 1)
    qf_winners = play_round(r16_winners, QF_PAIRS)
    for k in range(4):
        np.add.at(reach["sf"], qf_winners[:, k], 1)
    sf_winners = play_round(qf_winners, ((0, 1), (2, 3)))
    for k in range(2):
        np.add.at(reach["final"], sf_winners[:, k], 1)

    # The final is one match on neutral ground.
    a, b = sf_winners[:, 0], sf_winners[:, 1]
    champs = np.where(rng.random(S) < neu[a, b], a, b)
    np.add.at(reach["win"], champs, 1)

    return {"teams": teams, "n_sims": int(S),
            **{k: v / S for k, v in reach.items()}}
