/* Season re-simulation, in the reader's browser.

   The pipeline's simulator is NumPy over 50,000 seasons; this is the same
   arithmetic over 5,000, on one thread, and it has to finish while somebody is
   looking at it. Three things buy that: every fixture's scoreline distribution
   is turned into a flattened cumulative table ONCE, before the first season is
   played; the tables live in one Float32Array rather than 380 small ones; and
   the per-season state is three reused typed arrays instead of fresh objects.

   Message in:
     { fixtures, rho, basePts, baseGf, baseGa, picks, nSims, seed, runId }
   where `fixtures` are the unplayed matches as
     { h, a, hi, ai, lh, la }                    (hi/ai are team indices)
   and `picks` maps "home|away" to either { res: "H" | "D" | "A" } or
   { hg, ag }. An exact score is simply fixed. A result-only pick is sampled
   from the same Dixon-Coles grid restricted to that outcome and renormalised,
   which is the honest reading of "Arsenal win": not a made-up 1-0, but the
   distribution of the scorelines in which Arsenal win.

   Messages out: { type: 'progress', done, total } every 1,000 seasons, then
   { type: 'result', teams: [...] }. Both echo `runId` so the page can drop a
   stale run.                                                                */

const MAXG = 10;                 // model/config.py MAX_GOALS; P(>10) is ~1e-6
const NG = MAXG + 1;
const NC = NG * NG;              // cells in one flattened score grid

/* Deterministic and seeded on purpose. The baseline table and the edited one
   are drawn with the same stream, so the difference between them is the picks
   and not the luck of two different draws. */
function rng32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const LGAM = new Float64Array(NG);
for (let k = 1; k < NG; k++) LGAM[k] = LGAM[k - 1] + Math.log(k);

/* Dixon-Coles low-score correction, ported from ratings.tau. Without it the
   model under-predicts 0-0 and 1-1, the two commonest scores in the league. */
function tau(h, a, lh, la, rho) {
  if (h === 0 && a === 0) return Math.max(1 - lh * la * rho, 1e-9);
  if (h === 0 && a === 1) return Math.max(1 + lh * rho, 1e-9);
  if (h === 1 && a === 0) return Math.max(1 + la * rho, 1e-9);
  if (h === 1 && a === 1) return Math.max(1 - rho, 1e-9);
  return 1;
}

/* P(home = h, away = a) over the truncated grid, flattened row-major. */
function scoreGrid(lh, la, rho) {
  const ph = new Float64Array(NG);
  const pa = new Float64Array(NG);
  const logh = Math.log(lh), loga = Math.log(la);
  for (let k = 0; k < NG; k++) {
    ph[k] = Math.exp(-lh + k * logh - LGAM[k]);
    pa[k] = Math.exp(-la + k * loga - LGAM[k]);
  }
  const p = new Float64Array(NC);
  let s = 0;
  for (let h = 0; h < NG; h++) {
    for (let a = 0; a < NG; a++) {
      const v = ph[h] * pa[a] * tau(h, a, lh, la, rho);
      p[h * NG + a] = v;
      s += v;
    }
  }
  for (let i = 0; i < NC; i++) p[i] /= s;
  return p;
}

const keeps = (res, h, a) =>
  res < 0 || (res === 0 ? h > a : res === 1 ? h === a : h < a);

/* Cumulative table for one fixture written straight into the shared buffer.
   `res` is -1 for the unrestricted grid, or 0/1/2 for a home win, draw or away
   win, in which case the other cells are dropped and the rest renormalised. */
function writeCdf(p, res, out, off) {
  let s = 0;
  for (let i = 0; i < NC; i++) {
    if (keeps(res, (i / NG) | 0, i % NG)) s += p[i];
  }
  if (!(s > 0)) return writeCdf(p, -1, out, off);   // restriction is impossible
  let c = 0;
  for (let i = 0; i < NC; i++) {
    if (keeps(res, (i / NG) | 0, i % NG)) c += p[i] / s;
    out[off + i] = c;
  }
  out[off + NC - 1] = 1;              // float32 rounding must not lose the tail
}

const RES = { H: 0, D: 1, A: 2 };

self.onmessage = function (ev) {
  const msg = ev.data || {};
  const clock = (self.performance && self.performance.now) ? self.performance : Date;
  const t0 = clock.now();

  const runId = msg.runId || 0;
  const nSims = msg.nSims || 5000;
  const rho = msg.rho || 0;
  const picks = msg.picks || {};
  const basePts = Int32Array.from(msg.basePts);
  const baseGf = Int32Array.from(msg.baseGf);
  const baseGa = Int32Array.from(msg.baseGa);
  const n = basePts.length;
  const TOP = msg.uclPlaces || 5;
  const DOWN = msg.relegationPlaces || 3;

  const rem = (msg.fixtures || []).filter(function (f) { return !f.played; });
  const nR = rem.length;

  // ---- one pass of setup, then no allocation inside the season loop ----
  const hIdx = new Int32Array(nR);
  const aIdx = new Int32Array(nR);
  const fixH = new Int8Array(nR);
  const fixA = new Int8Array(nR);
  const isFix = new Uint8Array(nR);
  const cdf = new Float32Array(nR * NC);

  for (let k = 0; k < nR; k++) {
    const f = rem[k];
    hIdx[k] = f.hi;
    aIdx[k] = f.ai;
    const p = picks[f.h + '|' + f.a];
    if (p && p.hg != null && p.ag != null) {
      isFix[k] = 1;
      fixH[k] = Math.max(0, Math.min(MAXG, p.hg | 0));
      fixA[k] = Math.max(0, Math.min(MAXG, p.ag | 0));
      continue;
    }
    const res = (p && p.res in RES) ? RES[p.res] : -1;
    writeCdf(scoreGrid(f.lh, f.la, rho), res, cdf, k * NC);
  }

  const pts = new Int32Array(n);
  const gf = new Int32Array(n);
  const ga = new Int32Array(n);
  const rnd = new Float64Array(n);
  const order = new Array(n);
  const sumPts = new Float64Array(n);
  const sumPos = new Float64Array(n);
  const cTitle = new Int32Array(n);
  const cTop = new Int32Array(n);
  const cDown = new Int32Array(n);
  const rand = rng32(msg.seed || 537);

  /* Premier League order: points, then goal difference, then goals scored.
     A genuine tie is settled by a play-off, so it is broken at random here. */
  const cmp = function (x, y) {
    return (pts[y] - pts[x])
      || ((gf[y] - ga[y]) - (gf[x] - ga[x]))
      || (gf[y] - gf[x])
      || (rnd[y] - rnd[x]);
  };

  for (let s = 0; s < nSims; s++) {
    pts.set(basePts); gf.set(baseGf); ga.set(baseGa);

    for (let k = 0; k < nR; k++) {
      let h, a;
      if (isFix[k]) {
        h = fixH[k]; a = fixA[k];
      } else {
        const u = rand();
        const base = k * NC;
        let lo = 0, hi = NC - 1;
        while (lo < hi) {                       // first cell whose CDF covers u
          const mid = (lo + hi) >> 1;
          if (u <= cdf[base + mid]) hi = mid; else lo = mid + 1;
        }
        h = (lo / NG) | 0;
        a = lo - h * NG;
      }
      const i = hIdx[k], j = aIdx[k];
      gf[i] += h; ga[i] += a;
      gf[j] += a; ga[j] += h;
      if (h > a) pts[i] += 3;
      else if (h === a) { pts[i] += 1; pts[j] += 1; }
      else pts[j] += 3;
    }

    for (let i = 0; i < n; i++) { rnd[i] = rand(); order[i] = i; }
    order.sort(cmp);
    for (let r = 0; r < n; r++) {
      const t = order[r];
      sumPts[t] += pts[t];
      sumPos[t] += r + 1;
      if (r === 0) cTitle[t]++;
      if (r < TOP) cTop[t]++;
      if (r >= n - DOWN) cDown[t]++;
    }

    if ((s + 1) % 1000 === 0 && s + 1 < nSims) {
      self.postMessage({ type: 'progress', runId: runId, done: s + 1, total: nSims });
    }
  }

  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = {
      title: cTitle[i] / nSims,
      top5: cTop[i] / nSims,
      releg: cDown[i] / nSims,
      meanPts: sumPts[i] / nSims,
      meanPos: sumPos[i] / nSims,
    };
  }
  self.postMessage({
    type: 'result', runId: runId, nSims: nSims, remaining: nR,
    ms: clock.now() - t0, teams: out,
  });
};
