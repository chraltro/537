/* Shared helpers: data loading, formatting, chrome, tooltip, small SVG charts. */

const BASE = location.pathname.replace(/[^/]*$/, '');
const store = {};

export async function data(name) {
  if (!store[name]) {
    const r = await fetch(`${BASE}data/${name}.json`, { cache: 'no-cache' });
    if (!r.ok) throw new Error(`${name}.json ${r.status}`);
    store[name] = await r.json();
  }
  return store[name];
}

export const pct = (x, d = 0) =>
  x >= 0.9995 ? '>99%' : (x > 0 && x < 0.005 ? '<1%' : `${(x * 100).toFixed(d)}%`);
export const signed = (x) => (x > 0 ? `+${x}` : `${x}`);
export const ord = (n) => {
  const s = ['th', 'st', 'nd', 'rd'];
  const v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
};

export function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/* ---------------- theme ---------------- */
export function initChrome(page) {
  const saved = localStorage.getItem('plf-theme');
  if (saved) document.documentElement.setAttribute('data-theme', saved);

  document.body.insertAdjacentHTML('afterbegin', `
    <a class="skip" href="#main">Skip to content</a>
    <header class="masthead"><div class="wrap">
      <a class="brand" href="${BASE}index.html">
        <span class="mark"></span>
        <b>Ninety</b><span>Premier League forecast</span>
      </a>
      <nav class="top">
        <a href="${BASE}index.html"${page === 'table' ? ' aria-current="page"' : ''}>Table</a>
        <a href="${BASE}matches.html"${page === 'matches' ? ' aria-current="page"' : ''}>Matches</a>
        <a href="${BASE}team.html"${page === 'team' ? ' aria-current="page"' : ''}>Clubs</a>
        <a href="${BASE}races.html"${page === 'races' ? ' aria-current="page"' : ''}>Races</a>
        <a href="${BASE}simulator.html"${page === 'sim' ? ' aria-current="page"' : ''}>What&nbsp;if</a>
        <a href="${BASE}method.html"${page === 'method' ? ' aria-current="page"' : ''}>Method</a>
        <button class="searchbtn" data-open-palette aria-label="Search (press slash)">
          <svg width="12" height="12" viewBox="0 0 16 16" aria-hidden="true">
            <circle cx="7" cy="7" r="4.6" fill="none" stroke="currentColor" stroke-width="1.7"/>
            <path d="M10.6 10.6 14 14" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
          </svg><kbd>/</kbd>
        </button>
        <button class="themetoggle" id="tt" title="Switch theme" aria-label="Switch colour theme">
          <svg width="15" height="15" viewBox="0 0 16 16" aria-hidden="true">
            <circle cx="8" cy="8" r="6.2" fill="none" stroke="currentColor" stroke-width="1.6"/>
            <path d="M8 1.8a6.2 6.2 0 0 0 0 12.4z" fill="currentColor"/>
          </svg>
        </button>
      </nav>
    </div></header>`);

  document.getElementById('tt').addEventListener('click', () => {
    const now = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', now);
    localStorage.setItem('plf-theme', now);
    window.dispatchEvent(new Event('themechange'));
  });

  document.body.insertAdjacentHTML('beforeend', `
    <footer><div class="wrap">
      <p>An open forecast of the Premier League, rebuilt in the spirit of FiveThirtyEight's
      Soccer Power Index. Ratings, match probabilities and season simulations are generated
      from public match data — no private feeds, no hand-tuned opinions.
      <a href="${BASE}method.html">How it works and how accurate it is →</a></p>
      <p>Match data: <a href="https://github.com/datasets/football-datasets">football-datasets</a>
      (mirroring football-data.co.uk) and <a href="https://github.com/openfootball/england">openfootball</a>.
      Not affiliated with the Premier League. Not betting advice.</p>
    </div></footer>`);
  document.body.insertAdjacentHTML('beforeend', '<div id="tip" role="tooltip"></div>');
}

/* ---------------- tooltip ---------------- */
let tipEl;
export function tip(html, ev) {
  tipEl = tipEl || document.getElementById('tip');
  if (!html) { tipEl.style.opacity = 0; return; }
  tipEl.innerHTML = html;
  tipEl.style.opacity = 1;
  const r = tipEl.getBoundingClientRect();
  let x = ev.clientX + 14, y = ev.clientY + 14;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - 14;
  if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - 14;
  tipEl.style.left = `${Math.max(8, x)}px`;
  tipEl.style.top = `${Math.max(8, y)}px`;
}
export function tipRows(title, rows) {
  return `<div class="t">${title}</div>` +
    rows.map(([k, v]) => `<div class="r"><span>${k}</span><b>${v}</b></div>`).join('');
}

/* ---------------- club colours ----------------
   A club's real colour is used wherever possible, but Fulham's black and
   Leeds' yellow are invisible against one surface or the other. Nudge any
   colour that cannot be seen toward the current ink until it can be, rather
   than hand-picking 20 approximations of the right colour.          */
const _hex = (h) => {
  const v = h.replace('#', '');
  const n = parseInt(v.length === 3 ? v.split('').map((c) => c + c).join('') : v, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};
const _lum = ([r, g, b]) => {
  const f = (c) => { c /= 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
};
const _ratio = (a, b) => {
  const [x, y] = [_lum(a), _lum(b)].sort((p, q) => q - p);
  return (x + 0.05) / (y + 0.05);
};
export function chipColor(hex) {
  const dark = document.documentElement.getAttribute('data-theme') !== 'light';
  const surface = dark ? [26, 26, 25] : [252, 252, 251];
  const target = dark ? [255, 255, 255] : [11, 11, 11];
  let c = _hex(hex);
  for (let i = 0; i < 12 && _ratio(c, surface) < 2.4; i++) {
    c = c.map((v, k) => Math.round(v + (target[k] - v) * 0.18));
  }
  return `rgb(${c.join(',')})`;
}

/* ---------------- sequential ramp ----------------
   One hue, light to dark, seven steps. Cells near zero recede into the
   surface so the eye lands on where a club actually finishes.        */
export function rampColor(p, max) {
  if (!(p > 0.002)) return 'var(--ramp-0)';             // genuinely never happens
  const t = Math.min(1, (p / (max || 1)) ** 0.55);      // compress, but keep the peak loud
  const step = Math.min(6, Math.max(1, Math.ceil(t * 6)));
  return `var(--ramp-${step})`;
}

/* ---------------- sortable tables ---------------- */
export function makeSortable(table, rows, render, initial) {
  let key = initial, dir = -1;
  const heads = [...table.querySelectorAll('th[data-k]')];
  const apply = () => {
    heads.forEach((h) => h.removeAttribute('data-dir'));
    const h = heads.find((x) => x.dataset.k === key);
    if (h) h.setAttribute('data-dir', dir === -1 ? 'desc' : 'asc');
    const sorted = [...rows].sort((a, b) => {
      const x = a[key], y = b[key];
      const c = typeof x === 'string' ? x.localeCompare(y) : x - y;
      return c * dir;
    });
    render(sorted);
  };
  heads.forEach((h) => {
    h.classList.add('sortable');
    h.addEventListener('click', () => {
      if (key === h.dataset.k) dir = -dir;
      else { key = h.dataset.k; dir = h.dataset.k === 'name' ? 1 : -1; }
      apply();
    });
  });
  apply();
}

/* ---------------- tiny SVG line chart ----------------
   Used for a club's rating trajectory. One series, so no legend box —
   the surrounding heading names it.                                   */
export function sparkline(values, { w = 260, h = 54, pad = 6 } = {}) {
  if (values.length < 2) return '';
  const lo = Math.min(...values), hi = Math.max(...values);
  const span = hi - lo || 1;
  const x = (i) => pad + (i * (w - pad * 2)) / (values.length - 1);
  const y = (v) => h - pad - ((v - lo) / span) * (h - pad * 2);
  const d = values.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(' ');
  const last = values.length - 1;
  return `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-hidden="true">
    <path d="${d}" fill="none" stroke="var(--accent)" stroke-width="2"
          stroke-linejoin="round" stroke-linecap="round"/>
    <circle cx="${x(last).toFixed(1)}" cy="${y(values[last]).toFixed(1)}" r="4"
            fill="var(--accent)" stroke="var(--surface)" stroke-width="2"/>
  </svg>`;
}

/* ================= command palette =================
   One keystroke to anywhere. `/` or Cmd/Ctrl-K opens it; typing filters
   clubs, matchweeks and pages; Enter goes. A dense data site is only fast
   if you can skip the navigation.                                        */
export function initPalette(teams) {
  const items = [
    { label: 'Projected table', kind: 'Page', href: `${BASE}index.html` },
    { label: 'All 380 matches', kind: 'Page', href: `${BASE}matches.html` },
    { label: 'Method and accuracy', kind: 'Page', href: `${BASE}method.html` },
    ...teams.map((t) => ({ label: t.name, kind: 'Club', href: `${BASE}team.html?t=${t.id}`, hint: t.short })),
    ...Array.from({ length: 38 }, (_, i) => ({
      label: `Matchweek ${i + 1}`, kind: 'Matchweek', href: `${BASE}matches.html?mw=${i + 1}` })),
  ];

  document.body.insertAdjacentHTML('beforeend', `
    <div id="palette" hidden>
      <div class="pal-backdrop" data-close></div>
      <div class="pal-box" role="dialog" aria-modal="true" aria-label="Search">
        <input id="pal-input" type="text" placeholder="Search clubs, matchweeks, pages…"
               autocomplete="off" spellcheck="false" aria-label="Search">
        <ul id="pal-list" role="listbox"></ul>
        <div class="pal-foot"><kbd>↑</kbd><kbd>↓</kbd> move · <kbd>↵</kbd> open · <kbd>esc</kbd> close</div>
      </div>
    </div>`);

  const el = document.getElementById('palette');
  const input = document.getElementById('pal-input');
  const list = document.getElementById('pal-list');
  let active = 0, shown = [];

  const render = () => {
    const q = input.value.trim().toLowerCase();
    shown = (q ? items.filter((i) =>
      i.label.toLowerCase().includes(q) || (i.hint || '').toLowerCase().includes(q)) : items
    ).slice(0, 9);
    active = Math.min(active, Math.max(shown.length - 1, 0));
    list.innerHTML = shown.map((i, n) => `
      <li role="option" aria-selected="${n === active}" class="${n === active ? 'on' : ''}" data-n="${n}">
        <span>${esc(i.label)}</span><span class="pal-kind">${i.kind}</span>
      </li>`).join('') || '<li class="pal-empty">Nothing matches that.</li>';
  };
  const open = () => {
    if (!el.hidden) return;
    el.hidden = false; input.value = ''; active = 0; render();
    lockScroll();
    input.focus();
  };
  const close = () => {
    if (el.hidden) return;
    el.hidden = true;
    unlockScroll();
  };
  const go = () => { if (shown[active]) location.href = shown[active].href; };

  addEventListener('keydown', (e) => {
    const typing = /^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement?.tagName || '');
    if (!el.hidden) {
      if (e.key === 'Escape') { close(); }
      else if (e.key === 'ArrowDown') { active = Math.min(active + 1, shown.length - 1); render(); e.preventDefault(); }
      else if (e.key === 'ArrowUp') { active = Math.max(active - 1, 0); render(); e.preventDefault(); }
      else if (e.key === 'Enter') { go(); e.preventDefault(); }
      return;
    }
    if ((e.key === '/' && !typing) || ((e.metaKey || e.ctrlKey) && e.key === 'k')) {
      e.preventDefault(); open();
    }
  });
  input.addEventListener('input', () => { active = 0; render(); });
  list.addEventListener('click', (e) => {
    const li = e.target.closest('li[data-n]');
    if (li) { active = +li.dataset.n; go(); }
  });
  el.addEventListener('click', (e) => { if (e.target.dataset.close !== undefined) close(); });
  document.querySelectorAll('[data-open-palette]').forEach((b) =>
    b.addEventListener('click', open));
}

/* ================= body scroll lock =================
   Without this, dragging inside a dialog on a phone scrolls the page behind it
   and the dialog appears frozen. Position-fixed is the only approach iOS
   Safari honours, so the scroll offset is saved and restored by hand. */
let lockedAt = 0, lockDepth = 0;
export function lockScroll() {
  if (lockDepth++) return;
  lockedAt = window.scrollY;
  document.body.style.position = 'fixed';
  document.body.style.top = `-${lockedAt}px`;
  document.body.style.width = '100%';
}
export function unlockScroll() {
  if (--lockDepth > 0) return;
  lockDepth = 0;
  document.body.style.position = '';
  document.body.style.top = '';
  document.body.style.width = '';
  window.scrollTo(0, lockedAt);
}

/* ================= modal ================= */
export function modal(html) {
  let m = document.getElementById('modal');
  if (!m) {
    document.body.insertAdjacentHTML('beforeend',
      '<div id="modal" hidden><div class="pal-backdrop" data-close></div>' +
      '<div class="modal-box" role="dialog" aria-modal="true" tabindex="-1"></div></div>');
    m = document.getElementById('modal');
    m.addEventListener('click', (e) => { if (e.target.dataset.close !== undefined) closeModal(); });
    addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });
  }
  const box = m.querySelector('.modal-box');
  box.innerHTML = `<button class="modal-x" data-close aria-label="Close">✕</button>${html}`;
  box.scrollTop = 0;
  m.hidden = false;
  lockScroll();
  box.focus();
}
export function closeModal() {
  const m = document.getElementById('modal');
  if (!m || m.hidden) return;
  m.hidden = true;
  unlockScroll();
}

/* ================= score heatmap =================
   The exact scoreline distribution, which is the honest answer to "what will
   the score be": not one number but a cloud, usually a fairly flat one.

   Both axes carry goal counts. Without them the grid is decoration — you can
   see that some cell is likely without being able to say which score it is. */
export function scoreGrid(grid, homeShort, awayShort, best) {
  const n = grid.length;
  const max = Math.max(...grid.flat());
  const ticks = Array.from({ length: n }, (_, i) => i);

  const head = `<div class="gcorner"></div>` +
    ticks.map((a) => `<div class="gtick">${a}</div>`).join('');

  const body = grid.map((row, h) =>
    `<div class="gtick row">${h}</div>` + row.map((p, a) => {
      const t = Math.min(1, (p / max) ** 0.55);
      const step = p < 0.002 ? 0 : Math.min(6, Math.max(1, Math.ceil(t * 6)));
      const isBest = best && best[0] === h && best[1] === a;
      return `<div class="gcell${isBest ? ' best' : ''}" style="background:var(--ramp-${step})"
        tabindex="0" role="img"
        aria-label="${homeShort} ${h}, ${awayShort} ${a}: ${(p * 100).toFixed(1)} percent"
        title="${homeShort} ${h}–${a} ${awayShort} · ${(p * 100).toFixed(1)}%"
        >${p >= 0.04 ? Math.round(p * 100) : ''}</div>`;
    }).join('')).join('');

  return `
    <div class="gwrap">
      <div class="gaxis-y">${esc(homeShort)} goals</div>
      <div class="gmain">
        <div class="ggrid" style="--n:${n}">${head}${body}</div>
        <div class="gaxis-x">${esc(awayShort)} goals</div>
      </div>
    </div>`;
}

/* ================= swing bar =================
   "49.1% → 29.4%" is a sentence, not a figure. A paired-dot track shows the
   same two numbers as a distance, so the size of what is at stake is visible
   rather than arithmetic the reader has to do. Ends carry the same colours as
   the win/draw/loss bars: blue for the home result, red for the away one. */
export function swingBar(away, home, label) {
  const lo = Math.min(away, home), hi = Math.max(away, home);
  return `<div class="swing" role="img"
      aria-label="${esc(label)}: ${(away * 100).toFixed(1)} percent if the away side wins,
                  ${(home * 100).toFixed(1)} percent if the home side wins">
    <span class="swing-track">
      <span class="swing-span" style="left:${lo * 100}%;width:${(hi - lo) * 100}%"></span>
      <span class="swing-dot away" style="left:${away * 100}%"></span>
      <span class="swing-dot home" style="left:${home * 100}%"></span>
    </span>
  </div>`;
}

/* ================= probability bar =================
   One series, one colour: how likely the model said a thing was.          */
export function probBar(p, tone = 'accent') {
  return `<span class="pbar" role="img" aria-label="${(p * 100).toFixed(1)} percent">
    <span style="width:${Math.max(p * 100, 1.5)}%;background:var(--${tone})"></span></span>`;
}

/* ================= multi-series line chart =================
   Used for the forecast's own movement over time. Series are direct-labelled
   at their endpoint, so identity never rests on colour alone.             */
export function lineChart(series, { w = 720, h = 260, pad = 34, fmt = (v) => v,
                                    yMax = 1, yLabel = '' } = {}) {
  if (!series.length || series[0].points.length < 2) return '';
  const n = series[0].points.length;
  const x = (i) => pad + 8 + (i * (w - pad - 96)) / Math.max(n - 1, 1);
  const y = (v) => h - pad - (v / yMax) * (h - pad - 14);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * yMax);
  return `<svg viewBox="0 0 ${w} ${h}" style="width:100%;height:auto" role="img"
      aria-label="${esc(yLabel)} over time">
    ${ticks.map((t) => `
      <line x1="${pad}" y1="${y(t)}" x2="${w - 92}" y2="${y(t)}" stroke="var(--grid)" stroke-width="1"/>
      <text x="${pad - 6}" y="${y(t) + 4}" fill="var(--muted)" font-size="10" text-anchor="end">${fmt(t)}</text>`).join('')}
    ${series.map((s) => `
      <polyline points="${s.points.map((p, i) => `${x(i).toFixed(1)},${y(p).toFixed(1)}`).join(' ')}"
        fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
      <circle cx="${x(n - 1).toFixed(1)}" cy="${y(s.points[n - 1]).toFixed(1)}" r="3.5"
        fill="${s.color}" stroke="var(--surface)" stroke-width="2"/>
      <text x="${w - 86}" y="${y(s.points[n - 1]) + 4}" fill="var(--ink2)" font-size="11">${esc(s.label)}</text>`).join('')}
  </svg>`;
}

/* ================= match dialog =================
   One implementation shared by every page that can open a match. `meta` is the
   forecast's team map; `opts.link` adds a permalink to the matches page. */
export function matchModal(m, meta, opts = {}) {
  const EVENT = { title: 'the title', ucl: 'a top-five finish', releg: 'relegation' };
  const h = meta[m.h], a = meta[m.a];
  const fmtDay = (iso) => new Date(iso + 'T12:00:00').toLocaleDateString('en-GB',
    { weekday: 'short', day: 'numeric', month: 'short' });
  const swings = (m.swings || []).map((s) => `
    <div class="swing-row">
      <span>${esc(meta[s.team].name)} — ${EVENT[s.event]}</span>
      ${swingBar(s.away, s.home, meta[s.team].name + ' ' + EVENT[s.event])}
      <b>${pct(s.away)} → ${pct(s.home)}</b>
    </div>`).join('');
  tip(null);
  modal(`
    <h3>${esc(h.name)} v ${esc(a.name)}</h3>
    <p class="msub">Matchweek ${m.md} · ${fmtDay(m.date)}${m.time ? ` · ${m.time}` : ''}
      ${m.played ? ` · finished ${m.hg}–${m.ag}` : ''}
      ${opts.link === false ? '' :
        ` · <a href="${BASE}matches.html?m=${m.h}--${m.a}">link to this match</a>`}</p>
    <div class="wdl" style="height:26px;margin-bottom:6px">
      <i class="h" style="flex:${Math.max(m.ph, .02)}"><b>${Math.round(m.ph * 100)}</b></i>
      <i class="d" style="flex:${Math.max(m.pd, .02)}"><b>${Math.round(m.pd * 100)}</b></i>
      <i class="a" style="flex:${Math.max(m.pa, .02)}"><b>${Math.round(m.pa * 100)}</b></i>
    </div>
    <div class="hint" style="margin-bottom:20px">
      ${esc(h.short)} win · draw · ${esc(a.short)} win &nbsp;·&nbsp;
      expected goals ${m.xgh} – ${m.xga} &nbsp;·&nbsp; over 2.5 goals ${pct(m.o25)}
      &nbsp;·&nbsp; both score ${pct(m.btts)}</div>
    <h4>Every plausible scoreline</h4>
    <p class="hint" style="margin:0 0 12px">The likeliest single score is ${m.sc[0]}–${m.sc[1]},
      and even that lands only ${pct(m.scp)} of the time.</p>
    ${scoreGrid(m.grid, h.short, a.short, m.sc)}
    ${swings ? `<h4 style="margin-top:24px">What rides on it</h4>
      <p class="hint" style="margin:0 0 6px">Where each club's season stands if the away
        side wins, and if the home side does.</p>
      ${swings}
      <div class="swing-key">
        <span><i style="background:var(--away)"></i>${esc(a.short)} win</span>
        <span><i style="background:var(--accent)"></i>${esc(h.short)} win</span>
      </div>` : ''}`);
}
