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
    <header class="masthead"><div class="wrap">
      <a class="brand" href="${BASE}index.html">
        <span class="mark"></span>
        <b>Ninety</b><span>Premier League forecast</span>
      </a>
      <nav class="top">
        <a href="${BASE}index.html"${page === 'table' ? ' aria-current="page"' : ''}>Table</a>
        <a href="${BASE}matches.html"${page === 'matches' ? ' aria-current="page"' : ''}>Matches</a>
        <a href="${BASE}team.html"${page === 'team' ? ' aria-current="page"' : ''}>Clubs</a>
        <a href="${BASE}method.html"${page === 'method' ? ' aria-current="page"' : ''}>Method</a>
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
