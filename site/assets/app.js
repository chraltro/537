/* Shared helpers: league state, data loading, formatting, chrome, tooltip,
   small SVG charts. */

/* ================= where the site lives =================
   Resolved from this module's own URL rather than the page's, so a 404 served
   at an arbitrary depth still finds the data and still links home correctly. */
const ROOT = new URL('..', import.meta.url).pathname;      // '/', '/537/', …
const DATA = `${ROOT}data/`;

/* ================= league state =================
   One site, several leagues. Which one is a URL parameter so that every link is
   shareable; the manifest decides what exists, what is ready, and the default.

   The manifest is awaited at module load — the masthead needs the league list
   and every data path needs the slug, so there is no useful work to do before
   it lands. A broken manifest degrades to the Premier League rather than
   blanking the site. */
const FALLBACK_MANIFEST = {
  default: 'premier-league',
  leagues: [{
    slug: 'premier-league', name: 'Premier League', country: 'England', ready: true,
    n_teams: 20, ucl_places: 5, releg_places: 3, releg_note: null,
  }],
};

async function loadManifest() {
  try {
    const r = await fetch(`${DATA}leagues.json`, { cache: 'no-cache' });
    if (!r.ok) throw new Error(`leagues.json ${r.status}`);
    const m = await r.json();
    if (!m || !Array.isArray(m.leagues) || !m.leagues.length) throw new Error('empty manifest');
    return m;
  } catch (e) {
    console.warn('league manifest unavailable, falling back to the Premier League:', e.message);
    return FALLBACK_MANIFEST;
  }
}

export const MANIFEST = await loadManifest();
export const LEAGUES = MANIFEST.leagues;

/* The default has to be a league that actually has data behind it, or the first
   visit 404s on every fetch. */
const _named = LEAGUES.find((l) => l.slug === MANIFEST.default);
const _ready = LEAGUES.filter((l) => l.ready);
export const DEFAULT_LEAGUE =
  ((_named && _named.ready) ? _named : (_ready[0] || _named || LEAGUES[0])).slug;

/* The competition lives in the URL so every link is shareable, which meant a
   bare visit always landed on the Premier League even for somebody who only
   ever reads the Eredivisie. Remember the last one chosen, and let an explicit
   `?lg=` still win over it. */
const LG_KEY = 'plf-league';
let _remembered = null;
try { _remembered = localStorage.getItem(LG_KEY); } catch { /* private mode */ }
const _asked = new URLSearchParams(location.search).get('lg') || _remembered;
/* An unknown league falls back rather than 404ing. A known but not-yet-live one
   is honoured when it is asked for by name: the switcher will not offer it, but
   a direct link renders whatever staging data its directory holds, which is how
   a league is reviewed before it goes live. */
const _chosen = LEAGUES.find((l) => l.slug === _asked)
             || LEAGUES.find((l) => l.slug === DEFAULT_LEAGUE);

/* Live record for the current league. Starts as the manifest entry and is
   topped up from the league's own JSON as that arrives (see absorb below), so
   the manifest is a fallback rather than a second source of truth. */
export const LG = { ...(_chosen || FALLBACK_MANIFEST.leagues[0]) };
try { localStorage.setItem(LG_KEY, LG.slug); } catch { /* private mode */ }
export const getLeague = () => LG.slug;

const NUMWORD = ['zero', 'one', 'two', 'three', 'four', 'five',
                 'six', 'seven', 'eight', 'nine', 'ten'];
const numWord = (n) => NUMWORD[n] || String(n);

/* "the Premier League" but "La Liga". A league name takes the definite article
   unless it already carries one of its own, or ends in the bare letter or
   number that makes it a name rather than a description (Serie A, Ligue 1). */
function withArticle(name) {
  const n = String(name || '').trim();
  const last = n.split(/\s+/).pop() || '';
  const ownArticle = /^(la|le|les|el|il|lo|los|die|der|das|de)\s/i.test(n);
  const bareTail = /^(\d+|[A-Za-z])$/.test(last);
  return (ownArticle || bareTail) ? n : `the ${n}`;
}

/* Every league-dependent phrase on the site is derived here, once, from
   counts. Nothing downstream is allowed to write "top five" by hand.

   A cup's league phase is the same table with different stakes: the top places
   go straight through, a middle band plays a knockout play-off and the rest are
   out. The counts still come from the manifest, so the only thing `kind` picks
   is which set of words describes them. */
export function lg() {
  const n = LG.n_teams || 20;
  const ucl = LG.ucl_places || 5;
  const rel = LG.releg_places || 3;
  const cup = LG.kind === 'cup';
  /* A second tier is the same table read against different stakes again: the
     top line is automatic promotion and a band under it plays off for one more
     place. Same two counts, same machinery, different words. */
  const promo = LG.kind === 'promotion';
  const adv = LG.advance_direct || ucl;       // straight into the last 16 / straight up
  const pla = LG.advance_playoff || 0;        // in via the play-off
  const the = withArticle(LG.name);
  /* A double round robin unless the data says otherwise; a cup's league phase
     is a partial one, so its own fixture count is the only honest source. */
  const nMatches = LG.matches_total || n * (n - 1);
  return {
    slug: LG.slug,
    name: LG.name,
    the,                                      // "the Premier League" / "La Liga"
    The: the.charAt(0).toUpperCase() + the.slice(1),   // sentence-initial
    country: LG.country || '',
    kind: cup ? 'cup' : (promo ? 'promotion' : 'league'),
    isCup: cup,
    isPromotion: promo,
    hasPlayoff: cup || promo,
    nTeams: n,
    uclPlaces: ucl,
    relegPlaces: rel,
    relegNote: LG.releg_note || '',
    advanceDirect: adv,
    advancePlayoff: pla,
    cutAt: adv + pla,                         // last place still in the cup
    nMatches,
    nWeeks: cup ? Math.round((nMatches * 2) / n) : (n - 1) * 2,
    roundWord: cup ? 'Matchday' : 'Matchweek',
    roundWords: cup ? 'matchdays' : 'matchweeks',
    roundAbbr: cup ? 'MD' : 'MW',
    /* A second tier now shows both routes up, so the automatic column has to
       say which one it is. "Up" on its own would read as either. */
    topN: promo ? 'Auto' : `Top ${adv}`,      // column header
    topWord: `top ${numWord(adv)}`,           // "the top five"
    topAdj: `top-${numWord(adv)}`,            // "top-five race"
    topFinish: promo ? 'automatic promotion' : `a top-${numWord(adv)} finish`,
    topMeans: cup ? 'a direct place in the last 16'
                  : (promo ? 'automatic promotion' : 'a Champions League place'),
    topRace: cup ? `the race for the top ${numWord(adv)}`
                 : (promo ? 'the promotion race' : `the top-${numWord(adv)} race`),
    winN: cup ? 'Trophy' : 'Title',           // column header
    winWord: cup ? 'the trophy' : 'the title',
    winRace: cup ? 'the race for the trophy' : 'the title race',
    /* What "first place" means: the line files measure the top of the table,
       which in a cup is the top of the league phase and not the trophy. */
    firstWord: cup ? 'first place in the league phase' : 'the title',
    playoffWord: cup ? 'the knockout play-off' : 'the promotion play-offs',
    downN: cup ? 'Out' : 'Down',
    downWord: cup ? 'elimination' : 'relegation',
    downVerb: cup ? 'go out' : 'go down',
    downRace: cup ? 'the race to survive the cut' : 'the relegation battle',
    surviveWord: cup ? 'survive the cut' : 'stay up',
    lastSafe: n - rel,                        // last position above the drop
    relegPhrase: cup ? `${rel} places that go out`
                     : `${rel} relegation place${rel === 1 ? '' : 's'}`,
    /* Appended, with its own leading space, wherever the drop is described. */
    relegTail: LG.releg_note ? ` ${LG.releg_note}.` : '',
  };
}

/* The events a fixture can swing, labelled for this league. A cup swings a
   different three — the direct places, the cut, and going out — and carries the
   league keys as well so anything reading `title`/`ucl`/`releg` still reads. */
export function eventLabels() {
  const W = lg();
  const base = { title: W.winWord, ucl: W.topFinish, releg: W.downWord };
  if (W.isPromotion) return { ...base, ucl: 'automatic promotion' };
  return W.isCup
    ? { ...base, top8: W.topFinish, qualify: 'reaching the knockout stage', out: 'elimination' }
    : base;
}

/* ================= rounds =================
   The league phase numbers its rounds; the knockout names them. Stage codes are
   the data's own, so anything unrecognised is shown as it arrived rather than
   guessed at. */
const STAGE = {
  KPO: 'Knockout play-off', R16: 'Round of 16', QF: 'Quarter-final',
  SF: 'Semi-final', F: 'Final', '3P': 'Third-place play-off',
};
export const isMatchweek = (md) =>
  md !== null && md !== undefined && String(md).trim() !== '' && !Number.isNaN(Number(md));

export function roundLabel(md, leg) {
  if (isMatchweek(md)) return `${lg().roundWord} ${md}`;
  const s = STAGE[md] || String(md);
  return leg ? `${s}, ${leg === 1 ? 'first' : 'second'} leg` : s;
}
export function roundShort(md) {
  return isMatchweek(md) ? `${lg().roundAbbr} ${md}` : String(md);
}

/* Two-legged ties arrive as two rows with the venues swapped, and which leg a
   row is cannot be read off the row itself. Worked out once, on load. */
function markLegs(rows) {
  const seen = new Map();
  rows.forEach((m) => {
    if (isMatchweek(m.md)) return;                     // league phase: single games
    const k = `${[m.h, m.a].sort().join('|')}@${m.md}`;
    const first = seen.get(k);
    if (!first) { seen.set(k, m); return; }
    const later = String(m.date) >= String(first.date) ? m : first;
    (later === m ? first : m).leg = 1;
    later.leg = 2;
  });
}

/* ================= links =================
   The chosen league has to survive every hop, so an internal link is built
   here rather than written literally. The default league is left clean: no
   parameter, so Premier League URLs look exactly as they always have. */
export function url(path) {
  const u = new URL(path, `${location.origin}${ROOT}`);
  if (LG.slug !== DEFAULT_LEAGUE) u.searchParams.set('lg', LG.slug);
  else u.searchParams.delete('lg');
  return u.pathname + u.search + u.hash;
}

/* For pages that rewrite their own query string (filters live in the URL). */
export function withLg(params) {
  if (LG.slug !== DEFAULT_LEAGUE) params.set('lg', LG.slug);
  else params.delete('lg');
  return params;
}

/* ================= data ================= */
const store = {};

export async function data(name) {
  if (!store[name]) {
    try {
      const r = await fetch(`${DATA}${LG.slug}/${name}.json`, { cache: 'no-cache' });
      if (!r.ok) throw new Error(`${name}.json ${r.status}`);
      store[name] = await r.json();
    } catch (e) {
      dataError(`${LG.name} ${name}.json`, e);
      throw e;
    }
    absorb(name, store[name]);
  }
  return store[name];
}

/* Files that belong to the whole site rather than to one competition: the
   cross-league ranking and the head-to-head record behind the comparison tool.
   They deliberately do not take a league slug, because their entire point is
   that they are not scoped to one. */
/* A page that cannot load its data should say so, not sit half-painted behind
   a skeleton for ever. Called by the data helpers below; the message names the
   file, because "something went wrong" helps nobody. */
export function dataError(what, err) {
  console.error(what, err);
  const main = document.getElementById('main') || document.body;
  if (document.getElementById('loadfail')) return;
  main.insertAdjacentHTML('afterbegin', `
    <div class="loadfail" id="loadfail" role="alert">
      <b>This page could not load its forecast.</b>
      <span>${esc(what)} did not arrive${err && err.message ? `: ${esc(err.message)}` : ''}.
      The site rebuilds every six hours, so this is usually a deploy in progress —
      try again in a few minutes. If it persists, the competition may not have
      been built this cycle.</span>
      <a href="${ROOT}index.html">Back to the Premier League</a>
    </div>`);
}

export async function siteData(name) {
  const key = `::${name}`;
  if (!store[key]) {
    const r = await fetch(`${DATA}${name}.json`, { cache: 'no-cache' });
    if (!r.ok) throw new Error(`${name}.json ${r.status}`);
    store[key] = await r.json();
  }
  return store[key];
}

/* A league's own files know more about it than the manifest does. Take what
   they carry and leave the manifest value standing where they do not — the
   first build of a league predates these fields. */
function absorb(name, d) {
  if (!d || typeof d !== 'object') return;
  const num = (k) => { if (typeof d[k] === 'number') LG[k] = d[k]; };
  if (name === 'forecast') {
    num('ucl_places');                                  // legacy top-level field
    num('matches_total');
    if (d.league && typeof d.league === 'object') Object.assign(LG, d.league);
    if (d.replay) { LG.replay = d.replay; LG.replay_note = d.replay_note || ''; }
  }
  if (name === 'sim_input') {
    num('ucl_places'); num('releg_places'); num('n_teams');
    ['kind', 'advance_direct', 'advance_playoff'].forEach((k) => {
      if (d[k] !== null && d[k] !== undefined) LG[k] = d[k];
    });
    /* The simulator never loads the forecast, so its fixture list is where the
       cup's partial round robin gets its size from. */
    if (!LG.matches_total && Array.isArray(d.fixtures)) LG.matches_total = d.fixtures.length;
  }
  if (name === 'matches' && Array.isArray(d.matches)) markLegs(d.matches);
  if (LG.replay) showReplayBanner();
}

/* ================= staging banner =================
   A league can be published for review before it is live, with a replayed
   season standing in for a draw that has not happened. That has to be
   unmissable, and it has to say what the dataset says about itself: the note is
   the file's own, never a sentence written here. Live data carries no `replay`
   key and so shows nothing. */
export function showReplayBanner() {
  if (!LG.replay || !LG.replay_note) return;
  if (document.getElementById('replaybanner')) return;
  const html = `<div class="banner" id="replaybanner" role="status">
      <b>Staging: ${esc(LG.replay)} replay</b>
      <span>${esc(LG.replay_note)}</span></div>`;
  const head = document.querySelector('header.masthead');
  if (head) head.insertAdjacentHTML('afterend', html);
  else document.body.insertAdjacentHTML('afterbegin', html);
}

/* For pages that never read forecast.json: only a league the manifest calls
   not-live can be staging, so only those pay for the extra fetch. */
export async function ensureReplayBanner() {
  if (!LG.replay && LG.ready === false) {
    try { await data('forecast'); } catch { /* nothing to declare */ }
  }
  showReplayBanner();
}

/* ================= page metadata =================
   Titles and descriptions name the league, so a shared link says which one. */
export function setMeta({ title, ogTitle, description, image }) {
  const set = (sel, v) => {
    const el = document.querySelector(sel);
    if (el && v) el.setAttribute('content', v);
  };
  if (title) document.title = title;
  set('meta[property="og:title"]', ogTitle || title);
  set('meta[name="description"]', description);
  set('meta[property="og:description"]', description);
  /* Share-card images are drawn at build time into site/og/. Note that link
     scrapers do not run JavaScript, so setting og:image here only helps clients
     that do; the per-page default in the HTML is what most of them read. */
  if (image) {
    const abs = new URL(image, `${location.origin}${ROOT}`).href;
    set('meta[property="og:image"]', abs);
    set('meta[name="twitter:image"]', abs);
  }
}

/* The build writes one card per competition and one per club. */
export const cardUrl = (slug, team) =>
  `${ROOT}og/${slug}${team ? `/${team}` : ''}.png`;

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

/* ---------------- chrome ---------------- */

/* ================= where you can go =================
   One registry, read by the masthead, the overflow menu and the command
   palette. Before this the three of them each kept their own list and had
   already drifted: the palette knew about the season review and the masthead
   did not.

   `group` is the honest division, and the reason the header stopped feeling
   crowded. Four of these pages are one competition seen four ways; two are not
   scoped to a competition at all, and putting them in one undifferentiated row
   of eight implied that switching league would change them, which it never
   did. */
export const PAGES = [
  { id: 'table',    label: 'Table',           long: 'Projected table',      file: 'index.html',     group: 'league', primary: true },
  { id: 'matches',  label: 'Matches',         long: 'Every match',          file: 'matches.html',   group: 'league', primary: true },
  { id: 'team',     label: 'Clubs',           long: 'Club pages',           file: 'team.html',      group: 'league', primary: true },
  { id: 'races',    label: 'Races',           long: 'The races',            file: 'races.html',     group: 'league', primary: true },
  { id: 'sim',      label: 'What if',         long: 'What if? simulator',   file: 'simulator.html', group: 'league' },
  { id: 'review',   label: 'Season review',   short: 'Review',              long: 'How wrong were we?',   file: 'review.html',    group: 'league' },
  { id: 'rankings', label: 'Global rankings', short: 'Rankings',            long: 'Global club rankings', file: 'rankings.html',  group: 'europe', site: true },
  { id: 'compare',  label: 'Compare clubs',   short: 'Compare',             long: 'Compare two clubs',    file: 'compare.html',   group: 'europe', site: true },
  { id: 'method',   label: 'Method',          long: 'Method and accuracy',  file: 'method.html',    group: 'about' },
];

const GROUPS = [
  ['league', 'This competition'],
  ['europe', 'Across Europe'],
  ['about', 'About'],
];

//: Pages not scoped to a competition, where `?lg=` would mean nothing.
const SITE_ONLY = new Set(PAGES.filter((p) => p.site).map((p) => p.file));

export const pageHref = (p) => (p.site ? `${ROOT}${p.file}` : url(p.file));

/* Links written into the HTML by hand still have to carry the league. Rewriting
   them here means a new page cannot forget to, and same-page anchors and
   outbound links are left alone. */
function lgifyLinks(root) {
  if (LG.slug === DEFAULT_LEAGUE) return;
  root.querySelectorAll('a[href]').forEach((a) => {
    const raw = a.getAttribute('href');
    if (!raw || raw.startsWith('#') || /^[a-z]+:/i.test(raw)) return;
    const u = new URL(raw, location.href);
    if (u.origin !== location.origin) return;
    /* The cross-league pages take no league. Stamping one on them put
       `?lg=serie-a` in the address bar of a page showing every club in Europe,
       which is a promise that page does not keep. */
    if (SITE_ONLY.has(u.pathname.split('/').pop())) return;
    u.searchParams.set('lg', LG.slug);
    a.setAttribute('href', u.pathname + u.search + u.hash);
  });
}

export function initChrome(page) {
  const saved = localStorage.getItem('plf-theme');
  if (saved) document.documentElement.setAttribute('data-theme', saved);

  const W = lg();
  /* A league that is not live yet stays in the list but cannot be chosen —
     except when it is the one being looked at, since a direct link to staging
     data has to leave the switcher showing where you are. What it is waiting
     for is the manifest's line to write, not this file's. */
  /* When the page you are on lives inside the overflow menu, the button wears
     its name. Otherwise the header would say "More" while you stood on Method,
     and you would have to open it to find out where you were. */
  const current = PAGES.find((p) => p.id === page);
  const here = current && !current.primary;
  /* The button names the page you are standing on, when that page lives in the
     menu. On a phone it shares a row with four tabs, and "Global rankings" does
     not fit next to them -- so the registry carries a short form for the names
     that are too long, and the bar shows whichever fits. */
  const hereLabel = here ? current.label : 'More';
  const hereShort = here ? (current.short || current.label) : 'More';

  const opts = LEAGUES.map((l) => {
    const here = l.slug === LG.slug;
    const wait = l.ready_note || 'Coming soon';
    return `<option value="${esc(l.slug)}"${here ? ' selected' : ''}${
      (l.ready || here) ? '' : ` disabled title="${esc(wait)}"`}>${
      esc(l.name)}${l.ready ? '' : ' (not live yet)'}</option>`;
  }).join('');

  document.body.insertAdjacentHTML('afterbegin', `
    <a class="skip" href="#main">Skip to content</a>
    <header class="masthead"><div class="wrap">
      <a class="brand" href="${url('index.html')}">
        <span class="mark"></span>
        <b>537</b>
      </a>
      <label class="lgswitch">
        <span class="vh">League</span>
        <select id="lgsel" aria-label="Choose a league">${opts}</select>
      </label>
      <nav class="top" aria-label="Sections">
        ${PAGES.filter((p) => p.primary).map((p) => `
          <a class="np" href="${pageHref(p)}"${
            page === p.id ? ' aria-current="page"' : ''}>${esc(p.label)}</a>`).join('')}
        <div class="navmenu">
          <button class="morebtn" id="moretog" aria-expanded="false" aria-haspopup="true"
                  aria-controls="morepop"${here ? ' data-here="1"' : ''}>
            <span class="full">${esc(hereLabel)}</span><span
              class="abbr">${esc(hereShort)}</span>
            <svg width="10" height="10" viewBox="0 0 16 16" aria-hidden="true" class="chev">
              <path d="M3 6l5 5 5-5" fill="none" stroke="currentColor"
                    stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </button>
          <div class="menu-pop" id="morepop" hidden>
            ${GROUPS.map(([g, heading]) => {
              const items = PAGES.filter((p) => p.group === g);
              if (!items.length) return '';
              return `<p class="menu-h">${esc(heading === 'This competition' ? W.name : heading)}</p>`
                + items.map((p) => `
                  <a class="${p.primary ? 'm-primary' : ''}" href="${pageHref(p)}"${
                    page === p.id ? ' aria-current="page"' : ''}>${esc(p.label)}</a>`).join('');
            }).join('')}
          </div>
        </div>
      </nav>
      <div class="tools">
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
      </div>
    </div></header>`);

  document.getElementById('tt').addEventListener('click', () => {
    const now = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', now);
    localStorage.setItem('plf-theme', now);
    window.dispatchEvent(new Event('themechange'));
  });

  /* Switching league keeps you on the page you were reading. Other parameters
     ride along; a club or match that does not exist over there degrades to that
     page's own default rather than an error. */
  document.getElementById('lgsel').addEventListener('change', (e) => {
    const u = new URL(location.href);
    if (e.target.value === DEFAULT_LEAGUE) u.searchParams.delete('lg');
    else u.searchParams.set('lg', e.target.value);
    location.href = u.toString();
  });

  document.body.insertAdjacentHTML('beforeend', `
    <footer><div class="wrap">
      <p>An open forecast of ${esc(W.the)}, rebuilt in the spirit of FiveThirtyEight's
      Soccer Power Index. Ratings, match probabilities and season simulations are generated
      from public match data, with no private feeds and no hand-tuned opinions.
      <a href="${url('method.html')}">How it works and how accurate it is →</a></p>
      <p>Match data: <a href="https://github.com/datasets/football-datasets">football-datasets</a>
      (mirroring football-data.co.uk) and <a href="https://github.com/openfootball">openfootball</a>.
      Not affiliated with ${esc(W.the)}. Not betting advice.</p>
      <p>Take it with you:
      <a href="${ROOT}cal/${esc(LG.slug)}.ics">fixtures in your calendar</a> ·
      <a href="${ROOT}feed.json">JSON feed</a> ·
      <a href="${ROOT}feed.xml">RSS</a> ·
      <a href="${ROOT}embed.html?lg=${esc(LG.slug)}">embeddable widget</a>. Each is a static
      file this build writes, refreshed every six hours.</p>
    </div></footer>`);
  document.body.insertAdjacentHTML('beforeend', '<div id="tip" role="tooltip"></div>');

  initNavMenu();
  trackHeadHeight();
  lgifyLinks(document);
  registerWorker();
}

/* The masthead's real height, published for anything that has to sit under it.
   Sticky table headers used to hardcode 60px, and 52px on a phone; the two-row
   phone masthead made both wrong and pinned the header over the first row.
   Measuring is cheap and survives the next change to the bar. */
function trackHeadHeight() {
  const head = document.querySelector('.masthead');
  if (!head) return;
  const set = () => document.documentElement.style.setProperty(
    '--head-h', `${Math.round(head.getBoundingClientRect().height)}px`);
  set();
  if (typeof ResizeObserver === 'function') new ResizeObserver(set).observe(head);
  else addEventListener('resize', set);
}

/* ================= the overflow menu =================
   Everything the masthead used to shout is in here instead, grouped by whether
   it belongs to the competition you are looking at or to the whole site. Small
   enough to write by hand: a button, a popup, and the four keys a menu owes you.

   The primary links appear twice in the markup, once inline and once here.
   Only ever one of the two is displayed — CSS hides the other outright, which
   takes it out of the accessibility tree as well — so nothing is announced or
   tabbed to twice. */
function initNavMenu() {
  const btn = document.getElementById('moretog');
  const pop = document.getElementById('morepop');
  if (!btn || !pop) return;
  const menu = btn.closest('.navmenu');
  const shown = () => [...pop.querySelectorAll('a')].filter((a) => a.offsetParent !== null);

  const open = (focusFirst) => {
    pop.hidden = false;
    btn.setAttribute('aria-expanded', 'true');
    if (focusFirst) shown()[0]?.focus();
  };
  const close = (refocus) => {
    if (pop.hidden) return;
    pop.hidden = true;
    btn.setAttribute('aria-expanded', 'false');
    if (refocus) btn.focus();
  };

  btn.addEventListener('click', (e) => {
    e.preventDefault();
    if (pop.hidden) open(false); else close(false);
  });
  /* One listener on the wrapper rather than one on each of the button and the
     popup. Escape has to work while focus is still on the button — which is
     where it is immediately after a click opens the menu — and a listener bound
     to the popup never sees that keystroke.

     Enter and Space are deliberately left alone: a <button> already turns them
     into a click, and intercepting them here as well would open the menu and
     then let the click close it again. */
  menu.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      if (!pop.hidden) close(true);
      return;
    }
    if (document.activeElement === btn) {
      if (e.key === 'ArrowDown') { e.preventDefault(); open(true); }
      return;
    }
    const items = shown();
    const i = items.indexOf(document.activeElement);
    if (e.key === 'ArrowDown') { e.preventDefault(); items[Math.min(i + 1, items.length - 1)]?.focus(); }
    else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (i <= 0) close(true); else items[i - 1].focus();
    }
  });
  /* Anything outside the menu closes it, including a click on the page behind
     the sticky header. `focusin` covers tabbing out, which a click listener
     alone would miss. */
  document.addEventListener('click', (e) => {
    if (!pop.hidden && !e.target.closest('.navmenu')) close(false);
  });
  document.addEventListener('focusin', (e) => {
    if (!pop.hidden && !e.target.closest('.navmenu')) close(false);
  });
}

/* Offline. Network-first with a cache fallback, never the other way round: the
   forecast is rebuilt every six hours and a cache-first worker would happily
   serve April's numbers in May. The page says when its data is from, so a
   fallback read is visible rather than silent. */
function registerWorker() {
  if (!('serviceWorker' in navigator) || location.protocol === 'file:') return;
  navigator.serviceWorker.register(`${ROOT}sw.js`, { scope: ROOT })
    .catch((e) => console.warn('offline support unavailable:', e.message));
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
/* A club has two colours in `team_meta.json` and the site only ever used one,
   which left the several clubs whose primaries collide after the contrast nudge
   looking identical. A thin second band costs nothing and separates them. */
export function chipStyle(t) {
  const a = chipColor(t.primary);
  const b = t.secondary ? chipColor(t.secondary) : null;
  return b && b !== a
    ? `background:linear-gradient(180deg,${a} 0 55%,${b} 55% 100%)`
    : `background:${a}`;
}

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
export function rampStep(p, max) {
  if (!(p > 0.002)) return 0;                           // genuinely never happens
  const t = Math.min(1, (p / (max || 1)) ** 0.55);      // compress, but keep the peak loud
  return Math.min(6, Math.max(1, Math.ceil(t * 6)));
}

export const rampColor = (p, max) => `var(--ramp-${rampStep(p, max)})`;

/* The text colour that step can carry. A seven-step ramp cannot be read with a
   single foreground: white on the palest step of the dark ramp was 1.79:1, and
   near-black on the darkest step of the light ramp was 1.99:1. Each step names
   its own in the stylesheet, chosen as whichever of white or near-black clears
   more contrast against it. */
export const rampInk = (p, max) => `var(--ink-${rampStep(p, max)})`;

/* ---------------- sortable tables ---------------- */
export function makeSortable(table, rows, render, initial, opts = {}) {
  /* A sortable header used to be a `th` with a click listener and nothing else:
     no way to reach it from a keyboard, and no way for a screen reader to learn
     which column was active or which way it ran. Both are fixed here rather
     than per page, because every table on the site goes through this function.

     The active column also rides in the URL, since a sorted table is a thing
     people send each other and the site's convention is that what you are
     looking at is what you can link to. */
  const param = opts.param === false ? null : (opts.param || 'sort');
  const url = new URL(location.href);
  let key = initial, dir = -1;
  if (param) {
    const want = url.searchParams.get(param);
    if (want && table.querySelector(`th[data-k="${CSS.escape(want)}"]`)) key = want;
    if (url.searchParams.get(`${param}dir`) === 'asc') dir = 1;
  }

  const heads = [...table.querySelectorAll('th[data-k]')];
  const apply = (push) => {
    heads.forEach((h) => {
      h.removeAttribute('data-dir');
      h.setAttribute('aria-sort', 'none');
    });
    const h = heads.find((x) => x.dataset.k === key);
    if (h) {
      h.setAttribute('data-dir', dir === -1 ? 'desc' : 'asc');
      h.setAttribute('aria-sort', dir === -1 ? 'descending' : 'ascending');
    }
    if (param && push) {
      const u = new URL(location.href);
      u.searchParams.set(param, key);
      u.searchParams.set(`${param}dir`, dir === -1 ? 'desc' : 'asc');
      history.replaceState({}, '', u.pathname + u.search + u.hash);
    }
    const sorted = [...rows].sort((a, b) => {
      const x = a[key], y = b[key];
      const c = typeof x === 'string' ? x.localeCompare(y) : x - y;
      return c * dir;
    });
    render(sorted);
  };

  const toggle = (h) => {
    if (key === h.dataset.k) dir = -dir;
    else { key = h.dataset.k; dir = h.dataset.k === 'name' ? 1 : -1; }
    apply(true);
  };
  heads.forEach((h) => {
    h.classList.add('sortable');
    h.setAttribute('tabindex', '0');
    h.setAttribute('role', 'button');
    if (!h.title) h.title = `Sort by ${h.textContent.trim()}`;
    h.addEventListener('click', () => toggle(h));
    h.addEventListener('keydown', (e) => {
      if (e.key !== 'Enter' && e.key !== ' ') return;
      e.preventDefault();
      toggle(h);
    });
  });
  apply(false);
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

/* ================= bracket strip =================
   A cup asks a different question from a league: not where a club finishes but
   how far it gets. This is that in one line — the two ways out of the league
   phase, then the chance of reaching each knockout round, on the same heat ramp
   the finishing-position strip uses so the two read as one system.

   The number sits under its cell rather than inside it: a ramp runs from nearly
   the surface colour to nearly the ink colour, and no single text colour is
   legible against both ends of it.                                          */
export function bracketStrip(t, opts = {}) {
  const W = lg();
  const cells = [
    [W.topN, t.p_top8, `finishes in the ${W.topWord}, which means ${W.topMeans}`],
    ['Play-off', t.p_playoff, 'finishes in the knockout play-off places'],
    ['Last 16', t.p_r16, 'reaches the round of 16'],
    ['Quarter', t.p_qf, 'reaches the quarter-finals'],
    ['Semi', t.p_sf, 'reaches the semi-finals'],
    ['Final', t.p_final, 'reaches the final'],
    ['Trophy', t.p_win, 'wins it'],
  ].filter(([, p]) => typeof p === 'number');
  if (cells.length < 3) return '';

  /* The strip scales to its container, so seven cells across a phone would
     shrink the labels out of legibility. Wrapping to fewer columns keeps the
     type the size it was drawn at. */
  const cols = Math.min(opts.cols || cells.length, cells.length);
  const cw = 92, gap = 6, split = 14, top = 15, ch = 38, rowH = top + ch + 26;
  const col = (i) => i % cols;
  const row = (i) => Math.floor(i / cols);
  /* The gap after the second cell is the one real boundary on the row:
     everything past it is the knockout, and the two cells before it are the two
     different ways of reaching it. */
  const x = (i) => col(i) * (cw + gap) + (row(i) === 0 && col(i) >= 2 ? split : 0);
  const y = (i) => row(i) * rowH;
  const w = cols * (cw + gap) - gap + (cols > 2 ? split : 0);
  const h = (row(cells.length - 1) + 1) * rowH - 6;
  const name = opts.name || t.name || '';

  const body = cells.map(([label, p, what], i) => `
    <g>
      <title>${esc(name)} ${esc(what)}: ${pct(p, 1)}</title>
      <text class="bl" x="${x(i) + cw / 2}" y="${y(i) + top - 5}" text-anchor="middle">${esc(label)}</text>
      <rect x="${x(i)}" y="${y(i) + top}" width="${cw}" height="${ch}" rx="4"
            fill="${rampColor(p, 1)}" stroke="var(--grid)" stroke-width="1"/>
      <text class="bv" x="${x(i) + cw / 2}" y="${y(i) + top + ch + 15}"
            text-anchor="middle">${pct(p)}</text>
    </g>`).join('');

  const rule = cols > 2 ? `<line x1="${x(2) - split / 2}" y1="${top - 2}"
    x2="${x(2) - split / 2}" y2="${top + ch + 2}" stroke="var(--rule)" stroke-width="1"/>` : '';

  return `<div class="bracket"><svg viewBox="0 0 ${w} ${h}" role="img"
      aria-label="${esc(name)}: ${cells.map(([l, p]) => `${l} ${pct(p, 1)}`).join(', ')}"
      preserveAspectRatio="xMidYMid meet">${rule}${body}</svg></div>`;
}

/* ================= command palette =================
   One keystroke to anywhere. `/` or Cmd/Ctrl-K opens it; typing filters
   clubs, matchweeks and pages; Enter goes. A dense data site is only fast
   if you can skip the navigation.                                        */
export function initPalette(teams) {
  const W = lg();
  /* Switching league from here lands on the same page, without the club or
     matchweek parameters that only meant something in the league you left. */
  const here = location.pathname.split('/').pop() || 'index.html';
  const swap = (slug) => {
    const u = new URL(here, `${location.origin}${ROOT}`);
    if (slug !== DEFAULT_LEAGUE) u.searchParams.set('lg', slug);
    return u.pathname + u.search;
  };

  /* Pages come from the same registry the masthead reads, so a page added to
     one is never missing from the other. Only the matches label is special: a
     cup's fixture list includes the knockout rounds that its forecast total
     does not count, so the number is left unsaid there. */
  const items = [
    ...PAGES.map((p) => ({
      label: (p.id === 'matches' && !W.isCup) ? `All ${W.nMatches} matches` : p.long,
      kind: p.site ? 'Europe' : 'Page',
      href: pageHref(p),
    })),
    ...teams.map((t) => ({ label: t.name, kind: 'Club', href: url(`team.html?t=${t.id}`), hint: t.short })),
    ...Array.from({ length: W.nWeeks }, (_, i) => ({
      label: `${W.roundWord} ${i + 1}`, kind: W.roundWord,
      href: url(`matches.html?mw=${i + 1}`) })),
    ...LEAGUES.filter((l) => l.ready && l.slug !== LG.slug).map((l) => ({
      label: `Switch to ${l.name}`, kind: 'League', href: swap(l.slug), hint: l.country || '' })),
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
    } else if (e.key === '?' && !typing) {
      e.preventDefault(); showShortcuts();
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

/* ================= tabs =================
   The feature build left `team.html` at eight stacked cards and `method.html` at
   ten screens: everything given equal weight, so nothing had any. This groups a
   page's cards without hiding its answer -- whatever sits above the strip stays
   unconditional, and only the depth below it is switched.

   Three properties it has to have, none optional:
     * the chosen tab is in the URL as a hash, so a tab is linkable and Back works;
     * every panel is shown when printing, since a tab is a screen affordance;
     * the strip is a real tablist, driven by arrow keys as well as clicks.

   `groups` is [{ id, label, sections: [element…] }]. Panels are wrapped rather
   than moved, so the page's own markup order still decides what is in each. */
export function initTabs(host, groups, { param = '' } = {}) {
  if (!host || groups.length < 2) return () => {};
  const wraps = groups.map((g) => {
    const div = document.createElement('div');
    div.id = `panel-${g.id}`;
    div.setAttribute('data-tabpanel', g.id);
    div.setAttribute('role', 'tabpanel');
    div.setAttribute('aria-labelledby', `tab-${g.id}`);
    g.sections.filter(Boolean).forEach((el) => div.appendChild(el));
    return div;
  });
  const anchor = document.createElement('div');
  anchor.className = 'tabstrip';
  anchor.setAttribute('role', 'tablist');
  anchor.innerHTML = groups.map((g) => `
    <button class="tab-btn" role="tab" id="tab-${esc(g.id)}"
            aria-controls="panel-${esc(g.id)}" data-tab="${esc(g.id)}"
            aria-selected="false" tabindex="-1">${esc(g.label)}</button>`).join('');
  host.appendChild(anchor);
  wraps.forEach((w) => host.appendChild(w));

  const btns = [...anchor.querySelectorAll('[data-tab]')];
  const show = (id, push) => {
    const found = groups.find((g) => g.id === id) ? id : groups[0].id;
    btns.forEach((b) => {
      const on = b.dataset.tab === found;
      b.setAttribute('aria-selected', String(on));
      b.tabIndex = on ? 0 : -1;
    });
    wraps.forEach((w) => { w.hidden = w.dataset.tabpanel !== found; });
    if (push) {
      const u = new URL(location.href);
      u.hash = found;
      history.replaceState({}, '', u.pathname + u.search + u.hash);
    }
    return found;
  };

  anchor.addEventListener('click', (e) => {
    const b = e.target.closest('[data-tab]');
    if (b) show(b.dataset.tab, true);
  });
  anchor.addEventListener('keydown', (e) => {
    const i = btns.indexOf(document.activeElement);
    if (i < 0) return;
    let j = null;
    if (e.key === 'ArrowRight') j = (i + 1) % btns.length;
    else if (e.key === 'ArrowLeft') j = (i - 1 + btns.length) % btns.length;
    else if (e.key === 'Home') j = 0;
    else if (e.key === 'End') j = btns.length - 1;
    if (j === null) return;
    e.preventDefault();
    btns[j].focus();
    show(btns[j].dataset.tab, true);
  });
  addEventListener('hashchange', () => show(location.hash.slice(1), false));

  const wanted = location.hash.slice(1)
    || (param && new URLSearchParams(location.search).get(param)) || groups[0].id;
  show(wanted, false);
  return show;
}

/* ================= following clubs =================
   Nine competitions and 174 clubs, and a reader almost always cares about one
   or two. The list is kept in this browser and never leaves it -- there is no
   server here to keep it on -- so it is small, forgiving of a cleared cache,
   and stores ids rather than names so a rename does not lose anybody. */
const FOLLOW_KEY = 'plf-follow';

export function followed() {
  try {
    const raw = JSON.parse(localStorage.getItem(FOLLOW_KEY) || '[]');
    return Array.isArray(raw) ? raw.filter((x) => typeof x === 'string') : [];
  } catch { return []; }
}

export const isFollowed = (id) => followed().includes(id);

/** Toggle one club and return whether it is now followed. */
export function toggleFollow(id) {
  const list = followed();
  const at = list.indexOf(id);
  if (at >= 0) list.splice(at, 1);
  else list.push(id);
  try { localStorage.setItem(FOLLOW_KEY, JSON.stringify(list)); } catch { /* private mode */ }
  dispatchEvent(new CustomEvent('followchange', { detail: { id, on: at < 0 } }));
  return at < 0;
}

/* The star, as one function, because three pages want the same control and a
   star that means "followed" on one page and "favourite" on another is worse
   than no star. `label` names the club so a screen reader hears which one. */
export function followButton(id, label) {
  const on = isFollowed(id);
  return `<button class="followbtn" data-follow="${esc(id)}" aria-pressed="${on}"
            title="${on ? 'Stop following' : 'Follow'} ${esc(label)}">
      <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true">
        <path d="M8 1.6l1.9 3.9 4.3.6-3.1 3 .7 4.3L8 11.4l-3.8 2 .7-4.3-3.1-3 4.3-.6z"
              fill="${on ? 'currentColor' : 'none'}" stroke="currentColor"
              stroke-width="1.4" stroke-linejoin="round"/>
      </svg><span>${on ? 'Following' : 'Follow'}</span>
    </button>`;
}

/* One delegated listener for the whole document: any star anywhere works, and
   every star for the same club updates together. */
addEventListener('click', (e) => {
  const btn = e.target.closest?.('[data-follow]');
  if (!btn) return;
  e.preventDefault();
  toggleFollow(btn.dataset.follow);
});
addEventListener('followchange', (e) => {
  document.querySelectorAll(`[data-follow="${CSS.escape(e.detail.id)}"]`)
    .forEach((b) => {
      b.setAttribute('aria-pressed', String(e.detail.on));
      const path = b.querySelector('path');
      if (path) path.setAttribute('fill', e.detail.on ? 'currentColor' : 'none');
      const txt = b.querySelector('span');
      if (txt) txt.textContent = e.detail.on ? 'Following' : 'Follow';
      const t = b.getAttribute('title') || '';
      b.setAttribute('title', e.detail.on
        ? t.replace(/^Follow /, 'Stop following ')
        : t.replace(/^Stop following /, 'Follow '));
    });
});

/* ================= copy a link =================
   Every view on this site is already in its URL. Saying so, with a button that
   puts it on the clipboard, is the difference between a shareable page and one
   a reader has to know is shareable. */
export function copyLink(btn, href = location.href) {
  const say = (msg) => {
    const was = btn.dataset.label || btn.textContent;
    btn.dataset.label = was;
    btn.textContent = msg;
    setTimeout(() => { btn.textContent = btn.dataset.label; }, 1600);
  };
  const fallback = () => {
    /* `navigator.clipboard` needs a secure context, which a file:// copy of
       this site is not. Select the URL instead so a reader can still copy it. */
    const ta = document.createElement('textarea');
    ta.value = href;
    ta.style.cssText = 'position:fixed;opacity:0';
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch { ok = false; }
    ta.remove();
    say(ok ? 'Copied' : 'Press ⌘C');
  };
  if (navigator.clipboard?.writeText) {
    navigator.clipboard.writeText(href).then(() => say('Copied'), fallback);
  } else fallback();
}

/* The site rewards keyboard use and advertised almost none of it. */
export function showShortcuts() {
  modal(`<h3>Keyboard shortcuts</h3>
    <p class="msub">Everything here works from any page.</p>
    <div id="tipish">
      ${[['/ or Ctrl-K', 'Search clubs, matchweeks and pages'],
         ['?', 'This list'],
         ['j / k', 'Move down and up a table, Enter to open'],
         ['← / →', 'Move between tabs when one is focused'],
         ['Esc', 'Close whatever is open'],
         ['Tab', 'Reach the sort headers; Enter sorts']]
        .map(([k, v]) => `<div class="r"><span>${esc(v)}</span><b>${esc(k)}</b></div>`).join('')}
    </div>`);
}

/* ================= row navigation =================
   j and k through a table's rows, Enter to follow the row's own link. Bound to
   a container rather than to each row, so a re-render never loses it. */
export function initRowKeys(container, { rowSel = 'tr[data-id]' } = {}) {
  if (!container) return;
  let at = -1;
  const rows = () => [...container.querySelectorAll(rowSel)];
  const mark = (list) => list.forEach((r, i) => r.classList.toggle('rowcursor', i === at));
  addEventListener('keydown', (e) => {
    if (/^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement?.tagName || '')) return;
    if (!document.getElementById('palette')?.hidden) return;
    const list = rows();
    if (!list.length) return;
    if (e.key === 'j' || e.key === 'k') {
      e.preventDefault();
      at = e.key === 'j' ? Math.min(at + 1, list.length - 1) : Math.max(at - 1, 0);
      mark(list);
      list[at].scrollIntoView({ block: 'nearest' });
    } else if (e.key === 'Enter' && at >= 0 && at < list.length) {
      const a = list[at].querySelector('a[href]');
      if (a) { e.preventDefault(); location.href = a.href; }
    }
  });
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
  box.querySelectorAll('[data-ics]').forEach((a) => a.addEventListener('click', (e) => {
    e.preventDefault();
    /* One fixture out of the competition's own feed, so the description is the
       same text a subscriber already gets and there is one format to maintain. */
    const [h, aw] = a.dataset.ics.split('|');
    window.location.href = `${ROOT}cal/${LG.slug}/${encodeURIComponent(h)}.ics`;
  }));
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

/* ================= a scoreline grid from published parameters =================
   The pipeline deliberately ships fitted expectations rather than ratings, so
   the browser can never drift away from the published forecast by reimplementing
   the model (see model/siminput.py). This is the one exception, and it is not a
   reimplementation: the global ranking publishes each club's attack and defence
   plus the fit's intercept, home term and rho, and lambda_home is
   `off_h x def_a x exp(home - mu)`. That is arithmetic on numbers already on the
   page, which is what makes "these two clubs, on neutral ground" answerable at
   all — there is no fixture between them for the pipeline to have precomputed. */
const _lgam = (() => {
  const g = [0];
  for (let k = 1; k <= 12; k++) g[k] = g[k - 1] + Math.log(k);
  return g;
})();

function _tau(h, a, lh, la, rho) {
  if (h === 0 && a === 0) return Math.max(1 - lh * la * rho, 1e-9);
  if (h === 0 && a === 1) return Math.max(1 + lh * rho, 1e-9);
  if (h === 1 && a === 0) return Math.max(1 + la * rho, 1e-9);
  if (h === 1 && a === 1) return Math.max(1 - rho, 1e-9);
  return 1;
}

export function dcGrid(lh, la, rho, maxg = 6) {
  const n = maxg + 1;
  const ph = [], pa = [];
  for (let k = 0; k < n; k++) {
    ph[k] = Math.exp(-lh + k * Math.log(lh) - _lgam[k]);
    pa[k] = Math.exp(-la + k * Math.log(la) - _lgam[k]);
  }
  const g = [];
  let s = 0;
  for (let h = 0; h < n; h++) {
    g[h] = [];
    for (let a = 0; a < n; a++) {
      const v = ph[h] * pa[a] * _tau(h, a, lh, la, rho);
      g[h][a] = v;
      s += v;
    }
  }
  for (let h = 0; h < n; h++) for (let a = 0; a < n; a++) g[h][a] /= s;
  return g;
}

export function dcOutcome(g) {
  let h = 0, d = 0, a = 0;
  g.forEach((row, i) => row.forEach((p, j) => {
    if (i > j) h += p; else if (i === j) d += p; else a += p;
  }));
  return [h, d, a];
}

/* Both clubs' lambdas for a neutral-ground match, from a global.json row. */
export function neutralLambdas(A, B, g) {
  const k = Math.exp(-g.mu);
  return [A.off * B.def * k, B.off * A.def * k];
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
      /* No tabindex. 121 focus stops inside a dialog is a trap, not access:
         the whole grid is one labelled figure below, and the numbers are also
         written out in the caption for a reader that cannot hover. */
      return `<div class="gcell${isBest ? ' best' : ''}"
        style="background:var(--ramp-${step});color:var(--ink-${step})"
        title="${homeShort} ${h}–${a} ${awayShort} · ${(p * 100).toFixed(1)}%"
        >${p >= 0.04 ? Math.round(p * 100) : ''}</div>`;
    }).join('')).join('');

  /* Spoken form of the same picture: the six likeliest scorelines, in words,
     so the grid is not the only way to read it. */
  const flat = [];
  grid.forEach((row, h) => row.forEach((p, a) => flat.push([h, a, p])));
  flat.sort((x, y) => y[2] - x[2]);
  const spoken = flat.slice(0, 6)
    .map(([h, a, p]) => `${homeShort} ${h}–${a} ${awayShort} ${(p * 100).toFixed(1)}%`)
    .join(', ');

  return `
    <div class="gwrap">
      <div class="gaxis-y">${esc(homeShort)} goals</div>
      <div class="gmain">
        <div class="ggrid" style="--n:${n}" role="img" tabindex="0"
             aria-label="Scoreline probabilities. Likeliest: ${esc(spoken)}."
             >${head}${body}</div>
        <div class="gaxis-x">${esc(awayShort)} goals</div>
      </div>
    </div>
    <p class="hint gspoken">Likeliest scorelines: ${esc(spoken)}.</p>`;
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

/* ================= rating timeline =================
   One club, every season on record. The old sparkline showed four points with
   no axis, which is a shape rather than a history; this is the same series with
   the years written under it and the peak and trough called out, because the
   question people actually ask of a rating chart is "when were they good".  */
export function timeline(points, { w = 860, h = 240, pad = 38, color = 'var(--accent)' } = {}) {
  if (!points || points.length < 2) return '';
  const vals = points.map((p) => p.spi);
  const lo = Math.floor(Math.min(...vals) / 5) * 5;
  const hi = Math.ceil(Math.max(...vals) / 5) * 5;
  const span = (hi - lo) || 1;
  const x = (i) => pad + (i * (w - pad - 16)) / (points.length - 1);
  const y = (v) => h - pad - ((v - lo) / span) * (h - pad - 20);
  const d = points.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)} ${y(p.spi).toFixed(1)}`).join(' ');
  const area = `${d} L${x(points.length - 1).toFixed(1)} ${h - pad} L${x(0).toFixed(1)} ${h - pad} Z`;

  const ticks = [lo, lo + span / 2, hi];
  const iMax = vals.indexOf(Math.max(...vals));
  const iMin = vals.indexOf(Math.min(...vals));
  const last = points.length - 1;
  /* Label roughly six seasons, always including the first and the last. */
  const every = Math.max(1, Math.round(points.length / 6));
  const xlab = points.map((p, i) =>
    (i === 0 || i === last || i % every === 0)
      ? `<text class="tl-axis" x="${x(i).toFixed(1)}" y="${h - pad + 16}"
              text-anchor="${i === 0 ? 'start' : (i === last ? 'end' : 'middle')}"
              >${esc(p.season.slice(2))}</text>` : '').join('');

  const mark = (i, anchor) => (i === last ? '' : `
    <circle cx="${x(i).toFixed(1)}" cy="${y(vals[i]).toFixed(1)}" r="3" fill="${color}"/>
    <text class="tl-peak" x="${x(i).toFixed(1)}" y="${(y(vals[i]) + (anchor === 'top' ? -9 : 17)).toFixed(1)}"
          text-anchor="middle">${vals[i].toFixed(1)}</text>`);

  return `<div class="timeline"><svg viewBox="0 0 ${w} ${h}" role="img"
      aria-label="Rating by season from ${esc(points[0].season)} to ${esc(points[last].season)}, ${
        points.map((p) => `${p.season} ${p.spi}`).join('; ')}">
    ${ticks.map((t) => `
      <line x1="${pad}" y1="${y(t)}" x2="${w - 8}" y2="${y(t)}" stroke="var(--grid)" stroke-width="1"/>
      <text class="tl-axis" x="${pad - 6}" y="${y(t) + 4}" text-anchor="end">${t}</text>`).join('')}
    <path d="${area}" fill="${color}" opacity="0.10"/>
    <path d="${d}" fill="none" stroke="${color}" stroke-width="2.2"
          stroke-linejoin="round" stroke-linecap="round"/>
    ${mark(iMax, 'top')}${mark(iMin, 'bottom')}
    <circle cx="${x(last).toFixed(1)}" cy="${y(vals[last]).toFixed(1)}" r="4.5"
            fill="${color}" stroke="var(--surface)" stroke-width="2"/>
    <text class="tl-peak" x="${x(last).toFixed(1)}" y="${(y(vals[last]) - 11).toFixed(1)}"
          text-anchor="end">${vals[last].toFixed(1)}</text>
    ${xlab}
  </svg></div>`;
}

/* ================= multi-series line chart =================
   Used for the forecast's own movement over time. Series are direct-labelled
   at their endpoint, so identity never rests on colour alone.             */
/* Two clubs' colours, guaranteed to be told apart.

   Half the clubs in Europe play in red. Drawing Arsenal against Bayern in their
   own colours produces two red lines, which is worse than useless on a chart
   whose entire job is comparison. So the pair is checked first, and when the two
   are too close the chart falls back to the colours this site already uses for
   the two sides of a match: the accent for the first club, the away red for the
   second. A reader who has seen one fixture bar on this site already knows which
   is which. */
export function distinctPair(a, b) {
  const ca = chipColor(a), cb = chipColor(b);
  const rgbA = _hex(ca), rgbB = _hex(cb);
  if (rgbA && rgbB) {
    /* Distance in plain RGB is a poor perceptual measure but a fine "are these
       obviously different" one, and it needs no colour-space maths. Below a
       sixth of the diagonal, two lines read as the same line. */
    const d = Math.sqrt(rgbA.reduce((t, v, i) => t + (v - rgbB[i]) ** 2, 0));
    if (d > 442 / 6) return [ca, cb];
  }
  return ['var(--accent)', 'var(--away)'];
}

/* The eight ratings, in the order they are drawn round a radar and listed in a
   table. Order matters: attack and creation sit next to each other because they
   are the pair a reader should compare, and defence next to discipline for the
   same reason. */
export const DIMS = [
  { key: 'att_r', label: 'Attack',
    hint: 'Goals against an average opponent' },
  { key: 'creation_r', label: 'Creation',
    hint: 'Shots on target per match. Big five only — no other feed has a shot in it' },
  { key: 'finishing_r', label: 'Finishing',
    hint: 'Goals per shot on target. Big five only' },
  { key: 'big_r', label: 'Big games',
    hint: 'Points per game against the top quarter of the division' },
  { key: 'home_r', label: 'Home',
    hint: 'Points per game at home minus points per game away' },
  { key: 'consistency_r', label: 'Consistency',
    hint: 'How little the goal difference moves match to match. Predictable, which is not the same as good' },
  { key: 'discipline_r', label: 'Discipline',
    hint: 'Cards and fouls, inverted — a high rating is a clean side. Big five only' },
  { key: 'def_r', label: 'Defence',
    hint: 'Goals conceded against an average opponent' },
];

/* A radar of the ratings, for one club or two.

   Drawn on 35-95, the band the ratings themselves live on, with a ring at the
   competition average so a shape can be read against something rather than
   admired in the abstract. An axis is dropped entirely when neither club has
   it -- four of the nine competitions have no shot feed, and an axis pinned at
   the middle for want of data reads as "average", which is a claim. */
export function radar(clubs, { size = 300, mid = 65 } = {}) {
  const have = DIMS.filter((d) => clubs.some((c) => c.values[d.key] != null));
  if (have.length < 3) return '';
  /* The frame is wider than the chart. "Consistency" set outside the leftmost
     spoke needs about sixty pixels of its own, and a square viewBox clipped it
     to "onsistency" -- so the box gains room either side rather than the radar
     shrinking to make space for its own labels. */
  const W = size + 150, H = size + 16;
  const cx = W / 2, cy = H / 2;
  const r = size / 2 - 22;
  const LO = 35, HI = 95;
  const at = (i, v) => {
    const a = (i / have.length) * 2 * Math.PI - Math.PI / 2;
    const k = Math.max(0, Math.min(1, (v - LO) / (HI - LO)));
    return [cx + Math.cos(a) * r * k, cy + Math.sin(a) * r * k];
  };
  const ring = (v, extra = '') => `<polygon points="${
    have.map((_, i) => at(i, v).map((n) => n.toFixed(1)).join(',')).join(' ')}"
    fill="none" stroke="var(--grid)" stroke-width="1" ${extra}/>`;
  const label = (d, i) => {
    const a = (i / have.length) * 2 * Math.PI - Math.PI / 2;
    const lx = cx + Math.cos(a) * (r + 16);
    const ly = cy + Math.sin(a) * (r + 16);
    const anchor = Math.abs(Math.cos(a)) < 0.3 ? 'middle'
      : (Math.cos(a) > 0 ? 'start' : 'end');
    return `<text class="rd-lab" x="${lx.toFixed(1)}" y="${(ly + 4).toFixed(1)}"
      text-anchor="${anchor}">${esc(d.label)}</text>`;
  };
  return `<div class="radar"><svg viewBox="0 0 ${W} ${H}" role="img"
      aria-label="Rating radar: ${have.map((d) => esc(d.label)).join(', ')}">
    ${[LO + 15, mid, HI - 10].map((v) => ring(v)).join('')}
    ${ring(mid, 'stroke="var(--rule)" stroke-dasharray="3 3"')}
    ${have.map((_, i) => {
      const [ex, ey] = at(i, HI);
      return `<line x1="${cx}" y1="${cy}" x2="${ex.toFixed(1)}" y2="${ey.toFixed(1)}"
        stroke="var(--grid)" stroke-width="1"/>`;
    }).join('')}
    ${clubs.map((c) => {
      /* A club missing one axis is drawn on the rest rather than dropped: the
         shape is still the truth about the axes it does have. */
      const pts = have.map((d, i) => (c.values[d.key] == null ? null : at(i, c.values[d.key])))
        .filter(Boolean);
      if (pts.length < 3) return '';
      return `<polygon points="${pts.map((p) => p.map((n) => n.toFixed(1)).join(',')).join(' ')}"
        fill="${c.color}" fill-opacity="${clubs.length > 1 ? 0.16 : 0.22}"
        stroke="${c.color}" stroke-width="2" stroke-linejoin="round"/>`;
    }).join('')}
    ${have.map(label).join('')}
  </svg></div>`;
}

/* Two clubs' rating histories on one pair of axes.

   They used to be two charts side by side, each on its own vertical scale, on
   the grounds that a shared axis running from zero would squash both series
   into the top of the frame. That is true of an axis from zero and is the wrong
   fix: a comparison chart whose two halves have different scales invites
   exactly the reading it cannot support, which is comparing the shapes. So the
   axis is shared and does not start at zero -- it covers what the two clubs
   actually did, and nothing else.

   The two need not cover the same seasons. They are aligned on the union, and a
   season a club spent in another division is a break in its line rather than a
   straight segment drawn across a gap it was not there for. */
export function spiCompare(a, b, { w = 860, h = 300, pad = 40 } = {}) {
  const series = [a, b].filter((s) => s && s.points && s.points.length);
  if (!series.length) return '';
  const seasons = [...new Set(series.flatMap((s) => s.points.map((p) => p.season)))].sort();
  if (seasons.length < 2) return '';
  const at = series.map((s) => {
    const by = new Map(s.points.map((p) => [p.season, p.spi]));
    return seasons.map((k) => (by.has(k) ? by.get(k) : null));
  });
  const vals = at.flat().filter((v) => v != null);
  const lo = Math.floor(Math.min(...vals) / 5) * 5;
  const hi = Math.ceil(Math.max(...vals) / 5) * 5;
  const span = (hi - lo) || 1;
  const x = (i) => pad + (i * (w - pad - 12)) / Math.max(seasons.length - 1, 1);
  const y = (v) => h - pad - ((v - lo) / span) * (h - pad - 22);

  /* One `path` per contiguous run, so a gap stays a gap. */
  const runs = (pts) => {
    const out = [];
    let cur = [];
    pts.forEach((v, i) => {
      if (v == null) { if (cur.length) out.push(cur); cur = []; }
      else cur.push([i, v]);
    });
    if (cur.length) out.push(cur);
    return out;
  };

  const ticks = [lo, lo + span / 4, lo + span / 2, lo + (3 * span) / 4, hi];
  const last = seasons.length - 1;
  const every = Math.max(1, Math.round(seasons.length / 8));
  return `<div class="timeline"><svg viewBox="0 0 ${w} ${h}" role="img"
      aria-label="Rating by season, ${series.map((s) => esc(s.label)).join(' against ')}, ${
        esc(seasons[0])} to ${esc(seasons[last])}">
    ${ticks.map((t) => `
      <line x1="${pad}" y1="${y(t).toFixed(1)}" x2="${w - 12}" y2="${y(t).toFixed(1)}"
            stroke="var(--grid)" stroke-width="1"/>
      <text class="tl-axis" x="${pad - 7}" y="${(y(t) + 4).toFixed(1)}"
            text-anchor="end">${t.toFixed(0)}</text>`).join('')}
    ${seasons.map((k, i) => (i === 0 || i === last || i % every === 0)
      ? `<text class="tl-axis" x="${x(i).toFixed(1)}" y="${h - pad + 16}"
              text-anchor="${i === 0 ? 'start' : (i === last ? 'end' : 'middle')}"
              >${esc(k.slice(2))}</text>` : '').join('')}
    ${series.map((s, si) => runs(at[si]).map((run) => `
      <path d="${run.map(([i, v], k) => `${k ? 'L' : 'M'}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(' ')}"
            fill="none" stroke="${s.color}" stroke-width="2.4"
            stroke-linejoin="round" stroke-linecap="round"/>
      ${run.length === 1 ? `<circle cx="${x(run[0][0]).toFixed(1)}"
            cy="${y(run[0][1]).toFixed(1)}" r="3" fill="${s.color}"/>` : ''}`).join('')
      + (() => {
        const pts = at[si];
        let i = pts.length - 1;
        while (i >= 0 && pts[i] == null) i -= 1;
        return i < 0 ? '' : `
          <circle cx="${x(i).toFixed(1)}" cy="${y(pts[i]).toFixed(1)}" r="4"
                  fill="${s.color}" stroke="var(--surface)" stroke-width="2"/>`;
      })()).join('')}
  </svg>
  <div class="tl-key">${series.map((s) => `
    <span><i style="background:${s.color}"></i>${esc(s.label)}</span>`).join('')}</div>
  </div>`;
}

/* Only this season's snapshots, newest last.

   `history.json` keeps 400 days, which is more than a season, and a snapshot
   carries the season it belongs to precisely so a chart does not splice one
   title race onto the next. An older file has no season stamp; there is nothing
   to filter on, so it is used whole rather than thrown away. */
export function thisSeason(snaps, season) {
  if (!snaps || !snaps.length) return [];
  const tagged = snaps.filter((s) => s.season);
  if (!tagged.length || !season) return snaps;
  const mine = tagged.filter((s) => s.season === season);
  return mine.length ? mine : snaps;
}

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
    ${series.map((s) => {
      /* A missing point is a gap in the line, never a zero on it. A club that
         was not in this division for part of the window has no probability for
         those days, and drawing one at the floor is a claim the data does not
         make -- it reads as "the model gave them no chance". */
      const runs = [];
      let cur = [];
      s.points.forEach((p, i) => {
        if (p == null) { if (cur.length) runs.push(cur); cur = []; }
        else cur.push([i, p]);
      });
      if (cur.length) runs.push(cur);
      if (!runs.length) return '';
      const tail = runs[runs.length - 1];
      const [li, lv] = tail[tail.length - 1];
      return runs.map((run) => `
        <polyline points="${run.map(([i, p]) => `${x(i).toFixed(1)},${y(p).toFixed(1)}`).join(' ')}"
          fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`).join('')
      + `<circle cx="${x(li).toFixed(1)}" cy="${y(lv).toFixed(1)}" r="3.5"
          fill="${s.color}" stroke="var(--surface)" stroke-width="2"/>
        <text x="${w - 86}" y="${y(lv) + 4}" fill="var(--ink2)" font-size="11">${esc(s.label)}</text>`;
    }).join('')}
  </svg>`;
}

/* ================= half time =================
   The results feed has carried a half-time score in every row since 2000-01 and
   the forecast never read it. A second Dixon-Coles fit on those goals answers a
   question the full-time model cannot: who is likely to be ahead at the break,
   and how often that survives. Shown as a bar rather than a second 7x7 grid —
   the interval is a smaller question than the result and should take less room.

   Absent for a competition whose feed has no half-time scores, in which case
   nothing is drawn rather than something being guessed. */
export function halfTime(m, h, a) {
  if (!Array.isArray(m.ht)) return '';
  const [ph, pd, pa] = m.ht;
  return `<h4>At half time</h4>
    <div class="wdl" style="height:20px;margin:8px 0 6px">
      <i class="h" style="flex:${Math.max(ph, .02)}">${ph >= .16 ? `<b>${Math.round(ph * 100)}</b>` : ''}</i>
      <i class="d" style="flex:${Math.max(pd, .02)}">${pd >= .16 ? `<b>${Math.round(pd * 100)}</b>` : ''}</i>
      <i class="a" style="flex:${Math.max(pa, .02)}">${pa >= .16 ? `<b>${Math.round(pa * 100)}</b>` : ''}</i>
    </div>
    <div class="hint" style="margin-bottom:20px">${esc(h.short)} ahead ${pct(ph)} ·
      level ${pct(pd)} · ${esc(a.short)} ahead ${pct(pa)}${
      m.htsc ? ` &nbsp;·&nbsp; likeliest half-time score ${m.htsc[0]}–${m.htsc[1]} (${pct(m.htscp)})` : ''}
      &nbsp;·&nbsp; fitted on the half-time scores in the results feed.</div>`;
}

/* ================= match dialog =================
   One implementation shared by every page that can open a match. `meta` is the
   forecast's team map; `opts.link` adds a permalink to the matches page. */
export function matchModal(m, meta, opts = {}) {
  const EVENT = eventLabels();
  const h = meta[m.h], a = meta[m.a];
  const fmtDay = (iso) => new Date(iso + 'T12:00:00').toLocaleDateString('en-GB',
    { weekday: 'short', day: 'numeric', month: 'short' });
  const swings = (m.swings || []).map((s) => `
    <div class="swing-row">
      <span>${esc(meta[s.team].name)}: ${EVENT[s.event]}</span>
      ${swingBar(s.away, s.home, meta[s.team].name + ' ' + EVENT[s.event])}
      <b>${pct(s.away)} → ${pct(s.home)}</b>
    </div>`).join('');
  tip(null);
  modal(`
    <h3>${esc(h.name)} v ${esc(a.name)}</h3>
    <p class="msub">${esc(roundLabel(m.md, m.leg))} · ${fmtDay(m.date)}${m.time ? ` · ${m.time}` : ''}
      ${m.played ? ` · finished ${m.hg}–${m.ag}` : ''}
      ${opts.link === false ? '' :
        ` · <a href="${url(`matches.html?m=${m.h}--${m.a}`)}">link to this match</a>`}
      ${m.played ? '' : ` · <a href="#" data-ics="${esc(m.h)}|${esc(m.a)}">add to calendar</a>`}</p>
    <div class="wdl" style="height:26px;margin-bottom:6px">
      <i class="h" style="flex:${Math.max(m.ph, .02)}"><b>${Math.round(m.ph * 100)}</b></i>
      <i class="d" style="flex:${Math.max(m.pd, .02)}"><b>${Math.round(m.pd * 100)}</b></i>
      <i class="a" style="flex:${Math.max(m.pa, .02)}"><b>${Math.round(m.pa * 100)}</b></i>
    </div>
    <div class="hint" style="margin-bottom:20px">
      ${esc(h.short)} win · draw · ${esc(a.short)} win &nbsp;·&nbsp;
      expected goals ${m.xgh} – ${m.xga} &nbsp;·&nbsp; over 2.5 goals ${pct(m.o25)}
      &nbsp;·&nbsp; both score ${pct(m.btts)}
      ${typeof m.csh === 'number' ? `&nbsp;·&nbsp; clean sheet
        ${esc(h.short)} ${pct(m.csh)} / ${esc(a.short)} ${pct(m.csa)}` : ''}</div>
    ${halfTime(m, h, a)}
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
