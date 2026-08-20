/* Offline support, deliberately the cautious kind.

   The forecast is rebuilt every six hours, so a cache-first worker would
   happily serve April's numbers in May — which is worse than an error page,
   because it looks fine. So: network first, and the cache is only ever a
   fallback for a request that failed. A reader offline gets the last numbers
   they actually saw; a reader online always gets today's.

   Nothing here is precached at install time either. Precaching a shell means
   guessing which pages matter and pinning a version of them; caching what was
   actually fetched means the app works offline for exactly the competitions
   somebody has looked at, which is the honest scope of "offline support" for a
   site with eight of them.                                                  */

const CACHE = '537-v2';

/* Same-origin GETs only. A cross-origin request has nothing to do with this
   site's data and should not be quietly served from its cache. */
const cacheable = (req) => {
  if (req.method !== 'GET') return false;
  const u = new URL(req.url);
  if (u.origin !== self.location.origin) return false;
  /* Share cards are large and are only ever read by link scrapers, which do
     not run service workers. No point spending a reader's storage on them. */
  return !u.pathname.includes('/og/');
};

self.addEventListener('install', (e) => {
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    const names = await caches.keys();
    await Promise.all(names.filter((n) => n !== CACHE).map((n) => caches.delete(n)));
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (e) => {
  if (!cacheable(e.request)) return;
  e.respondWith((async () => {
    try {
      const fresh = await fetch(e.request);
      if (fresh && fresh.ok) {
        const cache = await caches.open(CACHE);
        cache.put(e.request, fresh.clone());
      }
      return fresh;
    } catch (err) {
      const hit = await caches.match(e.request, { ignoreSearch: false });
      if (hit) return hit;
      /* A page navigation with nothing cached: fall back to whatever copy of
         the front page we have rather than the browser's offline dinosaur. */
      if (e.request.mode === 'navigate') {
        const home = await caches.match(new URL('index.html', self.registration.scope).href);
        if (home) return home;
      }
      throw err;
    }
  })());
});
