"""Share cards: the picture behind a shared link.

Every page on this site already declares `twitter:card = summary_large_image`
and, until now, shipped no image at all — so every link anyone has ever posted
rendered as a blank grey box. This draws the missing asset: a 1200x630 PNG per
competition and per club, generated from `forecast.json` at build time.

Drawn with Pillow rather than rendered from SVG or HTML, because the two
obvious alternatives both break a constraint: rasterising SVG needs cairo, and
screenshotting HTML needs a headless browser. Pillow plus a font that is already
on the runner is the whole dependency, and if either is missing the build prints
a line and carries on without cards rather than failing.

The output goes to `site/og/`, which is git-ignored: the numbers move every six
hours, the Pages artefact is built from the working tree rather than from what
is committed, and nobody needs five megabytes of PNG churn in the history.
"""
from __future__ import annotations

import datetime as dt
import os

W, H = 1200, 630
PAD = 64

#: Matches the site's own dark theme, which is what the cards should look like
#: whichever theme the reader happens to use — a share card has no reader yet.
BG = (17, 17, 16)
PANEL = (26, 26, 25)
INK = (247, 247, 245)
INK2 = (176, 176, 168)
MUTED = (124, 124, 116)
ACCENT = (57, 135, 229)
RULE = (48, 48, 46)

FONT_DIRS = (
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/dejavu",
    "/usr/share/fonts/TTF",
    "/Library/Fonts",
)
FONT_FILES = {"regular": "DejaVuSans.ttf", "bold": "DejaVuSans-Bold.ttf"}


class Unavailable(RuntimeError):
    """Pillow or a usable font is missing. Cards are a nice-to-have; a build
    that cannot draw them still has to produce a forecast."""


def _fonts():
    try:
        from PIL import ImageFont
    except ImportError as exc:                       # pragma: no cover
        raise Unavailable(f"Pillow not installed ({exc})") from exc
    found = {}
    for key, name in FONT_FILES.items():
        for d in FONT_DIRS:
            p = os.path.join(d, name)
            if os.path.exists(p):
                found[key] = p
                break
    if len(found) != len(FONT_FILES):
        raise Unavailable("DejaVu fonts not found on this machine")

    def at(key, size):
        return ImageFont.truetype(found[key], size)
    return at


def _hex(h: str) -> tuple[int, int, int]:
    v = (h or "#7A8290").lstrip("#")
    if len(v) == 3:
        v = "".join(c * 2 for c in v)
    try:
        n = int(v, 16)
    except ValueError:
        n = 0x7A8290
    return ((n >> 16) & 255, (n >> 8) & 255, n & 255)


def _readable(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Nudge a club colour toward the ink until it is visible on the panel.

    The same idea as `chipColor` in the front end, and for the same reason:
    Fulham's black and Leeds' yellow are each invisible against one surface.
    """
    def lum(c):
        def f(x):
            x /= 255
            return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4
        return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2])

    out = list(rgb)
    for _ in range(12):
        a, b = sorted((lum(out), lum(PANEL)), reverse=True)
        if (a + 0.05) / (b + 0.05) >= 2.6:
            break
        out = [round(v + (255 - v) * 0.18) for v in out]
    return tuple(out)


def _pct(x) -> str:
    x = float(x or 0)
    if x >= 0.9995:
        return ">99%"
    if 0 < x < 0.005:
        return "<1%"
    return f"{x * 100:.0f}%"


def _ord(n: int) -> str:
    """1st, 2nd, 3rd, 4th ... 11th, 21st, 24th. The teens are the special case,
    not the twenties -- getting that backwards is how a 24-club division walks
    off the end of a four-element list."""
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def _base(at):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 8], fill=ACCENT)
    return img, d


def _footer(d, at, right: str):
    d.line([(PAD, H - 96), (W - PAD, H - 96)], fill=RULE, width=1)
    d.text((PAD, H - 74), "chraltro.github.io/537", font=at("bold", 22), fill=INK2)
    f = at("regular", 20)
    d.text((W - PAD - d.textlength(right, font=f), H - 72), right,
           font=f, fill=MUTED)


def _clip(d, text: str, font, width: float) -> str:
    """Trim a name to fit, with an ellipsis, rather than letting it overrun."""
    if d.textlength(text, font=font) <= width:
        return text
    while text and d.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text + "…"


def league_card(fc: dict, words: dict) -> "object":
    """Top of the projected table, plus the one-sentence answer."""
    at = _fonts()
    img, d = _base(at)
    rows = fc.get("teams", [])[:6]
    lead = max(fc.get("teams", []), key=lambda t: t.get("title", 0), default=None)

    name = fc.get("league", {}).get("name", "Football")
    f = at("bold", 52)
    d.text((PAD, 48), _clip(d, words.get("question", f"Who wins {name}?"), f,
                            W - PAD * 2), font=f, fill=INK)
    who = (f"{lead['name']} favourite at {_pct(lead.get('title'))} · "
           if lead is not None else "")
    d.text((PAD, 114),
           who + f"{fc.get('matches_played', 0)} of {fc.get('matches_total', 0)} "
                 f"played · {fc.get('n_sims', 0):,} simulated seasons",
           font=at("regular", 23), fill=MUTED)

    top, rh = 206, 52
    d.rounded_rectangle([PAD - 18, 162, W - PAD + 18, top + rh * len(rows) + 4],
                        radius=14, fill=PANEL)
    hdr = at("bold", 19)
    d.text((PAD + 46, 176), "CLUB", font=hdr, fill=MUTED)
    for label, x in ((words.get("win", "TITLE"), 758), (words.get("top", "TOP"), 918),
                     ("PTS", 1058)):
        d.text((x, 176), label, font=hdr, fill=MUTED)
    y = top
    for i, t in enumerate(rows):
        d.text((PAD, y + 6), str(i + 1), font=at("regular", 24), fill=MUTED)
        col = _readable(_hex(t.get("primary", "")))
        d.rounded_rectangle([PAD + 32, y + 4, PAD + 40, y + 30], radius=4, fill=col)
        f = at("bold", 28)
        d.text((PAD + 56, y), _clip(d, t.get("name", t["id"]), f, 620),
               font=f, fill=INK)
        fv = at("bold", 26)
        d.text((758, y + 1), _pct(t.get("title")), font=fv, fill=INK)
        d.text((918, y + 1), _pct(t.get("ucl")), font=fv, fill=INK2)
        d.text((1058, y + 1), f"{t.get('pts', 0):.0f}", font=fv, fill=INK2)
        y += rh
    _footer(d, at, _updated(fc))
    return img


def club_card(fc: dict, team: dict, rank: int, words: dict) -> "object":
    """One club: the rating, the projection and the three headline chances."""
    at = _fonts()
    img, d = _base(at)
    col = _readable(_hex(team.get("primary", "")))

    d.rounded_rectangle([PAD, 92, PAD + 14, 92 + 76], radius=6, fill=col)
    f = at("bold", 62)
    d.text((PAD + 40, 88), _clip(d, team.get("name", team["id"]), f, W - PAD * 2 - 60),
           font=f, fill=INK)
    lg = fc.get("league", {}).get("name", "")
    d.text((PAD + 40, 168), f"{lg} · {fc.get('season', '')}",
           font=at("regular", 26), fill=MUTED)

    cards = [
        (words.get("finish", "Projected finish"), _ord(rank),
         f"{team.get('pts', 0):.0f} pts · 80% land {team.get('pts_lo')}–{team.get('pts_hi')}"),
        (words.get("win", "Title"), _pct(team.get("title")), "across every simulation"),
        (words.get("top", "Qualify"), _pct(team.get("ucl")), words.get("topnote", "")),
        (words.get("down", "Relegated"), _pct(team.get("releg")),
         f"SPI {team.get('spi', 0):.1f}"),
    ]
    x = PAD
    cw = (W - PAD * 2 - 3 * 18) / 4
    for label, value, note in cards:
        d.rounded_rectangle([x, 240, x + cw, 408], radius=14, fill=PANEL)
        d.text((x + 22, 260), label.upper(), font=at("bold", 18), fill=MUTED)
        d.text((x + 22, 294), value, font=at("bold", 52), fill=INK)
        fn = at("regular", 17)
        d.text((x + 22, 366), _clip(d, note, fn, cw - 44), font=fn, fill=MUTED)
        x += cw + 18

    pos = team.get("pos") or []
    if pos:
        bw = (W - PAD * 2) / len(pos)
        mx = max(pos) or 1
        for i, p in enumerate(pos):
            hgt = max(2, (p / mx) * 62)
            d.rectangle([PAD + i * bw + 1, 494 - hgt, PAD + (i + 1) * bw - 2, 494],
                        fill=col if i == rank - 1 else RULE)
        d.text((PAD, 500), words.get("posnote", "Chance of finishing in each place"),
               font=at("regular", 16), fill=MUTED)
    _footer(d, at, _updated(fc))
    return img


def global_card(g: dict) -> "object":
    """The cross-league ranking's own card.

    Rankings and Compare both pointed `og:image` at the Premier League table, so
    sharing "every club in Europe" showed one league's projected table. This is
    the top of the actual ranking, on the actual scale.
    """
    at = _fonts()
    img, d = _base(at)
    rows = g.get("clubs", [])[:6]
    f = at("bold", 52)
    d.text((PAD, 48), "Every club in Europe, one scale", font=f, fill=INK)
    d.text((PAD, 114),
           f"{g.get('n_clubs', 0)} clubs · {g.get('n_leagues', 0)} leagues · "
           f"{g.get('n_matches', 0):,} matches in one fit",
           font=at("regular", 23), fill=MUTED)

    top, rh = 206, 52
    d.rounded_rectangle([PAD - 18, 162, W - PAD + 18, top + rh * len(rows) + 4],
                        radius=14, fill=PANEL)
    hdr = at("bold", 19)
    d.text((PAD + 46, 176), "CLUB", font=hdr, fill=MUTED)
    d.text((760, 176), "LEAGUE", font=hdr, fill=MUTED)
    d.text((1058, 176), "SPI", font=hdr, fill=MUTED)
    y = top
    for i, t in enumerate(rows):
        d.text((PAD, y + 6), str(i + 1), font=at("regular", 24), fill=MUTED)
        col = _readable(_hex(t.get("primary", "")))
        d.rounded_rectangle([PAD + 32, y + 4, PAD + 40, y + 30], radius=4, fill=col)
        fb = at("bold", 28)
        d.text((PAD + 56, y), _clip(d, t.get("name", t["id"]), fb, 620),
               font=fb, fill=INK)
        fn = at("regular", 21)
        d.text((760, y + 4), _clip(d, t.get("league", ""), fn, 280), font=fn, fill=INK2)
        d.text((1058, y + 1), f"{t.get('spi', 0):.1f}", font=at("bold", 26), fill=INK)
        y += rh
    _footer(d, at, f"Fitted to {g.get('asof', '')}")
    return img


def _updated(fc: dict) -> str:
    try:
        g = dt.datetime.fromisoformat(fc["generated"])
        return f"Updated {g:%-d %b %Y}"
    except (KeyError, ValueError, TypeError):
        return "Open forecast"


def save(img, path: str) -> bool:
    """Write the PNG, but only when the bytes actually changed."""
    import io
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    data = buf.getvalue()
    if os.path.exists(path):
        with open(path, "rb") as fh:
            if fh.read() == data:
                return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return True
