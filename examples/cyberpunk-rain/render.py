"""Procedural cyberpunk rain scene: a detective in a long coat and his
holographic AI companion on a wet rooftop, in a neon megacity at night.
They walk in, stop, and she holds an umbrella over him while she stands in the
rain herself; every drop that passes through her makes the hologram spark and
flicker. The camera pushes in, cuts to close-ups, then pulls back to wide as
they walk away together.

Inspired by the mood of Blade Runner 2049. Pure numpy + Pillow, encoded with
ffmpeg. Techniques follow the skills found via `skills find`:
  - anthropics/skills@algorithmic-art  -> seeded, layered generative city
  - bbeierle12/skill-mcp-claude@postfx-bloom -> threshold/blur/composite bloom
  - digitalsamba/claude-code-video-toolkit@ffmpeg -> yuv420p + faststart encode

Usage: python3 render.py [out.mp4] [--preview SECONDS]
"""

import math
import os
import subprocess
import sys
import wave
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1280, 720
FPS = 24
DUR = 28.0
NFRAMES = int(FPS * DUR)
BAR = 92  # letterbox bars (~2.39:1)
HORIZON = 520  # rooftop edge / start of wet reflective floor
SEED = 2049

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def smooth(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


# ----------------------------------------------------------------------------
# Static city layers (built once per process)
# ----------------------------------------------------------------------------

FOG = np.array([0.20, 0.17, 0.17])  # warm smog near the horizon
NEON = [
    (1.0, 0.15, 0.6),  # magenta
    (0.1, 0.9, 1.0),  # cyan
    (1.0, 0.55, 0.1),  # amber
    (0.6, 0.3, 1.0),  # violet
    (1.0, 0.2, 0.2),  # red
]


def sky():
    y = np.linspace(0, 1, H)[:, None]  # (H, 1)
    top = np.array([0.02, 0.03, 0.06])
    mid = np.array([0.10, 0.13, 0.18])
    hz = np.array([0.34, 0.22, 0.17])
    a = np.clip(y / 0.72, 0, 1)
    col = top * (1 - a) + mid * a
    b = np.clip((y - 0.45) / 0.3, 0, 1) ** 1.6
    col = col * (1 - b) + hz * b
    img = np.repeat(col[:, None, :], W + 600, axis=1)
    return img.astype(np.float32)


def building_layer(rng, hmin, hmax, base_y, dark, fogmix, win_density, sign_p, width):
    """RGBA float layer of a building silhouette row with lit windows/signs."""
    img = Image.new("RGBA", (width, H), (0, 0, 0, 0))
    glow = Image.new("RGB", (width, H), (0, 0, 0))
    d = ImageDraw.Draw(img)
    g = ImageDraw.Draw(glow)
    x = -40
    body = tuple(int(c * 255) for c in (np.array(dark) * (1 - fogmix) + FOG * fogmix))
    while x < width:
        bw = int(rng.integers(40, 140))
        bh = int(rng.integers(hmin, hmax))
        top = base_y - bh
        d.rectangle([x, top, x + bw, base_y + 400], fill=body + (255,))
        # stepped crowns / antennas
        if rng.random() < 0.4:
            cw = bw // 2
            d.rectangle([x + bw // 4, top - bh // 8, x + bw // 4 + cw, top], fill=body + (255,))
        if rng.random() < 0.35:
            ax = x + int(rng.integers(5, bw - 5))
            d.line([ax, top - 60, ax, top], fill=body + (255,), width=2)
            g.ellipse([ax - 2, top - 63, ax + 2, top - 59], fill=(255, 40, 40))
        # windows
        for wy in range(top + 6, base_y + 200, 7):
            for wx in range(x + 4, x + bw - 4, 6):
                if rng.random() < win_density:
                    warm = rng.random() < 0.6
                    c = (255, 180, 100) if warm else (140, 220, 255)
                    k = rng.uniform(0.25, 0.9) * (1 - fogmix * 0.7)
                    g.rectangle([wx, wy, wx + 2, wy + 3], fill=tuple(int(v * k) for v in c))
        # neon signs: vertical bars of pseudo glyphs
        if rng.random() < sign_p:
            col = NEON[int(rng.integers(len(NEON)))]
            k = 1 - fogmix * 0.5
            c = tuple(int(255 * v * k) for v in col)
            sx = x + int(rng.integers(4, max(5, bw - 24)))
            sy = top + int(rng.integers(10, max(11, bh // 2)))
            sh = int(rng.integers(60, 180))
            if rng.random() < 0.5:
                g.rectangle([sx, sy, sx + 16, sy + sh], outline=c, width=2)
                for gy in range(sy + 6, sy + sh - 8, 16):
                    glyph(g, rng, sx + 3, gy, 10, c)
            else:
                sw = int(rng.integers(50, 120))
                g.rectangle([sx, sy, sx + sw, sy + 22], fill=tuple(v // 3 for v in c), outline=c, width=2)
                for gx in range(sx + 5, sx + sw - 12, 14):
                    glyph(g, rng, gx, sy + 5, 11, c)
        x += bw + int(rng.integers(-10, 25))
    a = np.asarray(img, dtype=np.float32) / 255.0
    e = np.asarray(glow, dtype=np.float32) / 255.0
    rgb = a[..., :3] + e
    return rgb, a[..., 3:4]


def glyph(g, rng, x, y, s, c):
    for _ in range(int(rng.integers(2, 4))):
        if rng.random() < 0.5:
            yy = y + int(rng.integers(0, s))
            g.line([x, yy, x + s, yy], fill=c, width=2)
        else:
            xx = x + int(rng.integers(0, s))
            g.line([xx, y, xx, y + s], fill=c, width=2)


def hologram_ad(width):
    """Giant translucent advertisement figure on the skyline."""
    S = 2
    img = Image.new("L", (width * S // 4, H * S), 0)
    d = ImageDraw.Draw(img)
    cx, top = 120 * S, 150 * S
    # stylized reclining head/shoulders, facing left
    d.ellipse([cx - 40 * S, top, cx + 40 * S, top + 100 * S], fill=255)  # head
    d.ellipse([cx - 50 * S, top - 14 * S, cx + 54 * S, top + 78 * S], fill=255)  # bob hair
    d.chord([cx - 58 * S, top + 20 * S, cx + 62 * S, top + 120 * S], 180, 360, fill=255)  # hair ends
    d.rectangle([cx - 16 * S, top + 90 * S, cx + 16 * S, top + 140 * S], fill=255)  # neck
    d.polygon(
        [(cx - 120 * S, top + 260 * S), (cx - 90 * S, top + 140 * S), (cx + 90 * S, top + 140 * S),
         (cx + 130 * S, top + 260 * S), (cx + 100 * S, top + 400 * S), (cx - 100 * S, top + 400 * S)],
        fill=255,
    )  # shoulders/torso
    img = img.resize((width // 4, H), Image.LANCZOS)
    m = np.asarray(img, dtype=np.float32) / 255.0
    edge = np.asarray(img.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(1.5)), dtype=np.float32) / 255.0
    return m, edge


# ----------------------------------------------------------------------------
# Characters
# ----------------------------------------------------------------------------

def ik(sh, target, l1, l2, forward=False):
    """Two-bone IK; returns the elbow (bending down/outward) or knee (forward)."""
    dx, dy = target[0] - sh[0], target[1] - sh[1]
    d = min(math.hypot(dx, dy), l1 + l2 - 0.01)
    base = math.atan2(dy, dx)
    alpha = math.acos(max(-1.0, min(1.0, (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d))))
    cands = [(sh[0] + l1 * math.cos(base + k * alpha), sh[1] + l1 * math.sin(base + k * alpha)) for k in (1, -1)]
    return max(cands, key=(lambda p: p[0]) if forward else (lambda p: p[1] + 0.3 * p[0]))


def umbrella_geom(hand, lift, opened):
    """Umbrella swung up from her side (closed) and opened over his head.

    Returns canopy polygon, shaft segment, rim tips (drip sources) and the
    two outer rim points."""
    phi = math.radians(90 + 160 * lift)  # shaft direction hand -> tip
    dvec = (math.cos(phi), math.sin(phi))
    perp = (-dvec[1], dvec[0])
    apex = (hand[0] + 100 * dvec[0], hand[1] + 100 * dvec[1])
    R = 7 + 95 * opened
    hgt = 58 + 32 * (1 - opened)

    def wp(u, w):
        return (apex[0] + u * perp[0] - w * dvec[0], apex[1] + u * perp[1] - w * dvec[1])

    top = [wp(u, hgt * (1 - math.sqrt(max(0.0, 1 - (u / R) ** 2)))) for u in np.linspace(-R, R, 31)]
    panels = 8
    bottom, tips = [], []
    for k in range(panels, -1, -1):
        u0 = -R + 2 * R * k / panels
        tips.append(wp(u0, hgt))
        if k == 0:
            break
        for f in (0.25, 0.5, 0.75):
            u = u0 - 2 * R / panels * f
            # seen from slightly below: the rim bulges down, scalloped between ribs
            bulge = 12 * opened * (1 - (u / R) ** 2) - 5 * opened * math.sin(math.pi * f)
            bottom.append(wp(u, hgt + bulge))
    poly = top + bottom
    return poly, (hand[0], hand[1] + 12), apex, tips, (wp(-R, hgt), wp(R, hgt))


def rot(pts, pivot, ang):
    c, s_ = math.cos(ang), math.sin(ang)
    return [(pivot[0] + (x - pivot[0]) * c - (y - pivot[1]) * s_, pivot[1] + (x - pivot[0]) * s_ + (y - pivot[1]) * c)
            for x, y in pts]


def foot_target(p, D, amp, lift_h, rest_dx, beta=0.62):
    """Foot position relative to the hip for gait phase p (0..1).

    Stance (0..beta): the foot is planted, sliding back relative to the hip at
    exactly the body's speed, so it never skates on the ground. Swing: the foot
    lifts and eases forward to the next heel strike."""
    p %= 1.0
    if p < beta:
        x, y = D * beta / 2 - D * p, 0.0
    else:
        q = (p - beta) / (1 - beta)
        e = 0.5 - 0.5 * math.cos(math.pi * q)
        x, y = -D * beta / 2 + D * beta * e, -lift_h * math.sin(math.pi * q) ** 1.3
    return amp * x + (1 - amp) * rest_dx, amp * y


def leg_shapes(hip, foot_rel, l1, l2, w1, w2, shoe):
    """Thigh + shin via IK (knee bends forward) and a small shoe."""
    foot = (hip[0] + foot_rel[0], hip[1] + foot_rel[1])
    knee = ik(hip, foot, l1, l2, forward=True)
    heel, toe, hgt = shoe
    sole = [(foot[0] + heel, foot[1]), (foot[0] + toe, foot[1]), (foot[0] + toe - 2, foot[1] - hgt),
            (foot[0] + heel + 1, foot[1] - hgt - 2)]
    return [("line", [hip, knee], w1), ("line", [knee, foot], w2), ("poly", sole)]


def body_bob(c, amp, h):
    """Highest at mid-stance of each leg, lowest at double support."""
    return h * amp * 0.5 * (1 + math.cos(4 * math.pi * (c - 0.31)))


def detective_shapes(fx, fy, c, amp):
    """Long coat, collar up, seen 3/4 from behind, walking right.

    fx is the pelvis x; c is the gait cycle (0..1 per two steps).
    Returns world-space shapes: ("poly", pts) | ("ell", ...) | ("line", pts, width)."""
    D = 2 * STEP
    up = body_bob(c, amp, 3.0)
    hip_y = fy - 108 - up
    shapes = []
    rels = []
    for k, (side, off) in enumerate(((-6, 0.0), (6, 0.5))):  # far leg first
        rel = foot_target(c + off, D, amp, 12, side * 0.6)
        rels.append(rel[0])
        shapes += leg_shapes((fx + side, hip_y), (rel[0], fy - hip_y + rel[1]), 57, 55, 15, 12, (-6, 14, 7))
    lean = 0.05 * amp + 0.012 * amp * math.sin(4 * math.pi * c)
    pivot = (fx, hip_y)
    lag = math.sin(2 * math.pi * c - 1.1) * amp  # hem trails the stride
    back, front = min(rels), max(rels)
    hem_l, hem_r = -56 + 0.35 * back * amp - 3 * amp + 2 * lag, 54 + 0.3 * front * amp + 2 * lag
    by = lambda y: hip_y + 108 + y  # body coords relative to standing feet
    coat = [(fx - 30, by(-228)), (fx - 46, by(-214)), (fx - 50, by(-150)), (fx - 52 - 2 * amp, by(-70)),
            (fx + hem_l, by(-40)), (fx - 18 + lag, by(-35)), (fx + 18 + lag, by(-35)),
            (fx + hem_r, by(-40)), (fx + 50, by(-70)), (fx + 48, by(-150)), (fx + 44, by(-214)), (fx + 28, by(-228))]
    collar = [(fx - 26, by(-236)), (fx - 30, by(-222)), (fx + 26, by(-222)), (fx + 22, by(-240)), (fx + 12, by(-246)),
              (fx - 14, by(-246))]
    shapes += [("poly", rot(coat, pivot, lean)), ("poly", rot(collar, pivot, lean * 1.1))]
    hx, hy = rot([(fx + 1, by(-253))], pivot, lean * 1.15)[0]
    shapes.append(("ell", (hx + 1.5 * amp * math.sin(4 * math.pi * c - 2), hy, 14, 17)))
    return shapes


def companion_shapes(fx, fy, c, amp, elbow, hand, s, arm_up):
    """Holographic companion in profile, walking right; front arm in world coords."""
    D = 2 * STEP
    up = body_bob(c, amp, 2.4) * s
    hip_y = fy - 112 * s - up
    shapes = []
    for side, off in ((-4, 0.5), (4, 0.0)):
        rel = foot_target(c + off, D, amp, 8 * s, side * 0.5)
        shapes += leg_shapes((fx + side * s, hip_y), (rel[0], fy - hip_y + rel[1]), 57 * s, 56 * s, 9 * s, 7 * s,
                             (-3 * s, 10 * s, 5 * s))
    lean = 0.04 * amp
    pivot = (fx, hip_y)
    P = lambda pts: rot([(fx + x * s, hip_y + (y + 112) * s) for x, y in pts], pivot, lean)
    lag = 4 * amp * math.sin(2 * math.pi * c - 1.2)
    hair_b = 1.5 * amp * math.sin(4 * math.pi * c - 2.2)
    shapes += [
        ("ell", (*P([(0, -237)])[0], 12 * s, 15 * s)),
        ("poly", P([(-15, -244), (-13, -256), (0, -260), (14, -254), (16, -232 + hair_b), (10, -226 + hair_b), (8, -244)])),
        ("poly", P([(-5, -224), (5, -224), (6, -212), (-6, -212)])),
        ("poly", P([(-22, -212), (22, -212), (18, -150), (14, -120), (28 + lag, -42), (6 + lag * 0.5, -38),
                    (-6 + lag * 0.5, -38), (-28 + lag, -42), (-14, -120), (-18, -150)])),
    ]
    # free arm swings opposite the near leg, elbow bending as it comes forward
    sw = amp * math.sin(2 * math.pi * c + math.pi)
    shp = P([(16, -206)])[0]
    a1 = 0.24 * sw
    el = (shp[0] + 46 * s * math.sin(a1), shp[1] + 46 * s * math.cos(a1))
    a2 = a1 + 0.1 + 0.22 * max(0.0, sw) * amp
    hd = (el[0] + 42 * s * math.sin(a2), el[1] + 42 * s * math.cos(a2))
    shapes.append(("line", [shp, el, hd], 7 * s))
    sh = P([(-18, -206)])[0]
    shapes.append(("line", [sh, elbow, hand], 7 * s))
    shapes.append(("ell", (hand[0], hand[1], 5 * s, 5 * s)))
    return shapes


# ----------------------------------------------------------------------------
# Choreography: walking, camera moves and cuts
# ----------------------------------------------------------------------------

V_WALK = 72.0  # px/s: a slow, heavy stroll
STEP = 56.0  # px per step -> ~1.3 steps/s
X0 = 70.0


def walk_speed(t):
    return V_WALK * (1 - smooth(5.6, 7.0, t)) + V_WALK * 0.85 * smooth(19.2, 20.6, t)


_TG = np.arange(0, DUR + 1, 1 / 240)
_XG = X0 + np.cumsum([walk_speed(x) for x in _TG]) / 240


def motion(t):
    """K's world x, gait cycle (0..1 per two steps) and walk amplitude at t."""
    x = float(np.interp(t, _TG, _XG))
    return x, (x - X0) / (2 * STEP), min(1.0, walk_speed(t) / V_WALK * 1.25)


def camera(t):
    """(zoom, centre x, centre y) in world coords; hard cuts between shots."""
    kx, _, _ = motion(t)
    fy = HORIZON + 46
    pair = kx + 60
    if t < 6.5:  # wide establishing, then ease into a full-body tracking shot of the walk
        u = smooth(2.5, 6.5, t)
        z = 1.0 + 0.4 * u
        cx = 640 + (pair - 640) * (0.3 + 0.7 * u)
        cy = 380 + (fy - 150 - 380) * u
    elif t < 12.8:  # push in to a medium two-shot while she raises the umbrella
        u = smooth(6.5, 10.5, t)
        z = 1.4 + 0.45 * u
        cx = pair
        cy = fy - 150 - 25 * u
    elif t < 15.6:  # cut: close-up on him under the umbrella
        u = (t - 12.8) / 2.8
        z = 2.7 + 0.2 * u
        cx, cy = kx + 18 - 6 * u, fy - 262 + 4 * u
    elif t < 19.0:  # cut: close-up on her face, rain flickering through her
        u = (t - 15.6) / 3.4
        z = 3.0 + 0.25 * u
        cx, cy = kx + 118 + 5 * u, fy - 218
    else:  # cut: medium, then a long pull back to wide as they walk away
        u = smooth(19.5, 26.0, t)
        z = 1.5 - 0.5 * u
        cx = pair + (640 - pair) * u
        cy = fy - 160 + (380 - (fy - 160)) * u
    # subtle handheld drift
    cx += 2.0 * math.sin(t * 0.9) * z
    cy += 1.2 * math.sin(t * 1.3 + 1) * z
    hw, hh = W / (2 * z), H / (2 * z)
    return z, min(max(cx, hw), W - hw), min(max(cy, hh), H - hh)


# ----------------------------------------------------------------------------
# Renderer
# ----------------------------------------------------------------------------

class Scene:
    def __init__(self):
        rng = np.random.default_rng(SEED)
        self.sky = sky()
        LW = W + 600
        self.layers = [
            # (rgb, alpha, parallax)
            building_layer(rng, 180, 380, HORIZON - 40, (0.05, 0.06, 0.08), 0.6, 0.10, 0.25, LW) + (0.15,),
            building_layer(rng, 140, 300, HORIZON - 10, (0.025, 0.03, 0.045), 0.3, 0.12, 0.35, LW) + (0.35,),
            building_layer(rng, 60, 200, HORIZON + 20, (0.015, 0.02, 0.03), 0.12, 0.08, 0.5, LW) + (0.7,),
        ]
        self.ad_mask, self.ad_edge = hologram_ad(LW)
        # rain particles in normalised screen space: x, y, depth
        n = 2200
        self.rain = np.stack([rng.uniform(-0.2, 1.2, n), rng.uniform(0, 1, n), rng.uniform(0.15, 1, n)], 1)
        self.rain_speed = rng.uniform(0.9, 1.3, n)
        self.grain = rng.normal(0, 1, (8, H, W)).astype(np.float32)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        self.vignette = np.clip(1.15 - 0.55 * r ** 1.8, 0.2, 1)[..., None].astype(np.float32)
        self.xx, self.yy = xx, yy

    # -- spinner flying through the frame (world coords) --
    def spinner(self, d, glow, t, drift):
        t0, t1 = 18.5, 27.5
        if not (t0 < t < t1):
            return
        u = (t - t0) / (t1 - t0)
        x = -120 + u * (W + 300) - drift * 0.5
        y = 250 - 70 * u + 12 * math.sin(u * 7)
        s = 0.9 + 0.3 * u
        body = [(x - 34 * s, y), (x - 20 * s, y - 9 * s), (x + 20 * s, y - 10 * s), (x + 40 * s, y - 2 * s),
                (x + 36 * s, y + 6 * s), (x - 30 * s, y + 7 * s)]
        d.polygon(body, fill=(8, 10, 14))
        glow.ellipse([x + 34 * s, y - 3 * s, x + 44 * s, y + 3 * s], fill=(255, 245, 220))
        glow.polygon([(x + 40 * s, y), (x + 260 * s, y + 40 * s), (x + 260 * s, y - 15 * s)], fill=(60, 58, 50))
        if int(t * 3) % 2 == 0:
            glow.ellipse([x - 36 * s, y - 2 * s, x - 30 * s, y + 4 * s], fill=(255, 30, 30))
        glow.line([(x - 28 * s, y + 7 * s), (x + 30 * s, y + 6 * s)], fill=(30, 180, 255), width=2)

    def background(self, t, rng, cx):
        """Sky, city, giant ad, spinner and rooftop in world coords (W x H)."""
        drift = 120 + 200 * smooth(0, DUR, t) + (cx - 640) * 0.8
        img = self.sky[:, :W].copy()
        for li, (rgb, a, par) in enumerate(self.layers):
            off = int(min(max(drift * par, 0), 600))
            img = img * (1 - a[:, off:off + W]) + rgb[:, off:off + W] * a[:, off:off + W]
            if li == 1:
                adx = 880 - off
                flick = 0.75 + 0.25 * math.sin(t * 13) * math.sin(t * 3.1) + (0.4 if rng.random() < 0.04 else 0)
                m = np.zeros((H, W), np.float32)
                e = np.zeros((H, W), np.float32)
                aw = self.ad_mask.shape[1]
                x0, x1 = max(adx, 0), min(adx + aw, W)
                if x1 > x0:
                    m[:, x0:x1] = self.ad_mask[:, x0 - adx:x1 - adx]
                    e[:, x0:x1] = self.ad_edge[:, x0 - adx:x1 - adx]
                scan = 0.75 + 0.25 * np.sin((self.yy[:, :1] + t * 40) * 0.9)
                hue = 0.5 + 0.5 * math.sin(t * 0.6)
                c1 = np.array([1.0, 0.25, 0.7]) * (1 - hue) + np.array([0.3, 0.6, 1.0]) * hue
                img += (m * 0.42 * scan + e * 1.1)[..., None] * c1 * flick
                img += (np.clip(1 - np.abs(self.xx - (adx + 120)) / 380, 0, 1) ** 2 * 0.06)[..., None] * c1
        fogb = np.exp(-((self.yy - (HORIZON - 30)) / 120) ** 2)[..., None] * 0.18
        img = img * (1 - fogb) + FOG * fogb
        pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        glow_layer = Image.new("RGB", (W, H), (0, 0, 0))
        d = ImageDraw.Draw(pil)
        self.spinner(d, ImageDraw.Draw(glow_layer), t, drift)
        d.rectangle([0, HORIZON, W, H], fill=(6, 7, 10))
        return Image.fromarray(np.minimum(np.asarray(pil, np.int16) + np.asarray(glow_layer, np.int16), 255).astype(np.uint8))

    def frame(self, i):
        t = i / FPS
        rng = np.random.default_rng(1000 + i)
        z, ccx, ccy = camera(t)
        S = 2

        def to_s(p):
            return ((p[0] - ccx) * z + W / 2, (p[1] - ccy) * z + H / 2)

        def raster(shapes):
            m = Image.new("L", (W * S, H * S), 0)
            d = ImageDraw.Draw(m)
            for sh in shapes:
                if sh[0] == "poly":
                    d.polygon([(x * S, y * S) for x, y in map(to_s, sh[1])], fill=255)
                elif sh[0] == "ell":
                    cx_, cy_, rx, ry = sh[1]
                    x_, y_ = to_s((cx_, cy_))
                    d.ellipse([(x_ - rx * z) * S, (y_ - ry * z) * S, (x_ + rx * z) * S, (y_ + ry * z) * S], fill=255)
                elif sh[0] == "line":
                    pts = [(x * S, y * S) for x, y in map(to_s, sh[1])]
                    wdt = max(1, int(sh[2] * z * S))
                    d.line(pts, fill=255, width=wdt, joint="curve")
                    for x, y in (pts[0], pts[-1]):
                        d.ellipse([x - wdt / 2, y - wdt / 2, x + wdt / 2, y + wdt / 2], fill=255)
            return m.resize((W, H), Image.LANCZOS)

        # background: world image cropped by the camera, defocused when zoomed in
        bg = self.background(t, rng, ccx)
        hw, hh = W / (2 * z), H / (2 * z)
        box = (max(0.0, ccx - hw), max(0.0, ccy - hh), min(float(W), ccx + hw), min(float(H), ccy + hh))
        bg = bg.resize((W, H), Image.BICUBIC, box=box)
        if z > 1.15:
            bg = bg.filter(ImageFilter.GaussianBlur((z - 1) * 1.8))
        img = np.asarray(bg, dtype=np.float32) / 255.0

        # choreography
        kx, cyc, amp = motion(t)
        fy = HORIZON + 46
        cs = 0.92
        comp_x = kx + 122  # she walks/stands outside the umbrella, in the rain
        holo_col = np.array([0.55, 0.45, 1.0]) * (0.5 + 0.5 * math.sin(t * 0.8)) + np.array([0.2, 0.85, 1.0]) * (0.5 - 0.5 * math.sin(t * 0.8))
        lift = smooth(8.3, 10.8, t)
        opened = smooth(10.8, 12.3, t)
        hcyc = cyc + 0.27  # she falls into step slightly out of phase with him
        sh = rot([(comp_x - 18 * cs, fy + 4 - 206 * cs - body_bob(hcyc, amp, 2.4) * cs)],
                 (comp_x, fy + 4 - 112 * cs), 0.04 * amp)[0]
        carry = 0.18 * amp * math.sin(2 * math.pi * hcyc) * (1 - lift)  # closed umbrella swings a little
        down = (sh[0] - 4 + 80 * math.sin(carry), sh[1] + 86 * math.cos(carry))
        up = (kx + 46, fy - 254 + 2 * math.sin(t * 1.3) - body_bob(cyc, amp, 3.0))
        hand = (down[0] + (up[0] - down[0]) * lift, down[1] + (up[1] - down[1]) * lift - 40 * math.sin(math.pi * lift))
        elbow = ik(sh, hand, 48 * cs + 4, 46 * cs + 4)
        canopy, grip, apex, tips, (rimL, rimR) = umbrella_geom(hand, lift, opened)
        canopy_s = [to_s(p) for p in canopy]
        rimL_s, rimR_s, apex_s = to_s(rimL), to_s(rimR), to_s(apex)
        kx_s, _ = to_s((kx, fy))
        comp_s = to_s((comp_x, fy - 120))
        hs = int(round(to_s((0, HORIZON))[1]))  # screen row of the rooftop edge

        # him
        km = raster(detective_shapes(kx, fy, cyc, amp))
        kmask = np.asarray(km, dtype=np.float32)[..., None] / 255.0
        kedge = np.asarray(km.filter(ImageFilter.FIND_EDGES), dtype=np.float32)[..., None] / 255.0
        side = (self.xx - kx_s)[..., None]
        rim = kedge * (np.where(side > 0, 1.0, 0.0) * holo_col * 1.3 + np.where(side <= 0, 1.0, 0.0) * np.array([1.0, 0.55, 0.25]) * 0.35)
        figure_col = np.array([0.025, 0.028, 0.035])
        img = img * (1 - kmask) + figure_col * kmask + rim

        # rain layer in screen space (built before her so drops can disturb the hologram)
        rain = Image.new("L", (W, H), 0)
        rd = ImageDraw.Draw(rain)
        wind = -0.12
        for (x, y, dz), sp in zip(self.rain, self.rain_speed):
            yy = (y + t * sp * (0.9 + dz * 1.4)) % 1.0
            px = (x + wind * yy * dz + t * wind * 0.4) * W
            py = yy * (H + 80) - 40
            L = (8 + 34 * dz) * z ** 0.5  # close-ups: rain nearer the lens reads bigger
            rd.line([(px, py), (px + wind * L, py + L)], fill=int(40 + 150 * dz), width=1 if dz * z ** 0.5 < 0.75 else 2)
        if hs < H:  # splashes on the wet floor
            for _ in range(int(70 * (H - hs) / 200)):
                sx = rng.uniform(0, W)
                sy = rng.uniform(hs + 4, H - BAR + 20)
                r = rng.uniform(1.5, 4.5) * z ** 0.5
                rd.ellipse([sx - r * 2.2, sy - r * 0.5, sx + r * 2.2, sy + r * 0.5], outline=int(rng.uniform(70, 160)))
        shelter = Image.new("L", (W, H), 0)
        if opened > 0.05:
            sd = ImageDraw.Draw(shelter)
            drop = lambda p: (p[0] + wind * (H - p[1]), H)
            sd.polygon(canopy_s, fill=255)
            sd.polygon([rimL_s, rimR_s, drop(rimR_s), drop(rimL_s)], fill=int(255 * opened))
            shelter = shelter.filter(ImageFilter.GaussianBlur(3))
        shelter_a = np.asarray(shelter, dtype=np.float32) / 255.0
        rain_a = np.asarray(rain.filter(ImageFilter.GaussianBlur(0.6)), dtype=np.float32) / 255.0 * (1 - shelter_a)

        # drips off the outer rib tips + drops bursting on the canopy
        drip = Image.new("L", (W, H), 0)
        dd = ImageDraw.Draw(drip)
        if opened > 0.6:
            feet_s = to_s((0, fy + 4))[1]
            for k, tip in enumerate(tips[:2] + tips[-2:]):
                tx, ty = to_s(tip)
                for ph in (0.0, 0.37, 0.71):
                    u = (t * (1.3 + 0.17 * k) + ph + k * 0.13) % 1.0
                    fall = u * u * (feet_s - ty)
                    dx = wind * fall
                    dd.line([(tx + dx, ty + fall), (tx + dx + wind * 6 * z, ty + fall + (6 + 14 * u) * z ** 0.7)],
                            fill=int(220 * opened), width=max(2, int(1.5 * z ** 0.7)))
            topn = len(canopy_s) // 2
            for _ in range(18):
                px, py = canopy_s[int(rng.integers(0, topn))]
                rr = rng.uniform(1, 2.5) * z ** 0.6
                dd.ellipse([px - rr, py - rr - 1, px + rr, py + rr - 1], fill=int(rng.uniform(120, 255)))
        drip_a = np.asarray(drip.filter(ImageFilter.GaussianBlur(0.7)), dtype=np.float32) / 255.0

        # her: additive translucent hologram, disturbed wherever rain passes through
        comp_img = raster(companion_shapes(comp_x, fy + 4, hcyc, amp, elbow, hand, cs, lift))
        comp = np.asarray(comp_img, dtype=np.float32) / 255.0
        hit = rain_a * comp
        hit_img = Image.fromarray((np.clip(hit * 255, 0, 255)).astype(np.uint8))
        hitb = np.asarray(hit_img.filter(ImageFilter.GaussianBlur(3)), dtype=np.float32) / 255.0
        rows = np.where(hitb.sum(axis=1) > 0.6)[0]
        if len(rows):
            for _ in range(int(rng.integers(3, 9))):
                gy, gh = int(rng.choice(rows)), int(rng.integers(2, 6) * z ** 0.6)
                comp[gy:gy + gh] = np.roll(comp[gy:gy + gh], int(rng.choice([-1, 1]) * rng.integers(2, 9) * z ** 0.7), 1)
        ys = np.where(comp.max(axis=1) > 0)[0]
        if len(ys) > 14 and rng.random() < 0.12:  # larger glitch slice
            gy, gh = int(rng.integers(ys[0], ys[-1] - 12)), int(rng.integers(3, 12) * z ** 0.6)
            comp[gy:gy + gh] = np.roll(comp[gy:gy + gh], int(rng.integers(-16, 16) * z ** 0.7), 1)
        flick = 0.85 + 0.12 * math.sin(t * 21)
        if rng.random() < 0.28:
            flick *= rng.uniform(0.3, 0.7)
        elif rng.random() < 0.08:
            flick *= 1.35
        scan = 0.65 + 0.35 * (np.sin((self.yy + t * 30 * z) * 1.4 / max(1.0, z * 0.6)) > -0.2)
        comp_img = Image.fromarray((comp * 255).astype(np.uint8))
        halo = np.asarray(comp_img.filter(ImageFilter.GaussianBlur(10 * z ** 0.7)), dtype=np.float32) / 255.0
        cedge = np.asarray(comp_img.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
        face_y = to_s((0, fy - 240))[1]
        vert = np.clip((self.yy - face_y) / (260 * z), 0, 1)
        holes = 1 - np.clip(hitb * 6, 0, 0.9)
        body = comp * scan * (0.55 - 0.25 * vert) * holes
        sparks = np.clip(hit * 2.5 + hitb * 1.5, 0, 1.2)
        img = img * (1 - comp[..., None] * 0.35) + ((body + cedge * 0.8 * holes)[..., None] * holo_col + halo[..., None] * holo_col * 0.45) * flick
        img += sparks[..., None] * (np.array([0.85, 0.95, 1.0]) * 0.6 + holo_col * 0.6)
        light = np.exp(-(((self.xx - comp_s[0]) / (140 * z)) ** 2 + ((self.yy - comp_s[1]) / (180 * z)) ** 2))[..., None]
        img += light * holo_col * 0.07 * flick

        # the umbrella (physical, dark, rim-lit by her glow underneath)
        hook = [grip, (grip[0] - 1, grip[1] + 4), (grip[0] - 4, grip[1] + 6), (grip[0] - 6, grip[1] + 3)]
        um = raster([("poly", canopy), ("line", [grip, apex], 2), ("line", hook, 2)])
        umask = np.asarray(um, dtype=np.float32)[..., None] / 255.0
        uedge = np.asarray(um.filter(ImageFilter.FIND_EDGES), dtype=np.float32)[..., None] / 255.0
        under = np.clip((self.yy - apex_s[1]) / (60 * z), 0, 1)[..., None]
        urim = uedge * (holo_col * 1.4 * under * flick + np.array([1.0, 0.6, 0.35]) * 0.5 * (1 - under))
        img = img * (1 - umask * 0.97) + np.array([0.02, 0.022, 0.03]) * umask + urim

        # wet floor reflection below the rooftop edge (when it is on screen)
        if 0 < hs < H:
            keep = img[hs:].copy()  # her legs and feet stay crisp over the reflection
            keep_m = np.clip(comp[hs:] * 1.6 + halo[hs:] * 0.5, 0, 1)[..., None]
            nref = H - hs
            src_rows = np.clip(hs - 1 - (np.arange(nref) / 1.0).astype(int), 0, H - 1)
            k = np.arange(nref)[:, None]
            ripple = (np.sin(k * 0.35 / z + t * 6 + self.xx[0][None, :] * 0.02 / z) * (2 + k * 0.03) * z ** 0.5).astype(np.int32)
            cols = np.clip(self.xx[0][None, :].astype(np.int32) + ripple, 0, W - 1)
            refl = img[src_rows[:, None], cols]
            refl_img = Image.fromarray((np.clip(refl, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2.2 * z ** 0.5))
            refl = np.asarray(refl_img, dtype=np.float32) / 255.0
            fade = np.linspace(0.55, 0.25, nref)[:, None, None]
            img[hs:] = img[hs:] * (1 - fade) + refl * fade + np.array([0.01, 0.012, 0.016])
            img[hs:] = img[hs:] * (1 - keep_m) + keep * keep_m
            img[hs:] = img[hs:] * (1 - kmask[hs:]) + (figure_col * kmask[hs:] + rim[hs:])

        rain_tint = np.array([0.62, 0.72, 0.85]) + light * holo_col * 2.0
        img = img + rain_a[..., None] * rain_tint * 0.42 + drip_a[..., None] * (np.array([0.7, 0.8, 0.95]) * 0.5 + holo_col * 0.4)

        # bloom (threshold -> blur at two radii -> add)
        lum = img.mean(axis=2, keepdims=True)
        bright = np.clip(img * np.clip((lum - 0.45) * 2.5, 0, 1), 0, 2)
        small = Image.fromarray((np.clip(bright, 0, 1) * 255).astype(np.uint8)).resize((W // 4, H // 4), Image.BILINEAR)
        b1 = np.asarray(small.filter(ImageFilter.GaussianBlur(3)).resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0
        b2 = np.asarray(small.filter(ImageFilter.GaussianBlur(12)).resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0
        img = img + b1 * 0.7 + b2 * 0.9

        # grade: teal shadows / amber highlights, filmic shoulder
        lum = img.mean(axis=2, keepdims=True)
        img = img + (1 - np.clip(lum * 3, 0, 1)) * np.array([-0.01, 0.012, 0.025]) + np.clip(lum - 0.5, 0, 1) * np.array([0.06, 0.02, -0.04])
        img = 1 - np.exp(-img * 1.6)

        # chromatic aberration, grain, vignette
        img[..., 0] = np.roll(img[..., 0], 2, axis=1)
        img[..., 2] = np.roll(img[..., 2], -2, axis=1)
        img = img * self.vignette + self.grain[i % 8][..., None] * 0.022

        fade = smooth(0, 2.5, t) * (1 - smooth(DUR - 2.0, DUR, t))
        img *= fade
        out = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        dr = ImageDraw.Draw(out)
        self.title(dr, t)
        dr.rectangle([0, 0, W, BAR], fill=(0, 0, 0))
        dr.rectangle([0, H - BAR, W, H], fill=(0, 0, 0))
        self.subtitle(dr, t)
        return out.tobytes()

    def title(self, dr, t):
        a = smooth(1.5, 3.0, t) * (1 - smooth(4.8, 6.0, t))
        if a <= 0:
            return
        f = ImageFont.truetype(FONT, 34)
        txt = "L O S   A N G E L E S   ·   2 0 4 9"
        w = dr.textlength(txt, font=f)
        dr.text(((W - w) / 2, 200), txt, font=f, fill=tuple(int(255 * a * c) for c in (0.95, 0.75, 0.55)))

    def subtitle(self, dr, t):
        lines = [
            (13.0, 15.5, "“You’re standing in the rain.”"),
            (15.8, 18.9, "“It can’t hurt me. I just want you to stay dry.”"),
        ]
        f = ImageFont.truetype(FONT, 22)
        for s, e, txt in lines:
            a = smooth(s, s + 0.4, t) * (1 - smooth(e - 0.4, e, t))
            if a > 0:
                w = dr.textlength(txt, font=f)
                dr.text(((W - w) / 2, H - BAR + 32), txt, font=f, fill=(int(230 * a), int(220 * a), int(200 * a)))


_scene = None


def _init():
    global _scene
    _scene = Scene()


def _render(i):
    return _scene.frame(i)


# ----------------------------------------------------------------------------
# Audio: rain bed, synth pad drone, low "braam" swell, spinner fly-by,
# umbrella pop + canopy drumming, hologram crackle, footsteps
# ----------------------------------------------------------------------------

def onepole(x, cutoff, sr):
    # one-pole low-pass magnitude response applied in the FFT domain
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    H_ = 1 / np.sqrt(1 + (f / cutoff) ** 2)
    return np.fft.irfft(X * H_, len(x))


def synth_audio(path):
    sr = 44100
    n = int(sr * DUR)
    t = np.arange(n) / sr
    rng = np.random.default_rng(SEED)
    # rain: filtered noise + random droplet ticks
    noise = rng.normal(0, 1, n)
    rain = onepole(noise, 3500, sr) * 0.12 + (onepole(noise, 9000, sr) - onepole(noise, 1500, sr)) * 0.05
    ticks = np.zeros(n)
    idx = rng.integers(0, n - 400, 9000)
    ticks[idx] = rng.uniform(-1, 1, len(idx))
    env = np.exp(-np.arange(300) / 40)
    ticks = np.convolve(ticks, env, mode="same")
    rain += onepole(ticks, 6000, sr) * 0.25
    rain *= 0.8 + 0.2 * np.sin(2 * np.pi * 0.07 * t)

    # pad: detuned saws (additive), D minor add9, slow swell
    def saw(freq, detune):
        s = np.zeros(n)
        for h in range(1, 14):
            s += np.sin(2 * np.pi * freq * (1 + detune) * h * t + h) / h
        return s

    pad = np.zeros(n)
    for f in (73.42, 110.0, 174.61, 220.0, 329.63):
        for dt in (-0.004, 0.0, 0.005):
            pad += saw(f, dt)
    pad = onepole(pad, 900, sr)
    pad *= (0.25 + 0.75 * np.clip((t - 1) / 8, 0, 1)) * 0.035

    # braam swell at open
    br = np.zeros(n)
    for f in (36.71, 55.0, 73.42):
        for dt in (-0.006, 0.006):
            br += saw(f, dt)
    br = np.tanh(onepole(br, 400, sr) * 1.5)
    benv = np.clip(t / 1.2, 0, 1) * np.exp(-np.clip(t - 1.2, 0, None) / 2.0)
    br *= benv * 0.22

    # spinner fly-by 5..15s: bandpassed noise sweeping, panned L->R
    u = np.clip((t - 18.5) / 9, 0, 1)
    fly_env = np.exp(-((t - 23.0) / 2.6) ** 2)
    hum = np.sin(2 * np.pi * (180 - 60 * u) * t) * 0.3 + (onepole(noise, 2000, sr) - onepole(noise, 500, sr))
    fly = hum * fly_env * 0.35

    # sparse piano-like high notes (sine bells) when she reaches out
    bells = np.zeros(n)
    for start, f in ((9.0, 587.33), (11.6, 698.46), (13.0, 880.0), (15.8, 659.25), (19.2, 587.33), (22.5, 523.25)):
        m = t >= start
        tt = t[m] - start
        bells[m] += (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt)) * np.exp(-tt / 2.2) * 0.07

    # umbrella snapping open (~11.4s), then rain drumming on its canopy
    pop = np.exp(-((t - 11.4) / 0.12) ** 2) * (onepole(noise, 1200, sr) - onepole(noise, 200, sr)) * 1.4
    drum = np.zeros(n)
    idx = rng.integers(int(12.0 * sr), n - 400, 1800)
    drum[idx] = rng.uniform(0.3, 1, len(idx))
    drum = np.convolve(drum, np.sin(np.arange(250) * 0.25) * np.exp(-np.arange(250) / 50), mode="same") * 0.05

    # hologram crackle: short digital blips whenever rain disrupts her
    crackle = np.zeros(n)
    for st in rng.uniform(1.5, DUR - 1, 70):
        k0, ln = int(st * sr), int(rng.uniform(0.01, 0.05) * sr)
        tt = np.arange(ln) / sr
        crackle[k0:k0 + ln] += np.sign(np.sin(2 * np.pi * rng.uniform(800, 3000) * tt)) * rng.uniform(0.01, 0.03)

    # wet footsteps synced to the walk cycle
    steps = np.zeros(n)
    half = np.floor((np.interp(t, _TG, _XG) - X0) / STEP)
    for k in np.where(np.diff(half) > 0)[0]:
        ln = int(0.07 * sr)
        steps[k:k + ln] += np.exp(-np.arange(len(steps[k:k + ln])) / (0.015 * sr)) * rng.uniform(0.7, 1.0)
    steps = onepole(steps * rng.normal(0, 1, n), 1400, sr) * 0.5

    fade = np.clip(t / 2.0, 0, 1) * np.clip((DUR - t) / 2.0, 0, 1)
    L = (steps + rain + pad + br + fly * (1 - u) + bells + pop + drum * 0.8 + crackle * 0.6) * fade
    R = (steps * 0.9 + rain[::-1] * 0.9 + rain * 0.1 + pad + br + fly * u + bells + pop + drum + crackle) * fade
    st = np.stack([L, R], 1)
    st /= np.abs(st).max() / 0.9
    pcm = (st * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def main():
    out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "cyberpunk_rain.mp4"
    if "--still" in sys.argv:
        tsec = float(sys.argv[sys.argv.index("--still") + 1])
        s = Scene()
        Image.frombytes("RGB", (W, H), s.frame(int(tsec * FPS))).save(out)
        return
    wav = os.path.splitext(out)[0] + ".wav"
    synth_audio(wav)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "slow", "-crf", "20",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(os.cpu_count(), initializer=_init) as pool:
        for k, buf in enumerate(pool.imap(_render, range(NFRAMES), chunksize=4)):
            p.stdin.write(buf)
            if k % 48 == 0:
                print(f"frame {k}/{NFRAMES}", flush=True)
    p.stdin.close()
    p.wait()
    os.remove(wav)
    print("wrote", out)


if __name__ == "__main__":
    main()
