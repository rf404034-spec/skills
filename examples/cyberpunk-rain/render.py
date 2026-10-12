"""Procedural cyberpunk rain scene: a detective in a long coat and his
holographic AI companion on a wet rooftop, in a neon megacity at night.

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
DUR = 22.0
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

def detective_poly(fx, fy, s=1.0):
    """Long coat with turned-up collar, seen 3/4 from behind. Feet at (fx, fy)."""
    P = lambda pts: [(fx + x * s, fy + y * s) for x, y in pts]
    coat = P([(-30, -228), (-46, -214), (-50, -150), (-52, -60), (-56, -38), (-20, -34),
              (-14, -40), (14, -40), (22, -34), (54, -38), (50, -70), (48, -150), (44, -214), (28, -228)])
    collar = P([(-26, -236), (-30, -222), (26, -222), (22, -240), (12, -246), (-14, -246)])
    legs = [P([(-20, -36), (-6, -36), (-6, 0), (-22, 0)]), P([(6, -36), (20, -36), (22, 0), (6, 0)])]
    head = (fx - 14 * s, fy - 270 * s, fx + 14 * s, fy - 236 * s)
    return coat, collar, legs, head


def companion_mask(fx, fy, t, s=1.0):
    """Holographic companion facing left; her arm rises toward him."""
    S = 2
    w, h = 220, 320
    img = Image.new("L", (w * S, h * S), 0)
    d = ImageDraw.Draw(img)
    ox, oy = 110, 310  # feet inside the local canvas

    def P(pts):
        return [((ox + x * s) * S, (oy + y * s) * S) for x, y in pts]

    d.ellipse([(ox - 12 * s) * S, (oy - 252 * s) * S, (ox + 12 * s) * S, (oy - 222 * s) * S], fill=255)
    d.polygon(P([(-15, -244), (-13, -256), (0, -260), (14, -254), (16, -232), (10, -226), (8, -244)]), fill=255)
    d.polygon(P([(-5, -224), (5, -224), (6, -212), (-6, -212)]), fill=255)
    # dress
    d.polygon(P([(-22, -212), (22, -212), (18, -150), (14, -120), (30, -40), (-30, -40), (-14, -120), (-18, -150)]),
              fill=255)
    d.polygon(P([(-14, -42), (-4, -42), (-6, 0), (-13, 0)]), fill=255)
    d.polygon(P([(4, -42), (14, -42), (13, 0), (6, 0)]), fill=255)
    # back arm, resting
    d.line(P([(18, -206), (24, -150), (20, -110)]), fill=255, width=int(7 * s * S), joint="curve")
    # front arm, lifting toward his face
    lift = smooth(8.5, 13.5, t) * (1 - 0.15 * smooth(17.5, 19.5, t))
    sh = (-18, -206)
    el = (sh[0] - 46 * math.sin(math.radians(10 + 75 * lift)), sh[1] + 46 * math.cos(math.radians(10 + 75 * lift)))
    hd = (el[0] - 42 * math.sin(math.radians(20 + 110 * lift)), el[1] + 42 * math.cos(math.radians(20 + 110 * lift)))
    d.line(P([sh, el, hd]), fill=255, width=int(7 * s * S), joint="curve")
    r = 5 * s * S
    hx, hy = P([hd])[0]
    d.ellipse([hx - r, hy - r, hx + r, hy + r], fill=255)
    img = img.resize((w, h), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0, (int(fx - ox), int(fy - oy))


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
        # rain particles in normalised space: x, y, depth
        n = 2200
        self.rain = np.stack([rng.uniform(-0.2, 1.2, n), rng.uniform(0, 1, n), rng.uniform(0.15, 1, n)], 1)
        self.rain_speed = rng.uniform(0.9, 1.3, n)
        self.splash_rng_seed = SEED + 7
        self.grain = rng.normal(0, 1, (8, H, W)).astype(np.float32)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        self.vignette = np.clip(1.15 - 0.55 * r ** 1.8, 0.2, 1)[..., None].astype(np.float32)
        self.xx, self.yy = xx, yy

    # -- spinner flying through the frame --
    def spinner(self, d, glow, t, cam):
        if not (5.0 < t < 15.0):
            return
        u = (t - 5.0) / 10.0
        x = -120 + u * (W + 300) - cam * 0.5
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

    def frame(self, i):
        t = i / FPS
        rng = np.random.default_rng(1000 + i)
        cam = 260 * smooth(0, DUR, t)  # slow lateral drift
        img = self.sky[:, :W].copy()

        # city layers with parallax; giant ad sits on the middle layer
        for li, (rgb, a, par) in enumerate(self.layers):
            off = int(cam * par)
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
                # ad casts light into the rain haze
                img += (np.clip(1 - np.abs(self.xx - (adx + 120)) / 380, 0, 1) ** 2 * 0.06)[..., None] * c1

        # atmosphere: fog band over the horizon
        fogb = np.exp(-((self.yy - (HORIZON - 30)) / 120) ** 2)[..., None] * 0.18
        img = img * (1 - fogb) + FOG * fogb

        pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        glow_layer = Image.new("RGB", (W, H), (0, 0, 0))
        d = ImageDraw.Draw(pil)
        g = ImageDraw.Draw(glow_layer)
        self.spinner(d, g, t, cam)

        # rooftop ledge
        d.rectangle([0, HORIZON, W, H], fill=(6, 7, 10))
        img = np.asarray(pil, dtype=np.float32) / 255.0 + np.asarray(glow_layer, dtype=np.float32) / 255.0

        # characters
        kx = 560 - cam * 0.9 * 0.25
        fy = HORIZON + 46
        comp_x = kx + 92
        coat, collar, legs, head = detective_poly(kx, fy)
        cm, (cox, coy) = companion_mask(comp_x, fy + 4, t, 0.92)
        holo_col = np.array([0.55, 0.45, 1.0]) * (0.5 + 0.5 * math.sin(t * 0.8)) + np.array([0.2, 0.85, 1.0]) * (0.5 - 0.5 * math.sin(t * 0.8))

        S = 2
        km = Image.new("L", (W * S, H * S), 0)
        kd = ImageDraw.Draw(km)
        sc = lambda pts: [(x * S, y * S) for x, y in pts]
        kd.polygon(sc(coat), fill=255)
        kd.polygon(sc(collar), fill=255)
        for leg in legs:
            kd.polygon(sc(leg), fill=255)
        kd.ellipse([v * S for v in head], fill=255)
        km = km.resize((W, H), Image.LANCZOS)
        kmask = np.asarray(km, dtype=np.float32)[..., None] / 255.0
        # rim light from the hologram (right side) and the city (left)
        kedge = np.asarray(km.filter(ImageFilter.FIND_EDGES), dtype=np.float32)[..., None] / 255.0
        side = np.clip((self.xx - kx) / 40, -1, 1)[..., None]
        rim = kedge * (np.where(side > 0, 1.0, 0.0) * holo_col * 1.3 + np.where(side <= 0, 1.0, 0.0) * np.array([1.0, 0.55, 0.25]) * 0.35)
        figure_col = np.array([0.025, 0.028, 0.035])
        img = img * (1 - kmask) + figure_col * kmask + rim

        # companion hologram (additive, translucent, scanlines, glitch)
        hh, ww = cm.shape
        comp = np.zeros((H, W), np.float32)
        y0, x0 = max(coy, 0), max(cox, 0)
        y1, x1 = min(coy + hh, H), min(cox + ww, W)
        comp[y0:y1, x0:x1] = cm[y0 - coy:y1 - coy, x0 - cox:x1 - cox]
        if rng.random() < 0.08:  # glitch slice
            gy, gh = int(rng.integers(y0, y1 - 12)), int(rng.integers(3, 12))
            comp[gy:gy + gh] = np.roll(comp[gy:gy + gh], int(rng.integers(-14, 14)), 1)
        flick = 0.85 + 0.15 * math.sin(t * 21) + (-0.4 if rng.random() < 0.03 else 0)
        scan = 0.65 + 0.35 * (np.sin((self.yy + t * 30) * 1.4) > -0.2)
        comp_img = Image.fromarray((comp * 255).astype(np.uint8))
        halo = np.asarray(comp_img.filter(ImageFilter.GaussianBlur(10)), dtype=np.float32) / 255.0
        cedge = np.asarray(comp_img.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
        vert = np.clip((self.yy - (fy - 240)) / 260, 0, 1)  # brighter at the face
        body = comp * scan * (0.55 - 0.25 * vert)
        img = img * (1 - comp[..., None] * 0.35) + ((body + cedge * 0.8)[..., None] * holo_col + halo[..., None] * holo_col * 0.45) * flick
        # she lights his coat and the floor
        light = np.exp(-(((self.xx - comp_x) / 140) ** 2 + ((self.yy - (fy - 120)) / 180) ** 2))[..., None]
        img += light * holo_col * 0.07 * flick

        # wet floor reflection with ripples
        top = img[:HORIZON]
        nref = H - HORIZON
        src_rows = HORIZON - 1 - np.arange(nref)
        ripple = (np.sin(np.arange(nref)[:, None] * 0.35 + t * 6 + self.xx[0][None, :] * 0.02) * (2 + np.arange(nref)[:, None] * 0.03)).astype(np.int32)
        cols = np.clip(self.xx[0][None, :].astype(np.int32) + ripple, 0, W - 1)
        refl = top[src_rows[:, None], cols]
        refl_img = Image.fromarray((np.clip(refl, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2.2))
        refl = np.asarray(refl_img, dtype=np.float32) / 255.0
        fade = np.linspace(0.55, 0.25, nref)[:, None, None]
        img[HORIZON:] = img[HORIZON:] * (1 - fade) + refl * fade + np.array([0.01, 0.012, 0.016])
        # figures stand on the floor: redraw their lower parts over the reflection boundary
        img[HORIZON:] = img[HORIZON:] * (1 - kmask[HORIZON:]) + (figure_col * kmask[HORIZON:] + rim[HORIZON:])

        # rain
        rain = Image.new("L", (W, H), 0)
        rd = ImageDraw.Draw(rain)
        wind = -0.12
        for (x, y, z), sp in zip(self.rain, self.rain_speed):
            yy = (y + t * sp * (0.9 + z * 1.4)) % 1.0
            px = (x + wind * yy * z + t * wind * 0.4) * W
            py = yy * (H + 80) - 40
            L = 8 + 34 * z
            b = int(40 + 150 * z)
            rd.line([(px, py), (px + wind * L, py + L)], fill=b, width=1 if z < 0.75 else 2)
        # splashes on the floor + ripples
        for _ in range(70):
            sx = rng.uniform(0, W)
            sy = rng.uniform(HORIZON + 4, H - BAR + 20)
            r = rng.uniform(1.5, 4.5)
            rd.ellipse([sx - r * 2.2, sy - r * 0.5, sx + r * 2.2, sy + r * 0.5], outline=int(rng.uniform(70, 160)))
        rain_a = np.asarray(rain.filter(ImageFilter.GaussianBlur(0.6)), dtype=np.float32)[..., None] / 255.0
        rain_tint = np.array([0.62, 0.72, 0.85]) + light * holo_col * 2.0
        img = img + rain_a * rain_tint * 0.42

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

        # slow push-in
        z = 1.0 + 0.07 * smooth(0, DUR, t)
        out = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        if z > 1.0005:
            cw, ch = W / z, H / z
            ox, oy = (W - cw) * (0.5 + 0.3 * smooth(0, DUR, t)), (H - ch) * 0.6
            out = out.resize((W, H), Image.BICUBIC, box=(ox, oy, ox + cw, oy + ch))
        img = np.asarray(out, dtype=np.float32) / 255.0

        # chromatic aberration, grain, vignette
        img[..., 0] = np.roll(img[..., 0], 2, axis=1)
        img[..., 2] = np.roll(img[..., 2], -2, axis=1)
        img = img * self.vignette + self.grain[i % 8][..., None] * 0.022

        # title cards
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
        a = smooth(1.5, 3.5, t) * (1 - smooth(5.5, 7.0, t))
        if a <= 0:
            return
        f = ImageFont.truetype(FONT, 34)
        txt = "L O S   A N G E L E S   ·   2 0 4 9"
        w = dr.textlength(txt, font=f)
        dr.text(((W - w) / 2, 300), txt, font=f, fill=tuple(int(255 * a * c) for c in (0.95, 0.75, 0.55)))

    def subtitle(self, dr, t):
        lines = [
            (10.0, 13.0, "“The rain can’t touch me.”"),
            (14.0, 17.5, "“But I can still be here with you.”"),
        ]
        f = ImageFont.truetype(FONT, 22)
        for s, e, txt in lines:
            a = smooth(s, s + 0.5, t) * (1 - smooth(e - 0.5, e, t))
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
# Audio: rain bed, synth pad drone, low "braam" swell, spinner fly-by
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
    u = np.clip((t - 5) / 10, 0, 1)
    fly_env = np.exp(-((t - 10.0) / 2.6) ** 2)
    hum = np.sin(2 * np.pi * (180 - 60 * u) * t) * 0.3 + (onepole(noise, 2000, sr) - onepole(noise, 500, sr))
    fly = hum * fly_env * 0.35

    # sparse piano-like high notes (sine bells) when she reaches out
    bells = np.zeros(n)
    for start, f in ((9.0, 587.33), (11.0, 698.46), (13.0, 880.0), (15.5, 659.25), (18.0, 587.33)):
        m = t >= start
        tt = t[m] - start
        bells[m] += (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt)) * np.exp(-tt / 2.2) * 0.07

    fade = np.clip(t / 2.0, 0, 1) * np.clip((DUR - t) / 2.0, 0, 1)
    L = (rain + pad + br + fly * (1 - u) + bells) * fade
    R = (rain[::-1] * 0.9 + rain * 0.1 + pad + br + fly * u + bells) * fade
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
