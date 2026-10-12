"""Rain on an apartment window.

A spinner sweeps its searchlight across the rainy megacity, banks round and
locks the beam onto one floor-to-ceiling window. Inside, the light floods the
room and slowly dies away. A man stands at the glass, staring at the rain,
while the hologram of a woman flickers through the room behind him: she
touches him, glitches away to the sofa with a book, to the dining table with a
glass of red wine, and back to his side.

Same 2D style and toolkit as render.py (numpy + Pillow + ffmpeg); the city is
the same procedural skyline. Room and shots planned with ideas from the
interior-design-expert (zoning, circulation, layered light) and cinematography
(backlit silhouettes, frame within frame, motivated practicals) skills.

Usage:
  python3 apartment.py out.mp4 [--range 0 12]
  python3 apartment.py still.png --still 5,12,20
"""

import math
import os
import subprocess
import sys
import wave
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import render as R
from render import BAR, FPS, H, W, building_layer, hologram_ad, ik, onepole, rot, smooth

DUR = 40.0
SEED = 2049
T_CUT = 9.7  # exterior -> interior dissolve start
T_IN = 10.4  # dissolve end


def lerp(a, b, u):
    return a + (b - a) * u


def lerp2(a, b, u):
    return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)


def fbm(shape, seed, scales=(4, 8, 16, 32, 64), gains=(1, 0.5, 0.25, 0.13, 0.07)):
    rng = np.random.default_rng(seed)
    hgt, wid = shape
    out = np.zeros(shape, np.float32)
    for s, g in zip(scales, gains):
        small = rng.random((max(2, hgt * s // wid + 2), s + 2)).astype(np.float32)
        img = Image.fromarray((small * 255).astype(np.uint8)).resize((wid, hgt), Image.BICUBIC)
        out += np.asarray(img, np.float32) / 255.0 * g
    return out / sum(gains)


def rain_intensity(t):
    return 0.45 + 0.55 * smooth(6.0, 34.0, t)


def to_f(img):
    return np.asarray(img, np.float32) / 255.0


def to_u8(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def blur(a, r):
    return to_f(to_u8(a).filter(ImageFilter.GaussianBlur(r)))


def make_raster(width, height, to_s, z, S=2):
    """Rasterise vector shapes into an anti-aliased float mask."""

    def raster(shapes):
        m = Image.new("L", (width * S, height * S), 0)
        d = ImageDraw.Draw(m)
        for sh in shapes:
            if sh[0] == "poly":
                d.polygon([(x * S, y * S) for x, y in map(to_s, sh[1])], fill=255)
            elif sh[0] == "ell":
                cx_, cy_, rx, ry = sh[1]
                x_, y_ = to_s((cx_, cy_))
                d.ellipse([(x_ - rx * z) * S, (y_ - ry * z) * S, (x_ + rx * z) * S, (y_ + ry * z) * S], fill=255)
            else:
                pts = [(x * S, y * S) for x, y in map(to_s, sh[1])]
                wdt = max(1, int(sh[2] * z * S))
                d.line(pts, fill=255, width=wdt, joint="curve")
                for x, y in (pts[0], pts[-1]):
                    d.ellipse([x - wdt / 2, y - wdt / 2, x + wdt / 2, y + wdt / 2], fill=255)
        return to_f(m.resize((width, height), Image.LANCZOS))

    return raster


# ----------------------------------------------------------------------------
# The city (same skyline as the umbrella film)
# ----------------------------------------------------------------------------

class City:
    def __init__(self, rng):
        self.sky = R.sky()
        LW = W + 600
        self.layers = [
            building_layer(rng, 180, 380, R.HORIZON - 40, (0.05, 0.06, 0.08), 0.6, 0.10, 0.25, LW) + (0.15,),
            building_layer(rng, 140, 300, R.HORIZON - 10, (0.025, 0.03, 0.045), 0.3, 0.12, 0.35, LW) + (0.35,),
            building_layer(rng, 60, 200, R.HORIZON + 20, (0.015, 0.02, 0.03), 0.12, 0.08, 0.5, LW) + (0.7,),
        ]
        self.ad_mask, self.ad_edge = hologram_ad(LW)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.yy, self.xx = yy, xx

    def render(self, t, drift, rng):
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
        fogb = np.exp(-((self.yy - (R.HORIZON - 30)) / 120) ** 2)[..., None] * 0.18
        img = img * (1 - fogb) + R.FOG * fogb
        return img


def rain_streaks(t, z, inten, seeds, boost=None, n_max=2200, bright=0.42):
    rl = Image.new("L", (W, H), 0)
    rd = ImageDraw.Draw(rl)
    wind = -0.12
    n = int(len(seeds) * (0.35 + 0.65 * inten))
    for x, y, dz, sp in seeds[:n]:
        yy = (y + t * sp * (0.9 + dz * 1.4)) % 1.0
        px = (x + wind * yy * dz + t * wind * 0.4) * W
        py = yy * (H + 80) - 40
        L = (8 + 34 * dz) * z ** 0.5
        rd.line([(px, py), (px + wind * L, py + L)], fill=int(40 + 150 * dz), width=1 if dz * z ** 0.5 < 0.75 else 2)
    a = to_f(rl.filter(ImageFilter.GaussianBlur(0.6)))
    tint = np.array([0.62, 0.72, 0.85])
    if boost is not None:
        return a[..., None] * tint * bright + (a * boost)[..., None] * np.array([1.0, 0.98, 0.9]) * 2.2
    return a[..., None] * tint * bright


# ----------------------------------------------------------------------------
# The spinner: flies about, banks round, locks its searchlight on the window
# ----------------------------------------------------------------------------

TARGET = (1000.0, 349.0)  # the apartment window in exterior world coords


def spinner_state(t):
    """position, scale, facing (+1 right, -1 left, between = turning), roll, beam aim point."""
    if t < 3.4:  # far, crossing left to right, sweeping the streets
        u = t / 3.4
        p = (lerp(-90, 640, u), 215 - 40 * math.sin(u * 3.0))
        s, face, roll = 0.75, 1.0, 0.04 * math.sin(t * 2)
        aim = (p[0] + 230, p[1] + 260 + 90 * math.sin(t * 1.7))
    elif t < 6.0:  # banks hard and comes back toward us
        u = smooth(3.4, 6.0, t)
        p = (lerp(640, 690, u) + 120 * math.sin(math.pi * u), lerp(175, 248, u) - 30 * math.sin(math.pi * u))
        s = lerp(0.75, 1.45, u)
        face = math.cos(math.pi * u)  # yaw through 180 degrees
        roll = -0.22 * math.sin(math.pi * u)
        aim = (p[0] + 260 * face, p[1] + 230 - 60 * math.sin(math.pi * u))
    elif t < 7.6:  # completes the turn, sweeps across the tower
        u = smooth(6.0, 7.6, t)
        p = (lerp(690, 720, u), lerp(248, 238, u))
        s = lerp(1.45, 1.6, u)
        face = -math.cos(math.pi * u)
        roll = 0.12 * math.sin(math.pi * u)
        aim = (lerp(p[0] - 200, TARGET[0] - 120, u), lerp(p[1] + 240, TARGET[1] + 60, u))
    else:  # hovering, beam locked on the window
        u = smooth(7.6, 8.4, t)
        bob = 4 * math.sin(t * 1.7)
        p = (720 + 3 * math.sin(t * 0.8), 238 + bob)
        s, face, roll = 1.6, 1.0, 0.03 * math.sin(t * 1.3)
        aim = lerp2((TARGET[0] - 120, TARGET[1] + 60), (TARGET[0] + 2 * math.sin(t * 2), TARGET[1] + 2 * math.cos(t * 1.6)), u)
    return p, s, face, roll, aim


def spinner_shapes(p, s, face, roll):
    hull = [(-38, 0), (-30, -8), (-8, -13), (14, -12), (34, -5), (44, 0), (38, 6), (12, 9), (-28, 8)]
    canopy = [(-6, -12), (6, -19), (22, -16), (30, -6), (12, -10)]
    fin = [(-34, -4), (-46, -15), (-40, -2)]
    T = lambda pts: rot([(p[0] + x * s * face, p[1] + y * s) for x, y in pts], p, roll)
    return {
        "hull": [("poly", T(hull)), ("poly", T(fin))],
        "canopy": [("poly", T(canopy))],
        "head": T([(43, 0)])[0],
        "tail": T([(-37, -1)])[0],
        "under": T([(-26, 9), (30, 8)]),
    }


# ----------------------------------------------------------------------------
# Rain on the window glass (simulated once, snapshot per frame)
# ----------------------------------------------------------------------------

WIN = (260, 150, 1180, 598)  # window wall in room coords


class GlassRain:
    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        self.X0, self.Y0, self.X1, self.Y1 = WIN[0] - 20, WIN[1] - 20, WIN[2] + 20, WIN[3] + 20
        n0 = 1500
        self.bx = rng.uniform(self.X0, self.X1, n0)
        self.by = rng.uniform(self.Y0, self.Y1, n0)
        self.br = 0.6 + 2.6 * rng.random(n0) ** 2.2
        self.riv = []
        self.snaps = []
        dt = 1.0 / FPS
        t = 0.0
        for _ in range(int(DUR * FPS) + 1):
            self.snaps.append(self.snapshot(t))
            for _ in range(2):
                self.step(t, dt / 2, rng)
                t += dt / 2

    def snapshot(self, t):
        rv = [(d["x"], d["y"], d["r"], d["slide"]) for d in self.riv]
        tr = [np.array([(x, y, t - tt) for x, y, tt in d["trail"]], np.float32) for d in self.riv]
        return (np.stack([self.bx, self.by, self.br], 1).astype(np.float32), rv, [a for a in tr if len(a) > 1])

    def step(self, t, dt, rng):
        inten = rain_intensity(t)
        n = rng.poisson(300 * inten * dt)
        if n:
            self.bx = np.concatenate([self.bx, rng.uniform(self.X0, self.X1, n)])
            self.by = np.concatenate([self.by, rng.uniform(self.Y0, self.Y1, n)])
            self.br = np.concatenate([self.br, 0.6 + 2.8 * rng.random(n) ** 2.2])
        if len(self.bx) > 2600:
            k = len(self.bx) - 2600
            self.bx, self.by, self.br = self.bx[k:], self.by[k:], self.br[k:]
        if rng.random() < 3.5 * inten * dt:  # heavy beads let go and start running
            big = np.where(self.br > 2.3)[0]
            if len(big):
                j = int(rng.choice(big))
                self.riv.append(dict(x=float(self.bx[j]), y=float(self.by[j]), r=float(self.br[j]) + 0.8, v=0.0, slide=True,
                                     timer=rng.uniform(0.2, 0.9), ph=rng.uniform(0, 6.28), trail=[]))
                self.br[j] = 0
        if rng.random() < 1.6 * inten * dt:
            self.riv.append(dict(x=rng.uniform(self.X0, self.X1), y=self.Y0 + rng.uniform(0, 40), r=rng.uniform(2.4, 3.6),
                                 v=0.0, slide=True, timer=rng.uniform(0.3, 1.2), ph=rng.uniform(0, 6.28), trail=[]))
        dead = []
        for d in self.riv:
            d["timer"] -= dt
            if d["timer"] < 0:  # stick-slip: water snags on the glass, then lets go
                d["slide"] = not d["slide"]
                d["timer"] = rng.uniform(0.25, 1.1) if d["slide"] else rng.uniform(0.05, 0.6) / (d["r"] / 2.5)
            target = (18 + 30 * d["r"]) if d["slide"] else 0.0
            d["v"] += (target - d["v"]) * min(1.0, dt * 9)
            d["y"] += d["v"] * dt
            d["x"] += math.sin(d["ph"] + d["y"] * 0.04) * 0.3 * (d["v"] / 100) + rng.normal(0, 0.12)
            near = np.where((np.abs(self.bx - d["x"]) < d["r"] + 2.5) & (np.abs(self.by - d["y"]) < d["r"] + 2.5) & (self.br > 0))[0]
            if len(near):
                d["r"] = min(6.5, math.sqrt(d["r"] ** 2 + 0.35 * float(np.sum(self.br[near] ** 2))))
                self.br[near] = 0
            d["r"] = max(1.3, d["r"] - 0.035 * dt * d["v"] / 10)
            if d["v"] > 10 and rng.random() < 0.08:
                self.bx = np.append(self.bx, d["x"] + rng.normal(0, 0.8))
                self.by = np.append(self.by, d["y"] - d["r"] * 2)
                self.br = np.append(self.br, rng.uniform(0.5, 1.1))
            if not d["trail"] or abs(d["trail"][-1][1] - d["y"]) > 3:
                d["trail"].append((d["x"], d["y"], t))
            d["trail"] = [q for q in d["trail"] if t - q[2] < 3.0]
            if d["y"] > self.Y1:
                dead.append(d)
        for d in dead:
            self.riv.remove(d)
        keep = self.br > 0
        self.bx, self.by, self.br = self.bx[keep], self.by[keep], self.br[keep]


# ----------------------------------------------------------------------------
# The room: one-point perspective toward the window wall
#   living zone left (sofa under the window, rug, coffee table, reading lamp),
#   dining zone right and nearer (table, pendant), a clear path to the glass.
# ----------------------------------------------------------------------------

RW, RH = 1440, 810
VPX, VPY = 720, 330
K_D = 0.565  # depth falloff: scale(1) = window wall


def scale_at(D):
    return 1.0 / (1.0 + K_D * D)


def fpt(X, D, h=0.0):
    """Room point: lateral X (near-plane px), depth D (0 near .. 1 window), height h (px)."""
    s = scale_at(D)
    return (VPX + X * s, VPY + (419 - h) * s)


LAMP_X, LAMP_D = -660, 0.93
PENDANT_X, PENDANT_D = 450, 0.55
MAN = dict(X=40.0, D=0.9)


def box_side(X0, X1, D0, D1, h0, h1):
    """Front face quad of a box (X0..X1 at depth D0) from height h0 to h1."""
    return [fpt(X0, D0, h1), fpt(X1, D0, h1), fpt(X1, D0, h0), fpt(X0, D0, h0)]


def top_face(X0, X1, D0, D1, h):
    return [fpt(X0, D1, h), fpt(X1, D1, h), fpt(X1, D0, h), fpt(X0, D0, h)]


def build_room():
    """Static plates: walls/floor/ceiling (window left black) and furniture layers (RGBA)."""
    yy, xx = np.mgrid[0:RH, 0:RW].astype(np.float32)
    noise = fbm((RH, RW), 3)
    rng = np.random.default_rng(11)
    fine = blur(rng.random((RH, RW)).astype(np.float32), 0.8)

    plate = Image.new("RGB", (RW, RH), (0, 0, 0))
    d = ImageDraw.Draw(plate)
    wl, wt, wr, wb = WIN
    ceil = [(0, 0), (RW, 0), (RW, fpt(720, 0)[1] * 0 + 48), (wr, wt), (wl, wt), (0, 48)]
    floor = [(wl, wb), (wr, wb), (RW, 749), (RW, RH), (0, RH), (0, 749)]
    lwall = [(0, 48), (wl, wt), (wl, wb), (0, 749)]
    rwall = [(RW, 48), (wr, wt), (wr, wb), (RW, 749)]
    masks = {}
    for name, poly in (("ceil", ceil), ("floor", floor), ("lwall", lwall), ("rwall", rwall)):
        m = Image.new("L", (RW, RH), 0)
        ImageDraw.Draw(m).polygon(poly, fill=255)
        masks[name] = to_f(m)
    img = np.zeros((RH, RW, 3), np.float32)
    conc = (0.75 + 0.35 * noise + 0.12 * (fine - 0.5))[..., None]
    # concrete walls with formwork lines that converge to the vanishing point
    lines = np.zeros((RH, RW), np.float32)
    lg = Image.new("L", (RW, RH), 0)
    ld = ImageDraw.Draw(lg)
    for h in range(0, 560, 70):
        ld.line([fpt(-720, 0, h), fpt(-720, 1, h)], fill=255, width=2)
        ld.line([fpt(720, 0, h), fpt(720, 1, h)], fill=255, width=2)
    for Dd in np.linspace(0.1, 0.9, 5):
        ld.line([fpt(-720, Dd, 0), fpt(-720, Dd, 560)], fill=200, width=1)
        ld.line([fpt(720, Dd, 0), fpt(720, Dd, 560)], fill=200, width=1)
    lines = to_f(lg)
    wall_col = np.array([0.06, 0.05, 0.045])
    img += (masks["lwall"] + masks["rwall"])[..., None] * wall_col * conc * (1 - 0.25 * lines[..., None])
    img += masks["ceil"][..., None] * np.array([0.035, 0.03, 0.027]) * conc
    # polished concrete floor with board seams toward the window
    seams = Image.new("L", (RW, RH), 0)
    sd = ImageDraw.Draw(seams)
    for X in range(-700, 720, 140):
        sd.line([fpt(X, 0), fpt(X, 1)], fill=160, width=1)
    for Dd in (0.15, 0.4, 0.7):
        sd.line([fpt(-720, Dd), fpt(720, Dd)], fill=120, width=1)
    img += masks["floor"][..., None] * np.array([0.045, 0.04, 0.038]) * conc * (1 - 0.3 * to_f(seams)[..., None])
    # window header + sill (concrete) frame the glass
    d2 = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    dd = ImageDraw.Draw(d2)
    dd.rectangle([wl - 4, wt - 16, wr + 4, wt + 2], fill=(14, 12, 11))
    dd.rectangle([wl - 4, wb - 2, wr + 4, wb + 8], fill=(18, 15, 13))
    # bookshelf on the left wall, kitchen counter + under-cabinet light on the right
    for h in (120, 230, 340):
        dd.polygon([fpt(-720, 0.25, h), fpt(-720, 0.75, h), fpt(-700, 0.75, h - 6), fpt(-700, 0.25, h - 6)], fill=(28, 22, 18))
    rngb = np.random.default_rng(4)
    for h in (126, 236, 346):
        Dd = 0.27
        while Dd < 0.72:
            w_ = rngb.uniform(0.012, 0.03)
            bh = rngb.uniform(50, 80)
            c = tuple(int(v) for v in rngb.integers(20, 70, 3))
            dd.polygon([fpt(-718, Dd, h), fpt(-718, Dd + w_, h), fpt(-718, Dd + w_, h + bh), fpt(-718, Dd, h + bh)], fill=c)
            Dd += w_ + rngb.uniform(0.0, 0.006)
    dd.polygon([fpt(720, 0.02, 0), fpt(720, 0.45, 0), fpt(720, 0.45, 260), fpt(720, 0.02, 260)], fill=(18, 14, 12))
    dd.polygon([fpt(720, 0.02, 262), fpt(720, 0.45, 262), fpt(720, 0.45, 270), fpt(720, 0.02, 270)], fill=(50, 38, 28))
    dd.polygon([fpt(720, 0.02, 400), fpt(720, 0.45, 400), fpt(720, 0.45, 540), fpt(720, 0.02, 540)], fill=(15, 12, 10))
    base = to_f(d2)

    # light: warm reading lamp pool (left), pendant pool (right), under-cabinet strip
    lx, ly = fpt(LAMP_X, LAMP_D, 330)
    lamp = np.exp(-(((xx - lx) / 330) ** 2 + ((yy - ly - 40) / 300) ** 2))[..., None]
    px, py = fpt(PENDANT_X, PENDANT_D, 380)
    pend = np.exp(-(((xx - px) / 260) ** 2 + ((yy - py - 160) / 240) ** 2))[..., None]
    room_mask = (1 - 0.0)  # everything but the window
    base = base + lamp * np.array([0.45, 0.24, 0.09]) * 0.7 + pend * np.array([0.42, 0.22, 0.08]) * 0.6
    strip = np.exp(-(((yy - fpt(720, 0.2, 395)[1]) / 5) ** 2)) * (xx > 1300)
    base += strip[..., None] * np.array([0.6, 0.35, 0.15]) * 0.5
    # window stays empty in the plate (city composited per frame)
    win = np.zeros((RH, RW), np.float32)
    win[wt:wb, wl:wr] = 1
    base = base * (1 - win[..., None])

    # ---- furniture layers (RGBA) ------------------------------------------
    def layer():
        im = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
        return im, ImageDraw.Draw(im)

    far, fd = layer()
    # rug and low coffee table in the living zone
    fd.polygon([fpt(-640, 0.62), fpt(-200, 0.62), fpt(-200, 0.95), fpt(-640, 0.95)], fill=(46, 22, 18, 255))
    fd.polygon([fpt(-620, 0.64), fpt(-220, 0.64), fpt(-220, 0.93), fpt(-620, 0.93)], outline=(70, 40, 28, 255))
    fd.polygon(top_face(-540, -300, 0.72, 0.82, 110), fill=(34, 26, 20, 255))
    fd.polygon(box_side(-540, -300, 0.72, 0.82, 92, 110), fill=(20, 15, 12, 255))
    for X in (-530, -310):
        fd.polygon(box_side(X, X + 10, 0.72, 0.72, 0, 92), fill=(16, 12, 10, 255))
    fd.polygon(top_face(-470, -420, 0.75, 0.79, 113), fill=(120, 30, 30, 255))  # a book left on it
    # sofa under the window: back against the glass, facing the room
    S0, S1 = -640, -250
    fd.polygon(box_side(S0, S1, 0.95, 0.95, 100, 210), fill=(26, 19, 15, 255))  # backrest
    fd.polygon(top_face(S0, S1, 0.86, 0.95, 100), fill=(38, 28, 22, 255))  # seat
    fd.polygon(box_side(S0, S1, 0.86, 0.86, 20, 100), fill=(22, 16, 13, 255))  # seat front
    fd.line([fpt(S0, 0.86, 100), fpt(S1, 0.86, 100)], fill=(70, 50, 36, 255), width=2)
    fd.line([fpt(S0, 0.95, 210), fpt(S1, 0.95, 210)], fill=(60, 44, 34, 255), width=2)
    fd.line([fpt((S0 + S1) / 2, 0.86, 22), fpt((S0 + S1) / 2, 0.86, 98)], fill=(16, 12, 10, 255), width=2)
    for X in (S0 - 30, S1):
        fd.polygon(box_side(X, X + 30, 0.86, 0.95, 20, 150), fill=(22, 16, 13, 255))  # arms
        fd.polygon(top_face(X, X + 30, 0.86, 0.95, 150), fill=(40, 30, 24, 255))
    # reading lamp by the sofa
    fd.line([fpt(LAMP_X, LAMP_D, 0), fpt(LAMP_X, LAMP_D, 300)], fill=(30, 26, 22, 255), width=4)
    bx, by = fpt(LAMP_X, LAMP_D, 0)
    fd.ellipse([bx - 18, by - 5, bx + 18, by + 3], fill=(22, 18, 15, 255))
    near, nd = layer()
    # dining zone: table, two chairs, wine bottle, the pendant
    T0, T1, TD0, TD1 = 300, 620, 0.46, 0.6
    TH = 215
    # chairs at the two ends, seen side-on: seat, back, legs
    for X, flip in ((T0 - 70, -1), (T1 + 25, 1)):
        Xb = X + (45 if flip > 0 else 0)
        nd.polygon(top_face(X, X + 45, 0.5, 0.56, 130), fill=(30, 22, 17, 255))
        nd.polygon(box_side(Xb, Xb + 6, 0.5, 0.5, 130, 265), fill=(22, 16, 13, 255))
        nd.line([fpt(Xb + 3, 0.5, 265), fpt(Xb + 3, 0.5, 130)], fill=(60, 42, 30, 255), width=1)
        for XX in (X + 2, X + 40):
            nd.polygon(box_side(XX, XX + 5, 0.5, 0.5, 0, 130), fill=(16, 12, 10, 255))
    nd.polygon(top_face(T0, T1, TD0, TD1, TH), fill=(44, 32, 22, 255))
    nd.polygon(box_side(T0, T1, TD0, TD0, TH - 14, TH), fill=(30, 22, 16, 255))
    nd.line([fpt(T0, TD0, TH), fpt(T1, TD0, TH)], fill=(130, 80, 44, 255), width=2)
    for X in (T0 + 10, T1 - 24):
        nd.polygon(box_side(X, X + 14, TD0, TD0, 0, TH - 14), fill=(20, 15, 12, 255))
    nd.polygon(box_side(540, 556, 0.53, 0.53, TH, TH + 90), fill=(14, 22, 16, 255))
    nd.polygon(box_side(544, 552, 0.53, 0.53, TH + 90, TH + 120), fill=(14, 22, 16, 255))
    nd.line([fpt(554, 0.53, TH + 8), fpt(554, 0.53, TH + 80)], fill=(80, 110, 90, 255), width=1)
    nd.line([fpt(PENDANT_X, PENDANT_D, 560), fpt(PENDANT_X, PENDANT_D, 400)], fill=(20, 17, 15, 255), width=2)
    far_a = to_f(far)
    near_a = to_f(near)
    # warm light lands on the furniture tops nearest the lamps
    for L in (far_a, near_a):
        lit = 0.8 + 0.9 * (lamp[..., 0] * 0.8 + pend[..., 0])
        L[..., :3] *= lit[..., None]
    # emissive shades
    e = Image.new("RGB", (RW, RH))
    ed = ImageDraw.Draw(e)
    ax, ay = fpt(LAMP_X, LAMP_D, 300)
    ed.polygon([(ax - 22, ay - 30), (ax + 22, ay - 30), (ax + 30, ay + 8), (ax - 30, ay + 8)], fill=(255, 190, 120))
    qx, qy = fpt(PENDANT_X, PENDANT_D, 400)
    ed.polygon([(qx - 12, qy - 12), (qx + 12, qy - 12), (qx + 40, qy + 18), (qx - 40, qy + 18)], fill=(255, 175, 100))
    emis = to_f(e.filter(ImageFilter.GaussianBlur(1.2))) * 1.4
    cone = np.clip(1 - np.abs(xx - qx) / (40 + (yy - qy) * 0.5), 0, 1) * ((yy > qy + 18) & (yy < fpt(0, 0.5, 215)[1]))
    base += (cone ** 2)[..., None] * np.array([0.08, 0.05, 0.025])
    return base, far_a, near_a, emis, masks


# ----------------------------------------------------------------------------
# The man at the glass, seen from behind (backlit by the city)
# ----------------------------------------------------------------------------

def man_shapes(t, tilt, breathe, sway, hand_lift):
    s = scale_at(MAN["D"])
    fx, fy = fpt(MAN["X"], MAN["D"])
    u = 520 / 240 * s
    P = lambda x, y: (fx + x * u, fy + y * u)
    chest_up = breathe * 1.2
    hip = P(sway * 0.6, -112)
    sh_l = P(-40 + sway * 0.2, -202 - chest_up)
    sh_r = P(40 + sway * 0.2, -202 - chest_up * 1.1)
    neck = P(sway * 0.15, -226 - chest_up)
    head_c = rot([P(sway * 0.1, -249 - chest_up)], neck, tilt)[0]
    out = {"pants": [], "sweater": [], "skin": [], "hair": [], "shoe": []}
    for side, bend in ((-1, 3.0), (1, 0.0)):
        h0 = (hip[0] + side * 12 * u, hip[1])
        ank = P(side * 14, -6)
        knee = (lerp(h0[0], ank[0], 0.5) - side * bend * u * 0.6, lerp(h0[1], ank[1], 0.5))
        out["pants"].append(("line", [h0, knee, ank], 23 * u))
        out["shoe"].append(("ell", (ank[0], ank[1] + 3 * u, 8 * u, 4.5 * u)))
    tor = [P(-25 + sway * 0.6, -106), P(25 + sway * 0.6, -106), P(24 + sway * 0.4, -142), P(30, -182 - chest_up),
           (sh_r[0] + 2 * u, sh_r[1] + 8 * u), (sh_r[0] - 4 * u, sh_r[1] + 1 * u), P(13, -217 - chest_up),
           P(-13, -217 - chest_up), (sh_l[0] + 4 * u, sh_l[1] + 1 * u), (sh_l[0] - 2 * u, sh_l[1] + 8 * u),
           P(-30, -182 - chest_up), P(-24 + sway * 0.4, -142)]
    out["sweater"].append(("poly", tor))
    out["sweater"].append(("ell", (sh_l[0] + 3 * u, sh_l[1] + 6 * u, 7.5 * u, 7.5 * u)))
    out["sweater"].append(("ell", (sh_r[0] - 3 * u, sh_r[1] + 6 * u, 7.5 * u, 7.5 * u)))
    out["sweater"].append(("poly", [P(-12, -214 - chest_up), P(12, -214 - chest_up), P(10, -229 - chest_up), P(-10, -229 - chest_up)]))
    # left arm hangs, slow pendulum with the weight shift
    swing = 0.04 * math.sin(t * 2 * math.pi / 9.0 + 0.8)
    el_l = (sh_l[0] - 6 * u + swing * 30, sh_l[1] + 49 * u)
    hd_l = (el_l[0] + 4 * u + swing * 45, el_l[1] + 44 * u)
    out["sweater"].append(("line", [sh_l, el_l], 13 * u))
    out["sweater"].append(("line", [el_l, lerp2(el_l, hd_l, 0.85)], 11 * u))
    out["skin"].append(("ell", (hd_l[0], hd_l[1] + 4 * u, 4.5 * u, 8 * u)))
    # right hand: in his pocket ... until he lifts it to the glass
    pocket = P(28 + sway * 0.5, -110)
    glass_pt = P(34, -196)
    hand_r = lerp2(pocket, glass_pt, hand_lift)
    if hand_lift < 0.02:
        el_r = P(46, -152)
    else:
        el_r = ik(sh_r, hand_r, 50 * u, 47 * u)
        el_r = (max(el_r[0], sh_r[0] + 4 * u), el_r[1])
    out["sweater"].append(("line", [sh_r, el_r, hand_r], 12.5 * u))
    if hand_lift > 0.3:
        hx, hy = hand_r
        palm = [(hx - 5 * u, hy + 4 * u), (hx - 6 * u, hy - 8 * u), (hx - 4 * u, hy - 17 * u), (hx - 1 * u, hy - 19 * u),
                (hx + 2 * u, hy - 18 * u), (hx + 5 * u, hy - 15 * u), (hx + 7 * u, hy - 6 * u), (hx + 5 * u, hy + 4 * u)]
        out["skin"].append(("poly", palm))
    # neck, head from behind: hair covers it, ears just show
    out["skin"].append(("line", [P(sway * 0.15, -224 - chest_up), (head_c[0], head_c[1] + 14 * u)], 12 * u))
    out["skin"].append(("ell", (head_c[0], head_c[1], 15 * u, 19 * u)))
    for side in (-1, 1):
        ex, ey = rot([(head_c[0] + side * 15 * u, head_c[1] + 2 * u)], head_c, tilt)[0]
        out["skin"].append(("ell", (ex, ey, 2.2 * u, 4 * u)))
    hair = [(-16, 6), (-16.5, -6), (-14, -15), (-8, -21), (0, -23), (8, -21.5), (14.5, -15), (16.5, -6), (16, 6), (12, 12), (5, 14),
            (-5, 14), (-12, 12)]
    out["hair"].append(("poly", rot([(head_c[0] + x * u, head_c[1] + y * u) for x, y in hair], head_c, tilt)))
    joints = dict(head=head_c, sh_l=sh_l, sh_r=sh_r, el_l=el_l, hand_r=hand_r,
                  neck_l=rot([(head_c[0] - 10 * u, head_c[1] + 14 * u)], head_c, tilt)[0], u=u)
    return out, joints


# ----------------------------------------------------------------------------
# The hologram: profile figure, three poses, glitch teleports
# ----------------------------------------------------------------------------

def head_profile(c, s, look, hair_sw):
    face = [(-13, -2), (-12, -14), (-6, -22), (3, -23), (10, -18), (13, -10), (14, -5), (15, -2), (19, 3), (15, 6), (16, 9),
            (15.5, 11), (16, 13), (14, 15), (15, 19), (8, 22), (2, 22), (-6, 18), (-12, 10)]
    hair = [(12, -12), (9, -20), (2, -25), (-8, -23), (-15, -16), (-17, -2), (-16 + hair_sw, 12), (-13 + hair_sw, 18),
            (-6 + hair_sw * 0.6, 19), (-4, 10), (-2, 0), (4, -6), (11, -7)]
    f = rot([(c[0] + x * s, c[1] + y * s) for x, y in face], c, look)
    h = rot([(c[0] + x * s, c[1] + y * s) for x, y in hair], c, look)
    return [("poly", f), ("poly", h)]


def mirror(shapes, cx):
    out = []
    for sh in shapes:
        if sh[0] == "poly":
            out.append(("poly", [(2 * cx - x, y) for x, y in sh[1]]))
        elif sh[0] == "ell":
            x, y, rx, ry = sh[1]
            out.append(("ell", (2 * cx - x, y, rx, ry)))
        else:
            out.append(("line", [(2 * cx - x, y) for x, y in sh[1]], sh[2]))
    return out


def her_standing(fx, fy, s, t, reach=None, look=0.0, lean=0.0):
    br = math.sin(t * 1.6) * 0.8
    pivot = (fx, fy - 112 * s)
    P = lambda pts: rot([(fx + x * s, fy + y * s) for x, y in pts], pivot, lean)
    sw = 2.0 * math.sin(t * 0.9)
    shapes = []
    shapes.append(("poly", P([(-6, -60), (1, -60), (-1, -4), (-6, -4)])))
    shapes.append(("poly", P([(3, -60), (10, -60), (11, -4), (6, -4)])))
    for x0 in (-7, 5):
        shapes.append(("poly", P([(x0, -5), (x0 + 11, -3), (x0 + 12, 0), (x0, 0)])))
    sh_far = P([(-4, -199 - br)])[0]
    shapes.append(("line", [sh_far, P([(-6 + sw * 0.2, -150)])[0], P([(-3 + sw * 0.3, -108)])[0]], 6.5 * s))
    shapes.append(("poly", P([(-6, -206 - br), (-10, -199 - br), (-12, -170), (-8, -142), (-14, -112), (-23 + sw, -44),
                              (-4 + sw * 0.6, -40), (8 + sw * 0.6, -41), (25 + sw, -46), (13, -112), (9, -142), (16, -170 - br),
                              (14, -186 - br), (8, -200 - br), (5, -206 - br)])))
    shapes.append(("poly", P([(-5, -214 - br), (5, -214 - br), (5, -203 - br), (-6, -203 - br)])))
    head_c = P([(1, -236 - br)])[0]
    shapes += head_profile(head_c, s, look + lean * 0.5, 1.2 * math.sin(t * 1.3))
    sh = P([(1, -199 - br)])[0]
    if reach is not None:
        hand = reach
        el = ik(sh, hand, 47 * s, 44 * s)
    else:
        el = P([(3, -152)])[0]
        hand = P([(6, -110)])[0]
    shapes.append(("line", [sh, el, hand], 6.5 * s))
    shapes.append(("ell", (hand[0], hand[1], 4.5 * s, 4 * s)))
    return shapes, dict(head=head_c, hand=hand)


def her_sitting(hx, hy, s, t, page, look_up):
    """On the sofa, back to the left arm, knees up, a book on them."""
    br = math.sin(t * 1.5) * 0.6
    lean = -0.26
    P = lambda pts: rot([(hx + x * s, hy + y * s) for x, y in pts], (hx, hy), lean)
    Q = lambda pts: [(hx + x * s, hy + y * s) for x, y in pts]
    shapes = []
    knee = Q([(44, -44)])[0]
    for dx in (0, 4):
        shapes.append(("line", [Q([(4 + dx, -8)])[0], (knee[0] + dx * s, knee[1]), Q([(70 + dx, -2)])[0]], 10 * s))
    shapes.append(("poly", Q([(-14, 0), (-10, -22), (10, -27), (42, -51), (48, -43), (20, -10), (12, 0)])))
    shapes.append(("poly", P([(-10, -6), (-12, -40), (-11, -70), (-8, -94 - br), (6, -94 - br), (12, -84), (14, -62), (8, -38), (10, -6)])))
    shapes.append(("poly", P([(-4, -100 - br), (5, -100 - br), (5, -92), (-5, -92)])))
    head_c = P([(1, -122 - br)])[0]
    shapes += head_profile(head_c, s, 0.42 - 0.35 * look_up, 0.8 * math.sin(t))
    hand = (knee[0] - 4 * s, knee[1] - 20 * s)
    sh = P([(0, -88)])[0]
    el = ik(sh, hand, 46 * s, 42 * s)
    shapes.append(("line", [sh, el, hand], 6 * s))
    sp = (hand[0] + 4 * s, hand[1] - 2 * s)
    left = (sp[0] - 20 * s, sp[1] - 9 * s)
    right = (sp[0] + 17 * s, sp[1] - 12 * s)
    book = [("poly", [sp, left, (left[0], left[1] + 3 * s), (sp[0], sp[1] + 3 * s), (right[0], right[1] + 3 * s), right])]
    if 0 < page < 1:
        a0 = math.atan2(right[1] - sp[1], right[0] - sp[0])
        a1 = math.atan2(left[1] - sp[1], left[0] - sp[0])
        a = a0 + (a1 - a0) * (0.5 - 0.5 * math.cos(math.pi * page))
        L = 18 * s
        tip = (sp[0] + math.cos(a) * L, sp[1] + math.sin(a) * L - 4 * s * math.sin(math.pi * page))
        book.append(("line", [sp, ((sp[0] + tip[0]) / 2, (sp[1] + tip[1]) / 2 - 3 * s), tip], 1.5 * s))
    return shapes, book, dict(head=head_c, hand=hand)


POSES = [
    (15.2, 21.4, "beside"),
    (21.75, 27.4, "sofa"),
    (27.75, 33.4, "table"),
    (33.75, 40.5, "beside2"),
]
GLITCH = 0.32


def pose_at(t):
    for a, b, name in POSES:
        if a - GLITCH <= t <= b + GLITCH:
            g_in = 1 - smooth(a - GLITCH, a, t) if t < a else 0.0
            g_out = smooth(b, b + GLITCH, t) if t > b else 0.0
            yield name, max(g_in, g_out), (t - a)


# ----------------------------------------------------------------------------
# Cameras
# ----------------------------------------------------------------------------

def ext_camera(t):
    """Exterior world (W x H design coords): zoom and centre."""
    u = smooth(6.6, 10.4, t)
    z = 1.0 + 1.9 * u ** 1.3
    cx = lerp(640, TARGET[0], u) + 4 * math.sin(t * 0.6)
    cy = lerp(360, TARGET[1], u) + 3 * math.sin(t * 0.8 + 1)
    hw, hh = W / (2 * z), H / (2 * z)
    return z, min(max(cx, hw), W - hw), min(max(cy, hh), H - hh)


def int_camera(t):
    z = 1.06 - 0.06 * smooth(T_IN, 15.0, t)
    cx, cy = 720.0, 405.0
    cx += -55 * smooth(21.4, 23.0, t) * (1 - smooth(27.4, 29.0, t)) + 45 * smooth(27.4, 29.0, t) * (1 - smooth(33.4, 35.0, t))
    u = smooth(33.6, 39.0, t)
    z += 0.13 * u
    cx = lerp(cx, 760, u)
    cy = lerp(cy, 360, u)
    cx += 1.5 * math.sin(t * 0.7)
    cy += 1.0 * math.sin(t * 0.9 + 2)
    hw, hh = W / (2 * z), H / (2 * z)
    return z, min(max(cx, hw), RW - hw), min(max(cy, hh), RH - hh)


def beam_strength(t):
    return smooth(T_CUT - 0.2, T_IN, t) * (1 - smooth(11.4, 14.6, t)) * (0.93 + 0.07 * math.sin(t * 31))


# ----------------------------------------------------------------------------
# Renderer
# ----------------------------------------------------------------------------

HOLO_A = np.array([0.42, 0.36, 1.0])
HOLO_B = np.array([0.35, 0.85, 1.0])


class Scene:
    def __init__(self):
        rng = np.random.default_rng(SEED)
        self.city = City(rng)
        self.room, self.far, self.near, self.emis, self.masks = build_room()
        self.glass = GlassRain()
        self.rain_seeds = np.stack([rng.uniform(-0.2, 1.2, 2400), rng.uniform(0, 1, 2400), rng.uniform(0.15, 1, 2400),
                                    rng.uniform(0.9, 1.3, 2400)], 1)
        self.tower = self.build_tower(np.random.default_rng(31))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.xx, self.yy = xx, yy
        ryy, rxx = np.mgrid[0:RH, 0:RW].astype(np.float32)
        self.rxx, self.ryy = rxx, ryy
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        self.vignette = np.clip(1.12 - 0.5 * r ** 1.8, 0.25, 1)[..., None].astype(np.float32)
        self.grain = rng.normal(0, 1, (8, H, W)).astype(np.float32)
        self.knit = fbm((RH, RW), 21, scales=(64, 128, 256), gains=(1, 0.6, 0.4))

    # ---------------------------------------------------------------- exterior
    def build_tower(self, rng):
        """Apartment tower facade (world coords): window rects with interior light."""
        wins = []
        for row, fy in enumerate(range(52, 760, 26)):
            for col in range(12):
                x0 = 766 + col * 39
                r = (x0 + 5, fy + 5, x0 + 34, fy + 21)
                if 940 < x0 + 20 < 1060 and 320 < fy + 13 < 380:
                    continue  # the target window is drawn on its own
                lit = rng.random()
                kind = "warm" if lit < 0.08 else ("cool" if lit < 0.13 else "dark")
                wins.append((r, kind, rng.uniform(0.35, 1.0)))
        return wins

    def exterior(self, t, i, rng):
        z, ccx, ccy = ext_camera(t)
        to_s = lambda p: ((p[0] - ccx) * z + W / 2, (p[1] - ccy) * z + H / 2)
        raster = make_raster(W, H, to_s, z)
        inten = rain_intensity(t)
        drift = 120 + 40 * smooth(0, 10, t) + (ccx - 640) * 0.5
        city = self.city.render(t, drift, rng)
        hw, hh = W / (2 * z), H / (2 * z)
        box = (max(0.0, ccx - hw), max(0.0, ccy - hh), min(float(W), ccx + hw), min(float(H), ccy + hh))
        cimg = to_u8(city).resize((W, H), Image.BICUBIC, box=box)
        if z > 1.1:
            cimg = cimg.filter(ImageFilter.GaussianBlur((z - 1) * 1.4))
        img = to_f(cimg)

        # spinner and beam
        p, s, face, roll, aim = spinner_state(t)
        sp = spinner_shapes(p, s, face, roll)
        head = to_s(sp["head"])
        tgt = to_s(aim)
        dx, dy = tgt[0] - head[0], tgt[1] - head[1]
        L = math.hypot(dx, dy) + 1e-3
        nx, ny = -dy / L, dx / L
        spread = 0.2 * L
        bm = Image.new("L", (W, H), 0)
        ImageDraw.Draw(bm).polygon([(head[0] + nx * 3, head[1] + ny * 3), (head[0] - nx * 3, head[1] - ny * 3),
                                    (tgt[0] - nx * spread, tgt[1] - ny * spread), (tgt[0] + nx * spread, tgt[1] + ny * spread)], fill=255)
        bm = to_f(bm.filter(ImageFilter.GaussianBlur(6 * z ** 0.5)))
        along = np.clip(((self.xx - head[0]) * dx + (self.yy - head[1]) * dy) / (L * L), 0, 1)
        beam = bm * (0.55 * (1 - along) ** 1.4 + 0.12)

        # the apartment tower in front of the skyline (vector, stays crisp when we push in)
        lock = smooth(7.6, 8.4, t)
        tower_poly = [(760, 40), (1240, 40), (1240, 760), (760, 760)]
        tm = raster([("poly", tower_poly)])
        facade = np.array([0.03, 0.033, 0.04]) * (0.8 + 0.3 * self.knit[:H, :W, None])
        img = img * (1 - tm[..., None]) + facade * tm[..., None]
        wl = Image.new("RGB", (W, H), (0, 0, 0))
        wd = ImageDraw.Draw(wl)
        for (x0, y0, x1, y1), kind, k in self.tower:
            a_, b_ = to_s((x0, y0)), to_s((x1, y1))
            if kind == "warm":
                c = (int(150 * k), int(90 * k), int(45 * k))
            elif kind == "cool":
                c = (int(50 * k), int(80 * k), int(110 * k))
            else:
                c = (int(8 + 10 * k), int(9 + 11 * k), int(13 + 16 * k))  # dark glass catching the neon
            wd.rectangle([a_[0], a_[1], b_[0], b_[1]], fill=c)
        # concrete fins and slab edges catch a little city light
        for col in range(13):
            a_, b_ = to_s((764 + col * 39, 40)), to_s((767 + col * 39, 760))
            wd.rectangle([a_[0], a_[1], b_[0], b_[1]], fill=(26, 28, 34))
        for fy in range(52, 760, 78):
            a_, b_ = to_s((760, fy), ), to_s((1240, fy + 3))
            wd.rectangle([a_[0], a_[1], b_[0], b_[1]], fill=(22, 24, 30))
        # the target window: floor to ceiling, warm, a figure standing at the glass
        T0, T1 = to_s((948, 328)), to_s((1052, 372))
        wd.rectangle([T0[0], T0[1], T1[0], T1[1]], fill=(170, 100, 50))
        img += to_f(wl) * tm[..., None]
        man = raster([("ell", (1006, 334, 2.2, 2.6)), ("poly", [(1002, 337), (1010, 337), (1011, 360), (1008, 372), (1004, 372), (1001, 360)])])
        img = img * (1 - man[..., None] * 0.9) + man[..., None] * np.array([0.03, 0.025, 0.02])
        # beam on the facade: hot spot, glass flare when it hits the window
        hs = np.exp(-(((self.xx - tgt[0]) / (34 * z)) ** 2 + ((self.yy - tgt[1]) / (22 * z)) ** 2))
        img += (hs * (0.6 + 1.6 * lock))[..., None] * np.array([1.0, 0.97, 0.9]) * 0.9
        img += beam[..., None] * np.array([0.95, 0.95, 0.9])
        # spinner body over everything (it is in front of the tower)
        hull = raster(sp["hull"])
        img = img * (1 - hull[..., None]) + hull[..., None] * np.array([0.03, 0.035, 0.045])
        hedge = to_f(to_u8(hull).filter(ImageFilter.FIND_EDGES))
        img += hedge[..., None] * np.array([0.3, 0.35, 0.45]) * 0.6
        canopy = raster(sp["canopy"])
        img += canopy[..., None] * np.array([0.06, 0.1, 0.14])
        gl = Image.new("L", (W, H), 0)
        gd = ImageDraw.Draw(gl)
        r_ = 5 * s * z
        gd.ellipse([head[0] - r_, head[1] - r_ * 0.7, head[0] + r_, head[1] + r_ * 0.7], fill=255)
        img += to_f(gl.filter(ImageFilter.GaussianBlur(1.5)))[..., None] * np.array([1.4, 1.35, 1.2])
        if int(t * 3) % 2 == 0:
            tl = to_s(sp["tail"])
            img += np.exp(-(((self.xx - tl[0]) / (4 * z)) ** 2 + ((self.yy - tl[1]) / (4 * z)) ** 2))[..., None] * np.array([1.6, 0.1, 0.08])
        ug = Image.new("L", (W, H), 0)
        ImageDraw.Draw(ug).line([to_s(q) for q in sp["under"]], fill=255, width=max(2, int(2 * z)))
        img += to_f(ug.filter(ImageFilter.GaussianBlur(1.2)))[..., None] * np.array([0.15, 0.7, 1.2])

        img += rain_streaks(t, z, inten, self.rain_seeds, boost=beam)
        return img

    # ---------------------------------------------------------------- interior
    def man_motion(self, t):
        breathe = 0.5 + 0.5 * math.sin(t * 2 * math.pi / 4.6)
        sway = 2.2 * math.sin(t * 2 * math.pi / 9.0)
        tilt = -0.02 + 0.015 * math.sin(t * 0.5) - 0.12 * smooth(18.9, 20.0, t) * (1 - smooth(20.7, 22.5, t))
        hand = smooth(35.6, 37.4, t)
        return tilt, breathe, sway, hand

    def her_pose(self, name, lt, t, mj):
        u = mj["u"]
        if name in ("beside", "beside2"):
            Dh = 0.86
            s = 0.93 * 520 * scale_at(Dh) / 260
            fx, fy = fpt(-58, Dh)
            if name == "beside":
                rest = lerp2(mj["sh_l"], (mj["sh_l"][0] + 6 * u, mj["sh_l"][1] - 2 * u), 0.5 + 0.5 * math.sin(lt * 1.3))
                neck = mj["neck_l"]
                side = (fx + 10 * s, fy - 108 * s)
                a = smooth(0.6, 1.8, lt)
                b = smooth(3.4, 4.4, lt)
                c = smooth(5.2, 6.0, lt)
                tgt = lerp2(lerp2(lerp2(side, rest, a), neck, b), side, c)
                tgt = (tgt[0], tgt[1] - 16 * s * math.sin(math.pi * a) * (1 - b))
                lean = 0.06 * a + 0.04 * b - 0.08 * c
                look = -0.1 - 0.12 * b + 0.15 * c
            else:
                tgt = lerp2((fx + 10 * s, fy - 108 * s), mj["el_l"], smooth(0.4, 1.3, lt))
                lean, look = 0.05, -0.12
            shapes, _ = her_standing(fx, fy, s, t, reach=tgt, look=look, lean=lean)
            return shapes, [], (fx, fy - 140 * s), Dh
        if name == "sofa":
            Dh = 0.905
            s = 0.93 * 520 * scale_at(Dh) / 260
            hx, hy = fpt(-610, Dh, 100)
            page = (lt - 2.3) / 0.6
            look_up = smooth(4.4, 4.9, lt) * (1 - smooth(5.3, 5.6, lt))
            shapes, book, _ = her_sitting(hx, hy, s, t, page, look_up)
            return shapes, book, (hx + 20 * s, hy - 60 * s), Dh
        # at the dining table, facing him, swirling the glass
        Dh = 0.72
        s = 0.93 * 520 * scale_at(Dh) / 260
        fx, fy = fpt(430, Dh)
        ang = lt * 2 * math.pi / 1.15
        sw = smooth(0.4, 1.0, lt)
        hand = (fx + 26 * s + 5 * s * math.cos(ang) * sw, fy - 152 * s + 2.5 * s * math.sin(ang) * sw)
        shapes, _ = her_standing(fx, fy, s, t, reach=hand, look=-0.05 + 0.1 * smooth(2.5, 3.2, lt), lean=0.02)
        stem_b = (hand[0], hand[1] + 2 * s)
        stem_t = (hand[0], hand[1] - 9 * s)
        bowl = [(stem_t[0] - 7 * s, stem_t[1] - 14 * s), (stem_t[0] + 7 * s, stem_t[1] - 14 * s), (stem_t[0] + 5 * s, stem_t[1] - 2 * s),
                (stem_t[0], stem_t[1]), (stem_t[0] - 5 * s, stem_t[1] - 2 * s)]
        slosh = 0.35 * math.sin(ang - 1.2) * sw
        lvl = stem_t[1] - 6 * s
        wine = [(stem_t[0] - 5.5 * s, lvl + slosh * 6 * s), (stem_t[0] + 5.5 * s, lvl - slosh * 6 * s), (stem_t[0] + 4.5 * s, stem_t[1] - 2.2 * s),
                (stem_t[0], stem_t[1] - 0.4 * s), (stem_t[0] - 4.5 * s, stem_t[1] - 2.2 * s)]
        glass = [("line", [stem_b, stem_t], 1.2 * s), ("poly", bowl), ("line", [(stem_b[0] - 4 * s, stem_b[1]), (stem_b[0] + 4 * s, stem_b[1])], 1 * s)]
        shapes = mirror(shapes, fx)
        extra = mirror(glass, fx) + [("wine",) + tuple(mirror([("poly", wine)], fx))]
        return shapes, extra, (fx, fy - 140 * s), Dh

    def hologram(self, img, shapes, extra, g, t, rng, raster, pos, her_light):
        m = raster(shapes)
        flick = 0.8 + 0.12 * math.sin(t * 23) + (rng.uniform(-0.45, 0) if rng.random() < 0.14 else 0)
        tear = 0.03 + 0.97 * g
        ys = np.where(m.max(axis=1) > 0)[0]
        if len(ys) > 8:
            nb = int(3 + 26 * tear) if (g > 0 or rng.random() < 0.18) else 0
            for _ in range(nb):
                gy = int(rng.integers(ys[0], ys[-1] + 1))
                gh = int(rng.integers(2, 4 + int(14 * tear)))
                if rng.random() < 0.35 * tear:
                    m[gy:gy + gh] *= rng.uniform(0, 0.3)
                else:
                    m[gy:gy + gh] = np.roll(m[gy:gy + gh], int(rng.normal(0, 3 + 70 * tear * tear)), 1)
        alpha = (1 - g) ** 1.5
        mi = to_u8(m)
        edge = to_f(mi.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(0.7)))
        halo = to_f(mi.filter(ImageFilter.GaussianBlur(8)))
        scan = 0.72 + 0.28 * np.sin((self.ryy - t * 22) * 1.9)
        fine = 0.85 + 0.15 * (self.knit > 0.5)
        py = pos[1]
        grad = np.clip((self.ryy - py + 130) / 260, 0, 1)[..., None]
        col = HOLO_A * grad + HOLO_B * (1 - grad)
        body = (m * 0.32 * scan * fine + edge * 1.1)[..., None] * col + halo[..., None] * col * 0.35
        if g > 0.02:
            sh = int(2 + 10 * g)
            body[..., 0] = np.roll(body[..., 0], sh, 1)
            body[..., 2] = np.roll(body[..., 2], -sh, 1)
            body *= 1 + 1.6 * math.sin(math.pi * g)
        img = img * (1 - 0.25 * m[..., None] * alpha) + body * flick * alpha
        for e in extra:
            if e[0] == "wine":
                wm = raster([e[1]])
                img = img * (1 - 0.4 * wm[..., None] * alpha) + wm[..., None] * np.array([1.0, 0.04, 0.1]) * alpha * (0.95 + 0.05 * flick)
            else:
                pm = raster([e])
                img += pm[..., None] * np.array([0.7, 0.8, 1.0]) * 0.7 * alpha * flick
        sx, sy = pos
        light = np.exp(-(((self.rxx - sx) / 240) ** 2 + ((self.ryy - sy) / 280) ** 2))[..., None]
        img += light * HOLO_A * 0.06 * flick * alpha
        her_light = np.maximum(her_light, light * alpha * flick)
        if g > 0.05:
            d = Image.new("L", (RW, RH), 0)
            dd = ImageDraw.Draw(d)
            for _ in range(int(60 * g)):
                bx = sx + rng.normal(0, 60)
                by = sy + rng.normal(0, 100) - 60 * g
                w_ = rng.integers(2, 10)
                dd.rectangle([bx, by, bx + w_, by + 1], fill=int(255 * rng.uniform(0.3, 1)))
            img += to_f(d)[..., None] * HOLO_B * g
        return img, her_light

    def draw_man(self, img, parts, mj, raster, her_light, beam):
        masks = {k: raster(v) for k, v in parts.items()}
        alpha = np.clip(sum(masks.values()), 0, 1)
        hx = mj["head"][0]
        col = np.zeros((RH, RW, 3), np.float32)
        rib = 0.82 + 0.18 * np.sin((self.rxx - hx) * 0.9) ** 2
        knit = 0.85 + 0.3 * (self.knit - 0.5)
        col += masks["sweater"][..., None] * np.array([0.035, 0.036, 0.04]) * (rib * knit)[..., None]
        col += masks["pants"][..., None] * np.array([0.022, 0.02, 0.022]) * knit[..., None]
        col += masks["shoe"][..., None] * np.array([0.015, 0.013, 0.012])
        col += masks["hair"][..., None] * np.array([0.02, 0.016, 0.013]) * (0.7 + 0.6 * self.knit)[..., None]
        col += masks["skin"][..., None] * np.array([0.09, 0.06, 0.05])
        # warm fill from the reading lamp behind-left of the camera
        lamp_side = np.clip(0.6 - (self.rxx - hx) / 140, 0.05, 1)
        col += alpha[..., None] * (lamp_side * 0.35)[..., None] * np.array([0.12, 0.06, 0.025])
        # cool / magenta rim from the city behind him
        edge = to_f(to_u8(alpha).filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(1.0)))
        rimc = np.array([0.45, 0.55, 0.85]) + 0.3 * np.array([0.6, -0.3, 0.1]) * (self.rxx > hx)[..., None]
        col += edge[..., None] * rimc * 0.55
        col += alpha[..., None] * her_light * HOLO_A * 0.2
        # the searchlight wraps round him
        if beam is not None:
            col += edge[..., None] * beam[..., None] * np.array([2.2, 2.2, 2.0])
        return img * (1 - alpha[..., None]) + col

    def glass_layer(self, img, snap, glare):
        """Water on the window: lens drops refract the city, trails smear it."""
        beads, rivs, trails = snap
        wl, wt, wr, wb = WIN
        tm = Image.new("L", (RW, RH), 0)
        td = ImageDraw.Draw(tm)
        for tr in trails:
            pts = [(x, y) for x, y, a in tr]
            if len(pts) > 1:
                td.line(pts, fill=200, width=3, joint="curve")
        tmask = to_f(tm.filter(ImageFilter.GaussianBlur(1.0)))
        if tmask.max() > 0:
            shifted = np.roll(np.roll(img, 3, axis=1), -2, axis=0) * 1.12
            img = img * (1 - tmask[..., None] * 0.7) + shifted * tmask[..., None] * 0.7
        src = blur(img, 1.2)
        dispx = np.zeros((RH, RW), np.float32)
        dispy = np.zeros((RH, RW), np.float32)
        lens = np.zeros((RH, RW), np.float32)
        shade = np.zeros((RH, RW), np.float32)
        spec = np.zeros((RH, RW), np.float32)
        drops = [(x, y, r * 0.8, 1.0) for x, y, r in beads] + [(x, y, r * 0.8, 1.3 if s else 1.1) for x, y, r, s in rivs]
        for x, y, rs, el in drops:
            if rs < 0.8 or not (wl < x < wr and wt < y < wb):
                continue
            ry = rs * el
            x0, x1 = int(max(0, x - rs - 1)), int(min(RW, x + rs + 2))
            y0, y1 = int(max(0, y - ry - 1)), int(min(RH, y + ry + 2))
            ox = (self.rxx[y0:y1, x0:x1] - x) / rs
            oy = (self.ryy[y0:y1, x0:x1] - y) / ry
            d2 = ox * ox + oy * oy
            inside = d2 < 1
            view = 16 + rs * 5
            dispx[y0:y1, x0:x1] = np.where(inside, -ox * (rs + view), dispx[y0:y1, x0:x1])
            dispy[y0:y1, x0:x1] = np.where(inside, -oy * (ry + view), dispy[y0:y1, x0:x1])
            lens[y0:y1, x0:x1] = np.maximum(lens[y0:y1, x0:x1], inside * 1.0)
            dd = np.sqrt(np.minimum(d2, 1))
            shade[y0:y1, x0:x1] = np.maximum(shade[y0:y1, x0:x1], inside * np.clip((dd - 0.45) / 0.55, 0, 1) * 0.8)
            spec[y0:y1, x0:x1] = np.maximum(spec[y0:y1, x0:x1], np.exp(-(((ox + 0.35) / 0.25) ** 2 + ((oy + 0.4) / 0.25) ** 2)) * inside)
        sx = np.clip(self.rxx + dispx, 0, RW - 1).astype(np.int32)
        sy = np.clip(self.ryy + dispy, 0, RH - 1).astype(np.int32)
        refr = src[sy, sx] * 1.3 + 0.015
        refr = refr * (1 - shade[..., None]) + shade[..., None] * np.array([0.02, 0.025, 0.03])
        sparkle = 0.5 + 2.5 * glare
        img = img * (1 - lens[..., None]) + refr * lens[..., None] + (spec * sparkle)[..., None] * np.array([0.85, 0.9, 1.0]) * 0.5
        return img, lens

    def interior(self, t, i, rng):
        z, ccx, ccy = int_camera(t)
        inten = rain_intensity(t)
        raster = make_raster(RW, RH, lambda p: p, 1.0)
        wl, wt, wr, wb = WIN
        bstr = beam_strength(t)

        # city through the window (parallax with the camera), rain falling out there
        drift = 220 + (ccx - 720) * 0.9 + 30 * smooth(10, 40, t)
        city = self.city.render(t, drift, rng)
        city += rain_streaks(t, 1.0, inten, self.rain_seeds, bright=0.3)
        ww, wh = wr - wl, wb - wt
        cimg = to_f(to_u8(city).resize((ww, wh), Image.BICUBIC, box=(60, 70, 1220, 640)))
        img = self.room.copy()
        img[wt:wb, wl:wr] = cimg

        # the searchlight from the spinner outside: glare on the glass, then a shaft into the room
        glare = np.zeros((RH, RW), np.float32)
        beam = None
        if bstr > 0.002:
            apx, apy = lerp(470, 330, smooth(11.4, 14.6, t)), lerp(250, 230, smooth(11.4, 14.6, t))
            core = np.exp(-(((self.rxx - apx) / 34) ** 2 + ((self.ryy - apy) / 26) ** 2))
            halo = np.exp(-(((self.rxx - apx) / 150) ** 2 + ((self.ryy - apy) / 110) ** 2)) * 0.25
            streak = np.exp(-(((self.rxx - apx) / 520) ** 2 + ((self.ryy - apy) / 5) ** 2)) * 0.5
            glare = (core + halo) * bstr
            img[wt:wb, wl:wr] += (glare[wt:wb, wl:wr, None] * np.array([1.8, 1.75, 1.6]))
            img += (streak * bstr)[..., None] * np.array([0.6, 0.75, 1.0])
        snap = self.glass.snaps[min(i, len(self.glass.snaps) - 1)]
        img, lens = self.glass_layer(img, snap, glare)
        # mullions: frame within the frame
        mm = Image.new("L", (RW, RH), 0)
        md = ImageDraw.Draw(mm)
        for x in np.linspace(wl, wr, 5):
            md.rectangle([x - 6, wt, x + 6, wb], fill=255)
        md.rectangle([wl, wt, wr, wt + 8], fill=255)
        mmask = to_f(mm)[..., None]
        img = img * (1 - mmask) + mmask * np.array([0.015, 0.016, 0.02])

        # rain light from the window on the floor, patterned by the water on the glass
        pat = blur(lens + 0.5 * to_f(Image.fromarray((lens * 0).astype(np.uint8))), 2.5)
        fl = self.masks["floor"]
        depth_fall = np.clip(1 - (self.ryy - wb) / 260, 0, 1)
        src_y = np.clip(wb - (self.ryy - wb) * 1.9, wt, wb - 1).astype(np.int32)
        src_x = np.clip(VPX + (self.rxx - VPX) * 0.75, wl, wr - 1).astype(np.int32)
        shadow = pat[src_y, src_x]
        img += (fl * depth_fall * (1 - 0.8 * shadow))[..., None] * np.array([0.1, 0.12, 0.16]) * 0.6

        # far furniture (sofa under the window, rug, lamp)
        a = self.far[..., 3:4]
        img = img * (1 - a) + self.far[..., :3] * a

        if bstr > 0.002:
            # volumetric shaft, broken by falling water, dust motes, mullion stripes on the floor
            beam_m = Image.new("L", (RW, RH), 0)
            bd = ImageDraw.Draw(beam_m)
            bd.polygon([(apx - 70, apy - 40), (apx + 90, apy + 30), (RW + 200, RH + 40), (700, RH + 60)], fill=255)
            beam_m = to_f(beam_m.filter(ImageFilter.GaussianBlur(28)))
            dist = np.sqrt((self.rxx - apx) ** 2 + (self.ryy - apy) ** 2)
            falloff = np.clip(1 - dist / 1300, 0, 1) ** 1.3
            streaks = 0.75 + 0.25 * np.sin((self.rxx - apx) * 0.08 + (self.ryy - apy) * 0.02 - t * 3) * np.sin(self.rxx * 0.013 + t)
            beam = beam_m * falloff * streaks * bstr
            stripes = (np.abs(((self.rxx - VPX) / (self.ryy - VPY + 1e-3) * 300) % 230 - 115) < 14) * fl
            img += (beam * 0.75)[..., None] * np.array([0.95, 0.95, 0.9])
            img += (beam * fl * (1 - 0.7 * stripes) * 0.9)[..., None] * np.array([1.0, 0.98, 0.92])

        # hologram + man in depth order (sofa pose is behind him; others around him)
        her_light = np.zeros((RH, RW, 1), np.float32)
        man_out, mj = man_shapes(t, *self.man_motion(t))
        poses = list(pose_at(t))
        holo = [(self.her_pose(name, lt, t, mj), g) for name, g, lt in poses]
        for (shapes, extra, pos, Dh), g in holo:
            if Dh > MAN["D"]:
                img, her_light = self.hologram(img, shapes, extra, g, t, rng, raster, pos, her_light)
        img = self.draw_man(img, man_out, mj, raster, her_light, beam)
        for (shapes, extra, pos, Dh), g in holo:
            if MAN["D"] - 0.2 < Dh <= MAN["D"]:
                img, her_light = self.hologram(img, shapes, extra, g, t, rng, raster, pos, her_light)

        # polished floor: mirror the window and whoever stands at it
        nref = RH - wb
        rows = np.arange(nref)
        src_rows = np.clip(wb - 1 - (rows * 1.6).astype(np.int32), 0, RH - 1)
        refl = blur(img[src_rows], 3.0)
        fade = (np.linspace(0.32, 0.05, nref)[:, None, None]) * fl[wb:, :, None]
        img[wb:] = img[wb:] + refl * fade

        # near furniture (dining zone), then whoever is at the table
        a = self.near[..., 3:4]
        img = img * (1 - a) + self.near[..., :3] * a
        for (shapes, extra, pos, Dh), g in holo:
            if Dh <= MAN["D"] - 0.2:
                img, her_light = self.hologram(img, shapes, extra, g, t, rng, raster, pos, her_light)
        img += self.emis

        # camera crop
        hw, hh = W / (2 * z), H / (2 * z)
        out = to_u8(img).resize((W, H), Image.BICUBIC, box=(ccx - hw, ccy - hh, ccx + hw, ccy + hh))
        return to_f(out)

    def frame(self, i):
        t = i / FPS
        rng = np.random.default_rng(5000 + i)
        if t < T_CUT:
            img = self.exterior(t, i, rng)
        elif t >= T_IN:
            img = self.interior(t, i, rng)
        else:
            a = smooth(T_CUT, T_IN, t)
            img = self.exterior(t, i, rng) * (1 - a) + self.interior(t, i, rng) * a
            img += a * (1 - a) * 0.45  # the beam flares through the cut
        return self.finish(img, t, i)

    def finish(self, img, t, i):
        lum = img.mean(axis=2, keepdims=True)
        bright = np.clip(img * np.clip((lum - 0.42) * 2.5, 0, 1), 0, 2)
        small = to_u8(bright).resize((W // 4, H // 4), Image.BILINEAR)
        b1 = to_f(small.filter(ImageFilter.GaussianBlur(3)).resize((W, H), Image.BILINEAR))
        b2 = to_f(small.filter(ImageFilter.GaussianBlur(12)).resize((W, H), Image.BILINEAR))
        img = img + b1 * 0.65 + b2 * 0.85
        lum = img.mean(axis=2, keepdims=True)
        warm = smooth(T_CUT, T_IN, t)  # exterior keeps the teal/amber of the city; inside goes K-apartment amber
        img = img + (1 - np.clip(lum * 3, 0, 1)) * np.array([-0.008, 0.008, 0.02]) + np.clip(lum - 0.4, 0, 1) * np.array([0.06, 0.02, -0.04]) * (1 + warm * 0.5)
        img = lum + (img - lum) * (1 - 0.1 * warm)
        img = 1 - np.exp(-img * 1.65)
        img[..., 0] = np.roll(img[..., 0], 1, axis=1)
        img[..., 2] = np.roll(img[..., 2], -1, axis=1)
        img = img * self.vignette + self.grain[i % 8][..., None] * 0.022
        img *= smooth(0, 1.5, t) * (1 - smooth(DUR - 2.0, DUR, t))
        out = to_u8(img)
        dr = ImageDraw.Draw(out)
        dr.rectangle([0, 0, W, BAR], fill=(0, 0, 0))
        dr.rectangle([0, H - BAR, W, H], fill=(0, 0, 0))
        return out.tobytes()


_scene = None


def _init():
    global _scene
    _scene = Scene()


def _render(i):
    return _scene.frame(i)


# ----------------------------------------------------------------------------
# Sound
# ----------------------------------------------------------------------------

def synth_audio(path, t0, t1):
    sr = 44100
    n = int(sr * DUR)
    t = np.arange(n) / sr
    rng = np.random.default_rng(SEED)
    inten = np.interp(t, t[::441], np.array([rain_intensity(x) for x in t[::441]]))
    inside = np.clip((t - T_CUT) / (T_IN - T_CUT), 0, 1)
    noise = rng.normal(0, 1, n)
    rain_out = (onepole(noise, 5000, sr) - onepole(noise, 400, sr)) * 0.16
    rain_in = onepole(noise, 900, sr) * 0.22
    ticks = np.zeros(n)
    idx = rng.integers(0, n - 400, 26000)
    keep = rng.random(len(idx)) < inten[idx]
    ticks[idx[keep]] = rng.uniform(-1, 1, keep.sum())
    ticks = np.convolve(ticks, np.exp(-np.arange(200) / 18), mode="same")
    patter = (onepole(ticks, 7000, sr) - onepole(ticks, 1500, sr)) * 0.35
    rain = (rain_out * (1 - inside) + (rain_in + patter * inside) * inside + patter * 0.3 * (1 - inside)) * (0.45 + 0.55 * inten)

    def saw(freq, det):
        s = np.zeros(n)
        for h in range(1, 12):
            s += np.sin(2 * np.pi * freq * (1 + det) * h * t + h) / h
        return s

    pad = np.zeros(n)
    for f in (65.41, 98.0, 155.56, 196.0, 233.08):
        for d in (-0.004, 0.0, 0.005):
            pad += saw(f, d)
    pad = onepole(pad, 700, sr) * 0.03 * (0.4 + 0.6 * np.clip((t - 2) / 10, 0, 1))
    # spinner: turbine hum with a doppler sweep as it banks, then hovering; fades as the light dies
    pitch = 150 + 40 * np.sin(np.clip(t, 0, 7.6) * 0.8) - 30 * np.clip((t - 3.4) / 2.6, 0, 1)
    phase = np.cumsum(2 * np.pi * pitch / sr)
    turb = (np.sin(phase) * 0.5 + np.sin(phase * 2.01) * 0.25 + (onepole(noise, 1800, sr) - onepole(noise, 300, sr)) * 0.8)
    env = np.clip(t / 1.5, 0, 1) * (0.4 + 0.6 * np.clip((t - 3) / 4, 0, 1)) * (1 - np.clip((t - 11.4) / 3.2, 0, 1))
    spin = turb * env * (0.12 - 0.07 * inside)
    lock = np.zeros(n)
    m = t >= 8.0
    tt = t[m] - 8.0
    lock[m] = np.sin(2 * np.pi * 880 * tt) * np.exp(-tt / 0.25) * 0.05
    gl = np.zeros(n)
    for a, b, _ in POSES:
        for st in (a - GLITCH, b):
            k0 = int(max(0, st) * sr)
            ln = int(GLITCH * sr)
            seg = np.sign(np.sin(2 * np.pi * rng.uniform(300, 2400) * np.arange(ln) / sr * (1 + 3 * rng.random(ln))))
            seg *= rng.random(ln) > 0.4
            k1 = min(n, k0 + ln)
            gl[k0:k1] += (seg * 0.05 * np.hanning(ln))[: k1 - k0]
    bells = np.zeros(n)
    for st, f in ((16.4, 587.33), (18.9, 698.46), (22.2, 523.25), (28.3, 783.99), (34.2, 587.33), (36.6, 659.25)):
        m = t >= st
        tt = t[m] - st
        bells[m] += (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt)) * np.exp(-tt / 2.4) * 0.05
    mix = rain + pad + spin + lock + gl + bells
    st = np.stack([mix + rain[::-1] * 0.08, mix * 0.95 + np.roll(rain, 2000) * 0.08], 1)
    st *= (np.clip(t / 1.5, 0, 1) * np.clip((DUR - t) / 2, 0, 1))[:, None]
    st /= np.abs(st).max() / 0.9
    seg = st[int(t0 * sr):int(t1 * sr)].copy()
    if t0 > 0:
        seg[: int(0.4 * sr)] *= np.linspace(0, 1, int(0.4 * sr))[:, None]
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((seg * 32767).astype(np.int16).tobytes())


def main():
    out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "apartment.mp4"
    if "--still" in sys.argv:
        ts = [float(x) for x in sys.argv[sys.argv.index("--still") + 1].split(",")]
        s = Scene()
        for tt in ts:
            path = out if len(ts) == 1 else out.replace(".png", f"_{tt}.png")
            Image.frombytes("RGB", (W, H), s.frame(int(round(tt * FPS)))).save(path)
        return
    t0, t1 = 0.0, DUR
    if "--range" in sys.argv:
        k = sys.argv.index("--range")
        t0, t1 = float(sys.argv[k + 1]), float(sys.argv[k + 2])
    wav = os.path.splitext(out)[0] + ".wav"
    synth_audio(wav, t0, t1)
    frames = range(int(round(t0 * FPS)), int(round(t1 * FPS)))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "slow", "-crf", "22",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(os.cpu_count(), initializer=_init) as pool:
        for k, buf in enumerate(pool.imap(_render, frames, chunksize=2)):
            p.stdin.write(buf)
            if k % 48 == 0:
                print(f"frame {k}/{len(frames)}", flush=True)
    p.stdin.close()
    p.wait()
    os.remove(wav)
    print("wrote", out)


if __name__ == "__main__":
    main()
