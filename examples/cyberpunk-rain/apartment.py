"""Rain on an apartment window: a man stares out at the rain while the
hologram of a woman flickers through the room behind him. She touches him,
then glitches away to the sofa with a book, then to the dining table with a
glass of red wine. We watch it all from outside, through floor-to-ceiling
glass with rain running down it.

Same 2D style and toolkit as render.py (numpy + Pillow + ffmpeg). Colour
script after K's apartment in Blade Runner 2049: dim amber lamplight on raw
concrete, cool grey-blue rain light from the window, a cold violet hologram,
and one saturated red: the wine.

Usage:
  python3 apartment.py out.mp4 [--range 10.5 18.6]
  python3 apartment.py still.png --still 14.2
"""

import math
import os
import subprocess
import sys
import wave
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from render import BAR, FPS, H, W, building_layer, ik, onepole, rot, smooth

DUR = 36.0
SEED = 2049
RW, RH = 1440, 810  # room canvas; the camera crops it
FLOOR = 600  # wall base line in the room canvas


def lerp(a, b, u):
    return a + (b - a) * u


def lerp2(a, b, u):
    return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)


def fbm(shape, seed, scales=(4, 8, 16, 32, 64), gains=(1, 0.5, 0.25, 0.13, 0.07)):
    """Smooth value noise built from upsampled random grids (static textures)."""
    rng = np.random.default_rng(seed)
    hgt, wid = shape
    out = np.zeros(shape, np.float32)
    for s, g in zip(scales, gains):
        small = rng.random((max(2, hgt * s // wid + 2), s + 2)).astype(np.float32)
        img = Image.fromarray((small * 255).astype(np.uint8)).resize((wid, hgt), Image.BICUBIC)
        out += np.asarray(img, np.float32) / 255.0 * g
    return out / sum(gains)


def rain_intensity(t):
    return 0.35 + 0.65 * smooth(9.0, 30.0, t)


# ----------------------------------------------------------------------------
# Rain on the glass: beads, sliding rivulets that swallow beads, wet trails
# ----------------------------------------------------------------------------

class GlassRain:
    """Deterministic simulation stepped once for the whole film; snapshots per frame."""

    X0, X1, Y0, Y1 = -120, RW + 120, -120, RH + 60

    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        n0 = 1600
        self.bx = rng.uniform(self.X0, self.X1, n0)
        self.by = rng.uniform(self.Y0, self.Y1, n0)
        self.br = 0.7 + 3.6 * rng.random(n0) ** 2.2
        self.riv = []  # dicts: x, y, r, v, slide, timer, ph, trail[list of (x, y, t)]
        self.snaps = []
        dt = 1.0 / FPS
        t = 0.0
        for i in range(int(DUR * FPS) + 1):
            self.snaps.append(self.snapshot(t))
            for _ in range(2):
                self.step(t, dt / 2, rng)
                t += dt / 2

    def snapshot(self, t):
        rv = [(d["x"], d["y"], d["r"], d["slide"]) for d in self.riv]
        tr = [np.array([(x, y, t - tt) for x, y, tt in d["trail"]], np.float32) for d in self.riv]
        tr += [np.array(s, np.float32) for s in getattr(self, "dead_trails", [])]
        return (np.stack([self.bx, self.by, self.br], 1).astype(np.float32), rv, [a for a in tr if len(a) > 1])

    def step(self, t, dt, rng):
        inten = rain_intensity(t)
        # new beads from drops hitting the glass
        n = rng.poisson(420 * inten * dt)
        if n:
            self.bx = np.concatenate([self.bx, rng.uniform(self.X0, self.X1, n)])
            self.by = np.concatenate([self.by, rng.uniform(self.Y0, self.Y1, n)])
            self.br = np.concatenate([self.br, 0.7 + 3.8 * rng.random(n) ** 2.2])
        if len(self.bx) > 3200:
            k = len(self.bx) - 3200
            self.bx, self.by, self.br = self.bx[k:], self.by[k:], self.br[k:]
        # heavy beads let go and start running
        if rng.random() < 4.0 * inten * dt:
            big = np.where(self.br > 3.0)[0]
            if len(big):
                j = int(rng.choice(big))
                self.riv.append(dict(x=float(self.bx[j]), y=float(self.by[j]), r=float(self.br[j]) + 1.0, v=0.0, slide=True,
                                     timer=rng.uniform(0.2, 0.9), ph=rng.uniform(0, 6.28), trail=[]))
                self.br[j] = 0
        if rng.random() < 2.2 * inten * dt:
            self.riv.append(dict(x=rng.uniform(self.X0, self.X1), y=self.Y0 + rng.uniform(0, 80), r=rng.uniform(3.0, 4.5),
                                 v=0.0, slide=True, timer=rng.uniform(0.3, 1.2), ph=rng.uniform(0, 6.28), trail=[]))
        dead = []
        self.dead_trails = []
        for d in self.riv:
            d["timer"] -= dt
            if d["timer"] < 0:  # stick-slip: water snags on the glass, then lets go
                d["slide"] = not d["slide"]
                d["timer"] = rng.uniform(0.25, 1.1) if d["slide"] else rng.uniform(0.05, 0.6) / (d["r"] / 3)
            target = (25 + 38 * d["r"]) if d["slide"] else 0.0
            d["v"] += (target - d["v"]) * min(1.0, dt * 9)
            d["y"] += d["v"] * dt
            d["x"] += math.sin(d["ph"] + d["y"] * 0.03) * 0.35 * (d["v"] / 120) + rng.normal(0, 0.15)
            # swallow beads in the path
            near = np.where((np.abs(self.bx - d["x"]) < d["r"] + 3) & (np.abs(self.by - d["y"]) < d["r"] + 3) & (self.br > 0))[0]
            if len(near):
                d["r"] = min(8.5, math.sqrt(d["r"] ** 2 + 0.35 * float(np.sum(self.br[near] ** 2))))
                self.br[near] = 0
            # it sheds water as it goes: thinner, and leaves tiny beads behind
            d["r"] = max(1.6, d["r"] - 0.04 * dt * d["v"] / 10)
            if d["v"] > 10 and rng.random() < 0.08:
                self.bx = np.append(self.bx, d["x"] + rng.normal(0, 1))
                self.by = np.append(self.by, d["y"] - d["r"] * 2)
                self.br = np.append(self.br, rng.uniform(0.6, 1.3))
            if not d["trail"] or abs(d["trail"][-1][1] - d["y"]) > 3:
                d["trail"].append((d["x"], d["y"], t))
            d["trail"] = [p for p in d["trail"] if t - p[2] < 3.0]
            if d["y"] > self.Y1:
                dead.append(d)
        for d in dead:
            self.riv.remove(d)
        keep = self.br > 0
        self.bx, self.by, self.br = self.bx[keep], self.by[keep], self.br[keep]


# ----------------------------------------------------------------------------
# The room (static plate in room coordinates, lit in float)
# ----------------------------------------------------------------------------

LAMP = (172, 318)  # floor lamp shade
PENDANT = (1150, 382)


def glow(xx, yy, cx, cy, rx, ry, p=2.0):
    return np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) ** (p / 2))


def build_room():
    yy, xx = np.mgrid[0:RH, 0:RW].astype(np.float32)
    rng = np.random.default_rng(11)
    noise = fbm((RH, RW), 3)
    fine = np.asarray(Image.fromarray((rng.random((RH, RW)) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255
    img = np.zeros((RH, RW, 3), np.float32)

    # raw concrete wall: formwork lines, tie holes, stains
    wall = (yy >= 140) & (yy < FLOOR)
    base = np.array([0.062, 0.05, 0.043])
    conc = 0.75 + 0.35 * noise + 0.12 * (fine - 0.5)
    conc -= 0.18 * ((yy - 140) % 92 < 2.2) * wall
    conc -= 0.12 * ((xx + 40) % 240 < 2.0) * wall
    ties = ((((xx + 40) % 240) - 120) ** 2 + (((yy - 140) % 92) - 46) ** 2) < 16
    conc -= 0.35 * ties
    streak = fbm((RH, RW // 8), 9, scales=(16, 32, 64), gains=(1, 0.6, 0.3))
    streak = np.asarray(Image.fromarray((streak * 255).astype(np.uint8)).resize((RW, RH), Image.BICUBIC), np.float32) / 255
    conc -= 0.15 * np.clip(streak - 0.55, 0, 1) * 3 * (yy / RH)
    img[wall] = (base * conc[..., None])[wall]
    # ceiling with a recessed, dead light slot
    ceil = yy < 140
    img[ceil] = (np.array([0.05, 0.042, 0.036]) * (0.8 + 0.3 * noise[..., None]))[ceil]
    img[(yy > 118) & (yy < 124) & (xx > 300) & (xx < 1200)] = (0.02, 0.018, 0.016)
    # polished concrete floor
    flo = yy >= FLOOR
    img[flo] = (np.array([0.06, 0.05, 0.045]) * (0.8 + 0.4 * noise[..., None]))[flo]

    # warm lamplight pools
    lamp = glow(xx, yy, LAMP[0], LAMP[1] + 30, 330, 260, 1.6)[..., None]
    img += lamp * np.array([0.5, 0.27, 0.1]) * (wall | ceil)[..., None] * 0.95
    pend = glow(xx, yy, PENDANT[0], PENDANT[1] + 120, 220, 200, 1.6)[..., None]
    img += pend * np.array([0.42, 0.22, 0.08]) * wall[..., None] * 0.75
    # doorway to a dim corridor
    d = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    dr = ImageDraw.Draw(d)
    dr.rectangle([760, 330, 860, FLOOR], fill=(10, 9, 9))
    dr.rectangle([848, 340, 852, FLOOR], fill=(40, 34, 30))
    # wall screen (off), glossy
    dr.rectangle([290, 240, 530, 375], fill=(9, 9, 11), outline=(30, 26, 24))
    # sofa: low, deep, worn leather
    dr.rounded_rectangle([230, 485, 562, 545], 10, fill=(24, 18, 15))  # back
    dr.rounded_rectangle([238, 535, 556, 588], 6, fill=(30, 22, 18))  # seat
    dr.rounded_rectangle([222, 500, 262, 592], 8, fill=(22, 16, 13))  # arms
    dr.rounded_rectangle([530, 500, 570, 592], 8, fill=(22, 16, 13))
    for x in (240, 552):
        dr.rectangle([x, 588, x + 6, FLOOR], fill=(12, 10, 9))
    dr.line([(240, 537), (550, 537)], fill=(60, 44, 34), width=2)  # lit edge of the cushions
    dr.line([(398, 538), (398, 586)], fill=(18, 13, 11), width=2)
    dr.line([(232, 487), (560, 487)], fill=(58, 42, 32), width=2)
    # side table with a stack of books
    dr.rectangle([140, 520, 205, 526], fill=(28, 22, 18))
    dr.rectangle([169, 526, 175, FLOOR], fill=(20, 16, 13))
    for k, c in enumerate([(70, 30, 26), (40, 46, 52), (66, 58, 40)]):
        dr.rectangle([147 + k * 2, 508 - k * 6, 195 - k * 3, 514 - k * 6], fill=c)
    # floor lamp
    dr.rectangle([LAMP[0] - 2, 340, LAMP[0] + 2, FLOOR - 4], fill=(30, 26, 22))
    dr.ellipse([LAMP[0] - 21, FLOOR - 8, LAMP[0] + 21, FLOOR], fill=(24, 20, 17))
    # dining table, chairs, bottle, pendant cord
    dr.rectangle([1000, 541, 1300, 549], fill=(38, 28, 20))
    dr.line([(1000, 541), (1300, 541)], fill=(110, 70, 40), width=2)
    for x in (1012, 1282):
        dr.rectangle([x, 549, x + 7, FLOOR], fill=(22, 17, 13))
    for x, flip in ((968, 1), (1322, -1)):
        dr.rectangle([x - 4, 470, x + 4, FLOOR], fill=(18, 14, 12))
        dr.rectangle([x - 4 if flip > 0 else x - 30, 548, x + 30 if flip > 0 else x + 4, 556], fill=(20, 16, 13))
    dr.rectangle([1068, 495, 1080, 541], fill=(12, 20, 14))
    dr.rectangle([1071, 480, 1077, 496], fill=(12, 20, 14))
    dr.line([(1078, 498), (1078, 538)], fill=(70, 100, 80), width=1)
    dr.line([(PENDANT[0], 140), (PENDANT[0], PENDANT[1] - 10)], fill=(20, 17, 15), width=2)
    d = np.asarray(d, np.float32) / 255.0

    # floor reflections of the lamps (streaked, wet-looking polish)
    refl = glow(xx, yy, LAMP[0], FLOOR + 40, 40, 120, 1.5) * 0.25 + glow(xx, yy, PENDANT[0], FLOOR + 30, 60, 90, 1.5) * 0.2
    d += (refl * flo)[..., None] * np.array([1.0, 0.6, 0.3])
    # glowing shades (bloom will make them sing)
    emis = np.zeros_like(d)
    e = Image.new("RGB", (RW, RH))
    ed = ImageDraw.Draw(e)
    ed.polygon([(LAMP[0] - 24, 300), (LAMP[0] + 24, 300), (LAMP[0] + 32, 340), (LAMP[0] - 32, 340)], fill=(255, 190, 120))
    ed.polygon([(PENDANT[0] - 10, PENDANT[1] - 12), (PENDANT[0] + 10, PENDANT[1] - 12), (PENDANT[0] + 34, PENDANT[1] + 14),
                (PENDANT[0] - 34, PENDANT[1] + 14)], fill=(255, 175, 100))
    emis = np.asarray(e.filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255.0 * 1.4
    # light cone under the pendant (dust in the air)
    cone = np.clip(1 - np.abs(xx - PENDANT[0]) / (40 + (yy - PENDANT[1]) * 0.45), 0, 1) * ((yy > PENDANT[1]) & (yy < 545))
    d += (cone ** 2)[..., None] * np.array([0.08, 0.05, 0.025])
    return d + emis


# ----------------------------------------------------------------------------
# The man at the window (front view): a few material regions, idle motion
# ----------------------------------------------------------------------------

MAN = dict(x=905.0, y=742.0, u=1.86)  # feet position (room coords) and px per unit


def man_shapes(t, tilt, breathe, sway, hand_lift):
    """Returns dict of material -> shapes in room coords, plus key joints."""
    fx, fy, u = MAN["x"], MAN["y"], MAN["u"]
    P = lambda x, y: (fx + x * u, fy + y * u)
    hip = P(sway * 0.6, -112)
    chest_up = breathe * 1.2
    sh_l = P(-41 + sway * 0.2, -202 - chest_up)
    sh_r = P(41 + sway * 0.2, -202 - chest_up * 1.1)
    neck = P(sway * 0.15, -226 - chest_up)
    head_c = rot([P(sway * 0.1, -249 - chest_up)], neck, tilt)[0]
    out = {"pants": [], "sweater": [], "skin": [], "hair": [], "shoe": []}
    # legs: weight on his right leg, left knee eased
    for side, bend in ((-1, 3.0), (1, 0.0)):
        h0 = (hip[0] + side * 12 * u, hip[1])
        ank = P(side * 15, -6)
        knee = (lerp(h0[0], ank[0], 0.5) - side * bend * u * 0.6, lerp(h0[1], ank[1], 0.5))
        out["pants"].append(("line", [h0, knee, ank], 21 * u))
        out["shoe"].append(("ell", (ank[0] + side * 3 * u, ank[1] + 2 * u, 9 * u, 5 * u)))
    # torso: knit sweater, slight waist, broad shoulders
    tor = [P(-26 + sway * 0.6, -106), P(26 + sway * 0.6, -106), P(25 + sway * 0.4, -142), P(31, -182 - chest_up),
           (sh_r[0] + 2 * u, sh_r[1] + 8 * u), (sh_r[0] - 4 * u, sh_r[1] + 1 * u), P(13, -217 - chest_up),
           P(-13, -217 - chest_up), (sh_l[0] + 4 * u, sh_l[1] + 1 * u), (sh_l[0] - 2 * u, sh_l[1] + 8 * u),
           P(-31, -182 - chest_up), P(-25 + sway * 0.4, -142)]
    out["sweater"].append(("poly", tor))
    out["sweater"].append(("ell", (sh_l[0] + 3 * u, sh_l[1] + 6 * u, 7.5 * u, 7.5 * u)))
    out["sweater"].append(("ell", (sh_r[0] - 3 * u, sh_r[1] + 6 * u, 7.5 * u, 7.5 * u)))
    # high collar
    out["sweater"].append(("poly", [P(-12, -214 - chest_up), P(12, -214 - chest_up), P(10, -229 - chest_up), P(-10, -229 - chest_up)]))
    # left arm hangs; right hand in his pocket ... until he raises it to the glass
    # relaxed hanging arm: elbow soft, hand turned in, a slow pendulum with the sway
    swing = 0.04 * math.sin(t * 2 * math.pi / 9.0 + 0.8)
    el_l = (sh_l[0] - 7 * u + swing * 40, sh_l[1] + 49 * u)
    hd_l = (el_l[0] + 4 * u + swing * 60, el_l[1] + 44 * u)
    out["sweater"].append(("line", [sh_l, el_l], 13.5 * u))
    out["sweater"].append(("line", [el_l, (lerp(el_l[0], hd_l[0], 0.85), lerp(el_l[1], hd_l[1], 0.85))], 11.5 * u))
    out["skin"].append(("poly", [(hd_l[0] - 4 * u, hd_l[1] - 3 * u), (hd_l[0] + 4 * u, hd_l[1] - 3 * u), (hd_l[0] + 5 * u, hd_l[1] + 6 * u),
                                 (hd_l[0] + 2 * u, hd_l[1] + 14 * u), (hd_l[0] - 3 * u, hd_l[1] + 13 * u), (hd_l[0] - 5 * u, hd_l[1] + 5 * u)]))
    pocket = P(30 + sway * 0.5, -108)
    glass_pt = P(36, -168)
    hand_r = lerp2(pocket, glass_pt, hand_lift)
    el_r = ik(sh_r, hand_r, 50 * u, 47 * u)
    if hand_lift < 0.5:
        el_r = (max(el_r[0], sh_r[0] + 6 * u), el_r[1])
    out["sweater"].append(("line", [sh_r, el_r, hand_r], 13 * u))
    if hand_lift > 0.05:
        out["skin"].append(("ell", (hand_r[0], hand_r[1] - 4 * u * hand_lift, (6 + 3 * hand_lift) * u, (9 + 2 * hand_lift) * u)))
    # neck, head, ears, hair
    out["skin"].append(("line", [P(sway * 0.15, -224 - chest_up), (head_c[0], head_c[1] + 14 * u)], 13 * u))
    out["skin"].append(("ell", (head_c[0], head_c[1], 15 * u, 19 * u)))
    for side in (-1, 1):
        ex, ey = rot([(head_c[0] + side * 15 * u, head_c[1] + 1 * u)], head_c, tilt)[0]
        out["skin"].append(("ell", (ex, ey, 2.2 * u, 4 * u)))
    hair = [(-16.5, 2), (-16.5, -8), (-14, -16), (-8, -21.5), (0, -23.5), (8, -22), (14.5, -17), (16.5, -8), (16.5, 2),
            (14.5, -4), (12, -11), (6, -13.5), (-2, -14), (-9, -12.5), (-14, -6)]
    out["hair"].append(("poly", rot([(head_c[0] + x * u, head_c[1] + y * u) for x, y in hair], head_c, tilt)))
    joints = dict(head=head_c, sh_l=sh_l, sh_r=sh_r, hand_r=hand_r, el_l=el_l, cheek_l=rot([(head_c[0] - 11 * u, head_c[1] + 6 * u)], head_c, tilt)[0])
    return out, joints


# ----------------------------------------------------------------------------
# The hologram: profile figure, three poses, glitch teleports
# ----------------------------------------------------------------------------

def head_profile(c, s, look, hair_sw):
    """Face in profile (facing +x) around head centre c; look rotates it."""
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


def her_standing(fx, fy, s, t, reach=None, look=0.0, lean=0.0, hold_glass=None):
    """Standing in profile facing +x. reach: hand target for the near arm."""
    br = math.sin(t * 1.6) * 0.8
    pivot = (fx, fy - 112 * s)
    P = lambda pts: rot([(fx + x * s, fy + y * s) for x, y in pts], pivot, lean)
    sw = 2.0 * math.sin(t * 0.9)  # dress settles / sways
    shapes = []
    # legs, weight on the back leg, front knee soft
    shapes.append(("poly", P([(-6, -60), (1, -60), (-1, -4), (-6, -4)])))
    shapes.append(("poly", P([(3, -60), (10, -60), (11, -4), (6, -4)])))
    for x0 in (-7, 5):
        shapes.append(("poly", P([(x0, -5), (x0 + 11, -3), (x0 + 12, 0), (x0, 0)])))
    # far arm, hanging
    sh_far = P([(-4, -199 - br)])[0]
    shapes.append(("line", [sh_far, P([(-6 + sw * 0.2, -150)])[0], P([(-3 + sw * 0.3, -108)])[0]], 6.5 * s))
    # body + dress
    shapes.append(("poly", P([(-6, -206 - br), (-10, -199 - br), (-12, -170), (-8, -142), (-13, -112), (-22 + sw, -44),
                              (-4 + sw * 0.6, -40), (8 + sw * 0.6, -41), (24 + sw, -46), (12, -112), (9, -142), (16, -170 - br),
                              (14, -186 - br), (8, -200 - br), (5, -206 - br)])))
    shapes.append(("poly", P([(-5, -214 - br), (5, -214 - br), (5, -203 - br), (-6, -203 - br)])))
    head_c = P([(1, -236 - br)])[0]
    shapes += head_profile(head_c, s, look + lean * 0.5, 1.2 * math.sin(t * 1.3))
    # near arm
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
    """Sitting sideways on the sofa, back to the armrest, knees up, reading."""
    br = math.sin(t * 1.5) * 0.6
    lean = -0.28
    P = lambda pts: rot([(hx + x * s, hy + y * s) for x, y in pts], (hx, hy), lean)
    Q = lambda pts: [(hx + x * s, hy + y * s) for x, y in pts]  # legs stay put
    shapes = []
    # thighs up to the knees, shins down to the cushion
    knee = Q([(46, -40)])[0]
    for dx in (0, 4):
        shapes.append(("line", [Q([(4 + dx, -8)])[0], (knee[0] + dx * s, knee[1]), Q([(72 + dx, -2)])[0]], 10 * s))
    shapes.append(("poly", Q([(-14, 0), (-10, -22), (10, -26), (44, -48), (50, -40), (20, -10), (12, 0)])))  # dress over lap
    # torso leaning back on the armrest
    shapes.append(("poly", P([(-10, -6), (-12, -40), (-11, -70), (-8, -94 - br), (6, -94 - br), (12, -84), (14, -62), (8, -38), (10, -6)])))
    shapes.append(("poly", P([(-4, -100 - br), (5, -100 - br), (5, -92), (-5, -92)])))
    head_c = P([(1, -122 - br)])[0]
    shapes += head_profile(head_c, s, 0.42 - 0.35 * look_up, 0.8 * math.sin(t))
    # arms bring the book up on the knees
    hand = (knee[0] - 4 * s, knee[1] - 20 * s)
    sh = P([(0, -88)])[0]
    el = ik(sh, hand, 46 * s, 42 * s)
    shapes.append(("line", [sh, el, hand], 6 * s))
    # the book: open, seen edge-on; a page turning over the spine
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


# hologram poses on the timeline: (start, end, pose). Gaps are glitch teleports.
POSES = [
    (11.45, 17.75, "beside"),
    (18.05, 23.35, "sofa"),
    (23.65, 29.35, "table"),
    (29.65, 36.0, "beside2"),
]
GLITCH = 0.32  # seconds to tear apart / reassemble


def pose_at(t):
    for a, b, name in POSES:
        if a - GLITCH <= t <= b + GLITCH:
            g_in = 1 - smooth(a - GLITCH, a, t) if t < a else 0.0
            g_out = smooth(b, b + GLITCH, t) if t > b else 0.0
            yield name, max(g_in, g_out), (t - a)


# ----------------------------------------------------------------------------
# Camera (drone hovering outside the glass) and focus
# ----------------------------------------------------------------------------

def camera(t):
    """zoom, centre in room coords, focus (0 = on the glass, 1 = in the room)."""
    if t < 12.0:
        u = smooth(9.0, 12.5, t)
        z = 1.42 - 0.38 * u
        cx, cy = lerp(840, 740, u), lerp(300, 395, u)
    elif t < 17.4:
        u = smooth(12.0, 17.4, t)
        z = 1.04 - 0.04 * u
        cx, cy = lerp(740, 715, u), 395
    elif t < 23.2:
        u = smooth(17.4, 19.5, t)
        z = 1.0
        cx, cy = lerp(715, 650, u), 400
    elif t < 29.2:
        u = smooth(23.2, 25.0, t)
        z = 1.0
        cx, cy = lerp(650, 790, u), 400
    else:
        u = smooth(29.2, 32.0, t)
        z = 1.0 + 0.18 * u
        cx, cy = lerp(790, 820, u), lerp(400, 360, u)
    # hovering drone: slow drift plus a little buffeting in the wind
    cx += 3.0 * math.sin(t * 0.7) + 1.2 * math.sin(t * 2.3 + 1)
    cy += 2.0 * math.sin(t * 0.9 + 2) + 0.9 * math.sin(t * 3.1)
    focus = smooth(10.6, 12.2, t)
    return z, cx, cy, focus


# ----------------------------------------------------------------------------
# Renderer
# ----------------------------------------------------------------------------

HOLO_A = np.array([0.42, 0.36, 1.0])
HOLO_B = np.array([0.35, 0.85, 1.0])


class Scene:
    def __init__(self):
        rng = np.random.default_rng(SEED)
        self.room = build_room()
        self.glass = GlassRain()
        # reflection of the city behind the camera on the glass (neon smears)
        lay = building_layer(rng, 60, 300, 700, (0.02, 0.02, 0.03), 0.2, 0.05, 0.9, RW + 400)
        refl = Image.fromarray((np.clip(lay[0], 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(5))
        self.reflect = np.asarray(refl, np.float32) / 255.0
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.xx, self.yy = xx, yy
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        self.vignette = np.clip(1.12 - 0.5 * r ** 1.8, 0.25, 1)[..., None].astype(np.float32)
        self.grain = rng.normal(0, 1, (8, H, W)).astype(np.float32)
        self.knit = fbm((H, W), 21, scales=(64, 128, 256), gains=(1, 0.6, 0.4))
        self.outside_rain = np.stack([rng.uniform(0, 1, 260), rng.uniform(0, 1, 260), rng.uniform(0.3, 1, 260)], 1)

    def frame(self, i):
        t = i / FPS
        rng = np.random.default_rng(5000 + i)
        z, ccx, ccy, focus = camera(t)
        inten = rain_intensity(t)
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
                else:
                    pts = [(x * S, y * S) for x, y in map(to_s, sh[1])]
                    wdt = max(1, int(sh[2] * z * S))
                    d.line(pts, fill=255, width=wdt, joint="curve")
                    for x, y in (pts[0], pts[-1]):
                        d.ellipse([x - wdt / 2, y - wdt / 2, x + wdt / 2, y + wdt / 2], fill=255)
            return np.asarray(m.resize((W, H), Image.LANCZOS), np.float32) / 255.0

        # ---- the room plate through the camera --------------------------------
        hw, hh = W / (2 * z), H / (2 * z)
        box = (ccx - hw, ccy - hh, ccx + hw, ccy + hh)
        room = Image.fromarray((np.clip(self.room, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC, box=box)
        img = np.asarray(room, np.float32) / 255.0

        # rain light from the window falls into the room, patterned by the
        # water running down the glass (the 2049 rain-shadow look)
        snap = self.glass.snaps[min(i, len(self.glass.snaps) - 1)]
        shadow = self.rain_shadow(snap, ccx, ccy, z, t)
        neon = np.array([0.32, 0.4, 0.55]) + 0.12 * math.sin(t * 0.37) * np.array([0.6, -0.2, 0.2])
        window_light = (0.35 + 0.65 * np.clip((self.yy - 60) / 600, 0, 1))[..., None] * neon * 0.24
        img += window_light * (1 - 0.75 * shadow[..., None])

        # ---- the hologram (behind him, in the room) ---------------------------
        her_light = np.zeros((H, W, 1), np.float32)
        man_out, mj = man_shapes(t, *self.man_motion(t))
        for name, g, lt in pose_at(t):
            shapes, extra, pos = self.her_pose(name, lt, t, mj)
            img, her_light = self.hologram(img, shapes, extra, g, t, rng, raster, to_s, pos, z, her_light, name, lt)

        # ---- him, at the glass ---------------------------------------------
        img = self.draw_man(img, man_out, mj, raster, to_s, shadow, neon, her_light, t, z)

        # out of focus room while we look at the glass
        if focus < 0.98:
            r = 5.5 * (1 - focus)
            img = np.asarray(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r)), np.float32) / 255

        # ---- the glass: reflections, beads, rivulets, trails ----------------
        img = self.glass_layer(img, snap, ccx, ccy, z, focus, t, rng)

        # rain falling between the drone and the glass
        rl = Image.new("L", (W, H), 0)
        rd = ImageDraw.Draw(rl)
        for x, y, dz in self.outside_rain:
            if dz > 0.35 + 0.65 * inten:
                continue
            yy = (y + t * (1.6 + dz * 1.2)) % 1.0
            px, py = (x - 0.05 * yy) * W * 1.1, yy * (H + 120) - 60
            L = 30 + 60 * dz
            rd.line([(px, py), (px - 0.08 * L, py + L)], fill=int(30 + 60 * dz), width=1 + int(dz > 0.8))
        img += (np.asarray(rl.filter(ImageFilter.GaussianBlur(1.6)), np.float32) / 255.0)[..., None] * np.array([0.5, 0.56, 0.65]) * 0.6

        return self.finish(img, t, i)

    # ------------------------------------------------------------------------
    def man_motion(self, t):
        breathe = 0.5 + 0.5 * math.sin(t * 2 * math.pi / 4.6)
        sway = 2.2 * math.sin(t * 2 * math.pi / 9.0)
        # he leans into her hand on his cheek, then slowly straightens
        tilt = -0.02 + 0.015 * math.sin(t * 0.5) - 0.13 * smooth(15.0, 16.2, t) * (1 - smooth(16.8, 18.4, t))
        hand = smooth(30.6, 32.2, t)
        return tilt, breathe, sway, hand

    def her_pose(self, name, lt, t, mj):
        u = MAN["u"]
        if name in ("beside", "beside2"):
            fx, fy, s = 778.0, 708.0, 1.48
            if name == "beside":
                # reach for his shoulder, rest there, drift up to his cheek, let go
                rest = lerp2(mj["sh_l"], (mj["sh_l"][0] + 6 * u, mj["sh_l"][1] - 2 * u), 0.5 + 0.5 * math.sin(lt * 1.3))
                cheek = mj["cheek_l"]
                side = (fx + 10 * s, fy - 108 * s)
                a = smooth(0.55, 1.6, lt)
                b = smooth(3.15, 4.15, lt)
                c = smooth(5.15, 5.95, lt)
                tgt = lerp2(lerp2(lerp2(side, rest, a), cheek, b), side, c)
                # arc the hand up and over instead of a straight line
                tgt = (tgt[0], tgt[1] - 18 * s * math.sin(math.pi * a) * (1 - b))
                lean = 0.06 * a + 0.04 * b - 0.08 * c
                look = -0.1 - 0.12 * b + 0.15 * c
            else:
                tgt = lerp2((fx + 10 * s, fy - 108 * s), mj["el_l"], smooth(0.4, 1.3, lt))
                lean, look = 0.05, -0.12
            shapes, j = her_standing(fx, fy, s, t, reach=tgt, look=look, lean=lean)
            return shapes, [], (fx, fy - 140 * s)
        if name == "sofa":
            hx, hy, s = 270.0, 550.0, 1.12
            page = (lt - 2.55) / 0.6
            look_up = smooth(4.1, 4.6, lt) * (1 - smooth(5.0, 5.3, lt))
            shapes, book, j = her_sitting(hx, hy, s, t, page, look_up)
            return shapes, book, (hx + 20 * s, hy - 60 * s)
        # table: facing him (mirror), swirling the glass
        fx, fy, s = 990.0, 600.0, 0.92
        ang = lt * 2 * math.pi / 1.15
        sw = smooth(0.4, 1.0, lt)
        hand = (fx + 26 * s + 5 * s * math.cos(ang) * sw, fy - 152 * s + 2.5 * s * math.sin(ang) * sw)
        shapes, j = her_standing(fx, fy, s, t, reach=hand, look=-0.05 + 0.1 * smooth(2.5, 3.2, lt), lean=0.02)
        # wine glass in her hand: stem, bowl, wine sloshing behind the swirl
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
        return shapes, extra, (fx, fy - 140 * s)

    def hologram(self, img, shapes, extra, g, t, rng, raster, to_s, pos, z, her_light, name, lt):
        """Additive hologram with scanlines, edge glow, rain dropouts and glitch tearing."""
        m = raster(shapes)
        # tearing: bands slide sideways, some drop out; grows with g
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
                    m[gy:gy + gh] = np.roll(m[gy:gy + gh], int(rng.normal(0, 3 + 70 * tear * tear) * z), 1)
        alpha = (1 - g) ** 1.5
        mi = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8))
        edge = np.asarray(mi.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(0.7)), np.float32) / 255
        halo = np.asarray(mi.filter(ImageFilter.GaussianBlur(9 * z)), np.float32) / 255
        scan = 0.72 + 0.28 * np.sin((self.yy - t * 22) * 1.9)
        fine = 0.85 + 0.15 * (self.knit > 0.5)
        _, py = to_s(pos)
        grad = np.clip((self.yy - py + 160 * z) / (320 * z), 0, 1)[..., None]
        col = HOLO_A * grad + HOLO_B * (1 - grad)
        body = (m * 0.32 * scan * fine + edge * 1.1)[..., None] * col + halo[..., None] * col * 0.35
        # chromatic split when tearing
        if g > 0.02:
            sh = int(2 + 10 * g)
            body[..., 0] = np.roll(body[..., 0], sh, 1)
            body[..., 2] = np.roll(body[..., 2], -sh, 1)
            body *= 1 + 1.6 * math.sin(math.pi * g)
        img = img * (1 - 0.25 * m[..., None] * alpha) + body * flick * alpha
        # props: book pages a little brighter; the wine is the one red in the room
        for e in extra:
            if e[0] == "wine":
                wm = raster([e[1]])
                img += wm[..., None] * np.array([0.75, 0.04, 0.08]) * alpha * (0.9 + 0.1 * flick)
                img = img * (1 - 0.4 * wm[..., None] * alpha)  + wm[..., None] * np.array([0.55, 0.02, 0.05]) * alpha
            else:
                pm = raster([e])
                img += pm[..., None] * np.array([0.7, 0.8, 1.0]) * 0.7 * alpha * flick
        # she lights what is around her
        sx, sy = to_s(pos)
        light = np.exp(-(((self.xx - sx) / (260 * z)) ** 2 + ((self.yy - sy) / (300 * z)) ** 2))[..., None]
        img += light * HOLO_A * 0.06 * flick * alpha
        her_light = np.maximum(her_light, light * alpha * flick)
        # glitch debris drifting off her when she tears away / reassembles
        if g > 0.05:
            d = Image.new("L", (W, H), 0)
            dd = ImageDraw.Draw(d)
            for _ in range(int(60 * g)):
                bx = sx + rng.normal(0, 70 * z)
                by = sy + rng.normal(0, 120 * z) - 60 * g * z
                w_ = rng.integers(2, 10)
                dd.rectangle([bx, by, bx + w_ * z, by + max(1, z)], fill=int(255 * rng.uniform(0.3, 1)))
            img += (np.asarray(d, np.float32) / 255)[..., None] * HOLO_B * g
        return img, her_light

    def draw_man(self, img, parts, mj, raster, to_s, shadow, neon, her_light, t, z):
        masks = {k: raster(v) for k, v in parts.items()}
        alpha = np.clip(sum(masks.values()), 0, 1)
        hx, hy = to_s(mj["head"])
        # materials (dark: backlit by the lamps, front lit by the cool window)
        col = np.zeros((H, W, 3), np.float32)
        rib = 0.82 + 0.18 * np.sin((self.xx - hx) * 1.1 / z) ** 2  # knit ribbing
        knit = 0.85 + 0.3 * (self.knit - 0.5)
        col += masks["sweater"][..., None] * np.array([0.05, 0.055, 0.06]) * (rib * knit)[..., None]
        col += masks["pants"][..., None] * np.array([0.03, 0.028, 0.03]) * knit[..., None]
        col += masks["shoe"][..., None] * np.array([0.02, 0.018, 0.016])
        col += masks["hair"][..., None] * np.array([0.022, 0.018, 0.015]) * (0.8 + 0.4 * self.knit)[..., None]
        # skin: shaded face with sockets, nose ridge and jaw falloff
        u = MAN["u"] * z
        fx_ = (self.xx - hx) / (15 * u)
        fy_ = (self.yy - hy) / (19 * u)
        face = 1 - 0.55 * fx_ ** 2
        face -= 0.35 * np.exp(-(((np.abs(fx_) - 0.38) / 0.18) ** 2 + ((fy_ + 0.05) / 0.12) ** 2))  # eye sockets
        face += 0.25 * np.exp(-((fx_ / 0.07) ** 2 + ((fy_ - 0.12) / 0.25) ** 2))  # nose ridge
        face -= 0.2 * np.exp(-((fx_ / 0.3) ** 2 + ((fy_ - 0.52) / 0.05) ** 2))  # mouth
        face -= 0.25 * np.clip(fy_ - 0.35, 0, 1) * (np.abs(fx_) > 0.5)  # jaw shadow
        skin_c = np.array([0.2, 0.14, 0.115]) * np.clip(face, 0.25, 1.2)[..., None]
        col += masks["skin"][..., None] * skin_c
        # cool window light on his front, broken up by the water on the glass
        front = (1 - 0.8 * shadow)[..., None] * neon * 0.28
        col *= 1.0
        col += alpha[..., None] * front * np.array([0.32, 0.33, 0.34]) * (0.5 + 0.8 * masks["skin"])[..., None]
        # warm rim from the lamps behind him, mostly on the lamp side (left)
        edge = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8)).filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(1.0)), np.float32) / 255
        lamp_side = np.clip(0.55 - (self.xx - hx) / (160 * z), 0.15, 1)
        col += (edge * lamp_side)[..., None] * np.array([0.9, 0.55, 0.25]) * 0.55
        # her glow on him when she is close
        col += alpha[..., None] * her_light * HOLO_A * 0.25
        return img * (1 - alpha[..., None]) + col

    def rain_shadow(self, snap, ccx, ccy, z, t):
        """Low-res water pattern from the glass, projected into the room."""
        beads, rivs, trails = snap
        sw, sh = W // 4, H // 4
        im = Image.new("L", (sw, sh), 0)
        d = ImageDraw.Draw(im)
        # projected larger and offset: light passes the glass and spreads
        k = 1.35
        to = lambda x, y: (((x - ccx) * z * k + W / 2) / 4, ((y - ccy) * z * k + H / 2 - 30) / 4)
        for x, y, r, _ in rivs:
            px, py = to(x, y)
            rr = r * z * k / 4 + 0.6
            d.ellipse([px - rr, py - rr * 1.3, px + rr, py + rr * 1.3], fill=200)
        for tr in trails:
            pts = [to(x, y) for x, y, a in tr if a < 2.0]
            if len(pts) > 1:
                d.line(pts, fill=110, width=1)
        big = beads[beads[:, 2] > 1.4]
        for x, y, r in big[::2]:
            px, py = to(x, y)
            rr = r * z * k / 4
            d.ellipse([px - rr, py - rr, px + rr, py + rr], fill=150)
        im = im.filter(ImageFilter.GaussianBlur(1.2)).resize((W, H), Image.BILINEAR)
        return np.asarray(im, np.float32) / 255.0

    def glass_layer(self, img, snap, ccx, ccy, z, focus, t, rng):
        beads, rivs, trails = snap
        zg = 1 + (z - 1) * 1.5
        gcx = 720 + (ccx - 720) * 1.3
        gcy = 405 + (ccy - 405) * 1.3
        to = lambda x, y: ((x - gcx) * zg + W / 2, (y - gcy) * zg + H / 2)
        # faint reflection of the city behind the drone
        off = int(np.clip(200 - (ccx - 720) * 0.8, 0, 400))
        refl = self.reflect[:H, off:off + W]
        if refl.shape[1] == W:
            img = img + refl * 0.07
        # wet trails: thin water film -> slight shift, darker, a bright edge
        tm = Image.new("L", (W, H), 0)
        td = ImageDraw.Draw(tm)
        for tr in trails:
            pts = [to(x, y) for x, y, a in tr]
            if len(pts) > 1:
                w0 = max(3, int((2.0 + 0.6 * float(tr[:, 2].min() < 0.5) * 2) * zg * 1.6))
                td.line(pts, fill=int(200), width=w0, joint="curve")
        tmask = np.asarray(tm.filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255
        if tmask.max() > 0:
            shifted = np.roll(np.roll(img, 3, axis=1), -2, axis=0) * 1.12
            img = img * (1 - tmask[..., None] * 0.7) + shifted * tmask[..., None] * 0.7
            tedge = np.asarray(tm.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255
            img += tedge[..., None] * np.array([0.55, 0.6, 0.7]) * 0.07

        # drops as lenses: each shows an inverted, shrunken view of what is behind it
        src = img.copy()
        dispx = np.zeros((H, W), np.float32)
        dispy = np.zeros((H, W), np.float32)
        lens = np.zeros((H, W), np.float32)
        shade = np.zeros((H, W), np.float32)
        spec = np.zeros((H, W), np.float32)
        drops = [(x, y, r, 1.0) for x, y, r in beads] + [(x, y, r, 1.3 if s else 1.1) for x, y, r, s in rivs]
        for x, y, r, el in drops:
            px, py = to(x, y)
            rs = r * zg
            if rs < 0.9 or px < -10 or px > W + 10 or py < -10 or py > H + 10:
                continue
            ry = rs * el
            x0, x1 = int(max(0, px - rs - 1)), int(min(W, px + rs + 2))
            y0, y1 = int(max(0, py - ry - 1)), int(min(H, py + ry + 2))
            if x1 <= x0 or y1 <= y0:
                continue
            ox = (self.xx[y0:y1, x0:x1] - px) / rs
            oy = (self.yy[y0:y1, x0:x1] - py) / ry
            d2 = ox * ox + oy * oy
            inside = d2 < 1
            view = 18 + rs * 5
            dispx[y0:y1, x0:x1] = np.where(inside, -ox * (rs + view), dispx[y0:y1, x0:x1])
            dispy[y0:y1, x0:x1] = np.where(inside, -oy * (ry + view), dispy[y0:y1, x0:x1])
            lens[y0:y1, x0:x1] = np.maximum(lens[y0:y1, x0:x1], inside * 1.0)
            dd = np.sqrt(np.minimum(d2, 1))
            shade[y0:y1, x0:x1] = np.maximum(shade[y0:y1, x0:x1], inside * np.clip((dd - 0.45) / 0.55, 0, 1) * 0.85)
            spec[y0:y1, x0:x1] = np.maximum(spec[y0:y1, x0:x1], np.exp(-(((ox + 0.35) / 0.22) ** 2 + ((oy + 0.4) / 0.22) ** 2)) * inside)
        if lens.max() > 0:
            sx = np.clip(self.xx + dispx, 0, W - 1).astype(np.int32)
            sy = np.clip(self.yy + dispy, 0, H - 1).astype(np.int32)
            blurred = np.asarray(Image.fromarray((np.clip(src, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5)), np.float32) / 255
            refr = blurred[sy, sx] * 1.35 + 0.02
            refr = refr * (1 - shade[..., None]) + shade[..., None] * np.array([0.02, 0.025, 0.03])
            glass = img * (1 - lens[..., None]) + refr * lens[..., None] + spec[..., None] * np.array([0.9, 0.92, 1.0]) * 1.1
            # drops soften when the focus is in the room
            if focus > 0.05:
                r = 1.1 * focus
                gb = np.asarray(Image.fromarray((np.clip(glass, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r)), np.float32) / 255
                lm = np.asarray(Image.fromarray((lens * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r + 0.5)), np.float32)[..., None] / 255
                glass = img * (1 - lm) + gb * lm
            img = glass
        # window mullions at the frame edges (glass plane)
        fm = Image.new("L", (W, H), 0)
        fd = ImageDraw.Draw(fm)
        for gx in (60, 1380):
            a_ = to(gx, 0)
            fd.rectangle([a_[0] - 14 * zg, 0, a_[0] + 14 * zg, H], fill=255)
        ty = to(0, 96)[1]
        fd.rectangle([0, ty - 12 * zg, W, ty + 12 * zg], fill=255)
        fmask = np.asarray(fm, np.float32)[..., None] / 255
        img = img * (1 - fmask) + fmask * np.array([0.018, 0.02, 0.024])
        return img

    def finish(self, img, t, i):
        # bloom
        lum = img.mean(axis=2, keepdims=True)
        bright = np.clip(img * np.clip((lum - 0.42) * 2.5, 0, 1), 0, 2)
        small = Image.fromarray((np.clip(bright, 0, 1) * 255).astype(np.uint8)).resize((W // 4, H // 4), Image.BILINEAR)
        b1 = np.asarray(small.filter(ImageFilter.GaussianBlur(3)).resize((W, H), Image.BILINEAR), np.float32) / 255
        b2 = np.asarray(small.filter(ImageFilter.GaussianBlur(12)).resize((W, H), Image.BILINEAR), np.float32) / 255
        img = img + b1 * 0.6 + b2 * 0.8
        # grade: K's apartment - amber highlights, brown mids, cool lifted shadows
        lum = img.mean(axis=2, keepdims=True)
        img = img + (1 - np.clip(lum * 4, 0, 1)) * np.array([-0.004, 0.006, 0.014]) + np.clip(lum - 0.35, 0, 1) * np.array([0.08, 0.03, -0.05])
        img = lum + (img - lum) * 0.88
        img = 1 - np.exp(-img * 1.75)
        img[..., 0] = np.roll(img[..., 0], 1, axis=1)
        img[..., 2] = np.roll(img[..., 2], -1, axis=1)
        img = img * self.vignette + self.grain[i % 8][..., None] * 0.02
        fade = smooth(0, 1.5, t) * (1 - smooth(DUR - 2.0, DUR, t))
        img *= fade
        out = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
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
# Sound: rain on glass, room tone, synth pad, glitch teleports, glass + wine
# ----------------------------------------------------------------------------

def synth_audio(path, t0, t1):
    sr = 44100
    n = int(sr * DUR)
    t = np.arange(n) / sr
    rng = np.random.default_rng(SEED)
    inten = np.array([rain_intensity(x) for x in t[::441]])
    inten = np.interp(t, t[::441], inten)
    noise = rng.normal(0, 1, n)
    # muffled rain through the window + close patter on the glass
    rain = onepole(noise, 900, sr) * 0.22
    ticks = np.zeros(n)
    idx = rng.integers(0, n - 400, 26000)
    keep = rng.random(len(idx)) < inten[idx]
    ticks[idx[keep]] = rng.uniform(-1, 1, keep.sum())
    ticks = np.convolve(ticks, np.exp(-np.arange(200) / 18), mode="same")
    rain += (onepole(ticks, 7000, sr) - onepole(ticks, 1500, sr)) * 0.35
    rain *= 0.45 + 0.55 * inten

    def saw(freq, det):
        s = np.zeros(n)
        for h in range(1, 12):
            s += np.sin(2 * np.pi * freq * (1 + det) * h * t + h) / h
        return s

    pad = np.zeros(n)
    for f in (65.41, 98.0, 155.56, 196.0, 233.08):  # C minor, open voicing
        for d in (-0.004, 0.0, 0.005):
            pad += saw(f, d)
    pad = onepole(pad, 700, sr) * 0.03 * (0.4 + 0.6 * np.clip((t - 2) / 10, 0, 1))
    # glitch teleports: digital tearing bursts
    gl = np.zeros(n)
    for a, b, _ in POSES:
        for tt in (a - GLITCH, b):
            k0 = int(max(0, tt) * sr)
            ln = int(GLITCH * sr)
            seg = np.sign(np.sin(2 * np.pi * rng.uniform(300, 2400) * np.arange(ln) / sr * (1 + 3 * rng.random(ln))))
            seg *= rng.random(ln) > 0.4
            gl[k0:k0 + ln] += seg[: len(gl[k0:k0 + ln])] * 0.05 * np.hanning(ln)[: len(gl[k0:k0 + ln])]
    # soft hologram hum
    hum = np.sin(2 * np.pi * 120 * t) * 0.004 * (1 + np.sin(t * 7))
    # bell-like tones on the touches
    bells = np.zeros(n)
    for st, f in ((12.6, 587.33), (14.9, 698.46), (18.4, 523.25), (24.0, 783.99), (30.2, 587.33)):
        m = t >= st
        tt = t[m] - st
        bells[m] += (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt)) * np.exp(-tt / 2.4) * 0.05
    mix = rain + pad + gl + hum + bells
    st = np.stack([mix + rain[::-1] * 0.1, mix * 0.95 + np.roll(rain, 2000) * 0.1], 1)
    st *= (np.clip(t / 1.5, 0, 1) * np.clip((DUR - t) / 2, 0, 1))[:, None]
    st /= np.abs(st).max() / 0.9
    seg = st[int(t0 * sr):int(t1 * sr)]
    if t0 > 0:  # short fade-in on excerpts
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
           "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "slow", "-crf", "21",
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
