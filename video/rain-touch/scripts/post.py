#!/usr/bin/env python3
"""Grade, letterbox and assemble approved clips into output/final.mp4.

Takes output/approved/<shot>.mp4 for each shot in ../shots.json (missing
shots are skipped, so partial previews work), trims to the shot duration,
applies one shared grade plus film grain, 2.39:1 letterbox, a hologram
glitch on cuts marked `glitch_in`, and a fade on `fade_out`.

  python3 scripts/post.py
  python3 scripts/post.py --only s02 s03     # preview a subset
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
W, H, BAR_H = 1920, 1080, 804  # 1920x804 is 2.39:1 inside a 16:9 frame

# Warm shadows, cool highlights, lifted blacks, low saturation.
GRADE = (
    "eq=contrast=1.06:saturation=0.78:gamma=0.97,"
    "colorbalance=rs=0.05:gs=0.01:bs=-0.04:rh=-0.03:bh=0.05,"
    "curves=all='0/0.05 0.5/0.48 1/0.94'"
)
GLITCH = (
    "rgbashift=rh=-14:bh=14:enable='lt(t,0.25)',"
    "eq=brightness=0.25:enable='between(t,0.04,0.08)+between(t,0.16,0.2)',"
    "noise=alls=70:allf=t:enable='lt(t,0.12)'"
)
GRAIN = "noise=alls=9:allf=t+u"


def render(shot, src, dest):
    dur = shot["duration"]
    chain = [
        "fps=24",
        f"scale={W}:{BAR_H}:force_original_aspect_ratio=increase",
        f"crop={W}:{BAR_H}",
        GRADE,
    ]
    if shot.get("glitch_in"):
        chain.append(GLITCH)
    if shot.get("fade_out"):
        chain.append(f"fade=t=out:st={dur - 2}:d=2")
    chain += [GRAIN, f"pad={W}:{H}:0:(oh-ih)/2:black", "format=yuv420p"]
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-t", str(dur), "-an",
         "-vf", ",".join(chain), "-c:v", "libx264", "-crf", "16", "-preset", "slow", str(dest)],
        check=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", help="shot ids to include")
    args = p.parse_args()

    spec = json.loads((ROOT / "shots.json").read_text(encoding="utf-8"))
    (OUT / "post").mkdir(parents=True, exist_ok=True)
    parts = []
    for shot in spec["shots"]:
        if args.only and shot["id"] not in args.only:
            continue
        src = OUT / "approved" / f"{shot['id']}.mp4"
        if not src.exists():
            print(f"skip {shot['id']}: no approved clip")
            continue
        dest = OUT / "post" / f"{shot['id']}.mp4"
        render(shot, src, dest)
        parts.append(dest)
        print(f"graded {shot['id']}")
    if not parts:
        sys.exit("no clips to assemble")

    listing = OUT / "post" / "concat.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in parts))
    final = OUT / "final.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c", "copy", str(final)],
        check=True,
    )
    print(final)


if __name__ == "__main__":
    main()
