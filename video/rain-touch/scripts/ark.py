#!/usr/bin/env python3
"""Volcengine Ark client for the "rain-touch" short.

Generates keyframes with Seedream and clips with Seedance, using prompts
from ../shots.json. Standard library only.

  python3 scripts/ark.py image k -n 4          # character sheet variants
  python3 scripts/ark.py approve k output/images/k_2.png
  python3 scripts/ark.py image s02 -n 4        # keyframe (uses approved refs)
  python3 scripts/ark.py approve s02 output/images/s02_1.png
  python3 scripts/ark.py video s02 --fast      # draft clip with the cheap model
  python3 scripts/ark.py video s02             # final clip
  python3 scripts/ark.py cost                  # spend so far

Environment:
  ARK_API_KEY           optional when the key is stored as a Network secret for
                        ark.cn-beijing.volces.com (the proxy adds the header)
  ARK_IMAGE_MODEL       Seedream model / endpoint id (default below)
  ARK_VIDEO_MODEL       Seedance 2.0 model / endpoint id (required for video)
  ARK_VIDEO_MODEL_FAST  Seedance 2.0 Mini model / endpoint id (for --fast)
  ARK_BASE_URL          default https://ark.cn-beijing.volces.com/api/v3
"""

import argparse
import base64
import json
import mimetypes
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
APPROVED = OUT / "approved"
LOG = OUT / "log.jsonl"

BASE_URL = os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
IMAGE_MODEL = os.environ.get("ARK_IMAGE_MODEL", "doubao-seedream-4-0-250828")

# Rough RMB prices for the spend log; check the Ark console for real ones.
PRICE_IMAGE = float(os.environ.get("ARK_PRICE_IMAGE", "0.25"))  # per image
PRICE_VIDEO = float(os.environ.get("ARK_PRICE_VIDEO", "46"))  # per 1M tokens
PRICE_VIDEO_FAST = float(os.environ.get("ARK_PRICE_VIDEO_FAST", "23"))


def load_spec():
    with open(ROOT / "shots.json", encoding="utf-8") as f:
        return json.load(f)


def find_target(spec, target):
    if target in spec["characters"]:
        return spec["characters"][target]
    for shot in spec["shots"]:
        if shot["id"] == target:
            return shot
    sys.exit(f"unknown target: {target}")


def full_prompt(spec, item):
    style = item.get("style")
    if style:
        return f"{item.get('image_prompt') or item['prompt']}, {spec['styles'][style]}"
    return item.get("image_prompt") or item["prompt"]


def data_uri(path):
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode()}"


def approved_path(target):
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        path = APPROVED / f"{target}{ext}"
        if path.exists():
            return path
    sys.exit(f"{target} has no approved image yet: run `ark.py approve {target} <file>`")


def api(method, path, body=None):
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("ARK_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        hint = "" if e.code != 401 else " (no API key: set ARK_API_KEY or a Network secret for this host)"
        sys.exit(f"HTTP {e.code}{hint}: {e.read().decode(errors='replace')}")


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)
    return dest


def log(entry):
    OUT.mkdir(parents=True, exist_ok=True)
    entry["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def redact(body):
    """Shorten base64 images so dry runs stay readable."""
    text = json.dumps(body, ensure_ascii=False, indent=2)
    return "\n".join(
        line if "data:image" not in line else line.split("base64,")[0] + 'base64,..."' + ("," if line.endswith(",") else "")
        for line in text.splitlines()
    )


def cmd_image(args):
    spec = load_spec()
    item = find_target(spec, args.target)
    body = {
        "model": IMAGE_MODEL,
        "prompt": full_prompt(spec, item),
        "size": item.get("size", "3024x1296"),
        "response_format": "url",
        "watermark": False,
    }
    refs = [data_uri(approved_path(r)) for r in item.get("refs", [])]
    if refs:
        body["image"] = refs
    if args.dry_run:
        print(redact(body))
        return
    for i in range(1, args.n + 1):
        res = api("POST", "/images/generations", body)
        dest = download(res["data"][0]["url"], OUT / "images" / f"{args.target}_{int(time.time())}_{i}.png")
        log({"kind": "image", "target": args.target, "file": str(dest.relative_to(ROOT)), "cost_rmb": PRICE_IMAGE})
        print(dest)


def cmd_approve(args):
    APPROVED.mkdir(parents=True, exist_ok=True)
    src = Path(args.file)
    kinds = (".mp4",) if src.suffix == ".mp4" else (".png", ".jpg", ".jpeg", ".webp")
    for ext in kinds:
        (APPROVED / f"{args.target}{ext}").unlink(missing_ok=True)
    shutil.copy(src, APPROVED / f"{args.target}{src.suffix}")
    print(f"approved {args.file} as {args.target}")


def cmd_video(args):
    spec = load_spec()
    shot = find_target(spec, args.target)
    model = os.environ.get("ARK_VIDEO_MODEL_FAST" if args.fast else "ARK_VIDEO_MODEL")
    if not model and not args.dry_run:
        sys.exit("set ARK_VIDEO_MODEL / ARK_VIDEO_MODEL_FAST to the model or endpoint id from the Ark console")
    duration = args.duration or max(4, shot["duration"])
    text = (
        f"{shot['video_prompt']} --resolution {args.resolution} --duration {duration}"
        f" --ratio adaptive --watermark false"
    )
    body = {
        "model": model or "<ARK_VIDEO_MODEL>",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": data_uri(approved_path(args.target))}, "role": "first_frame"},
        ],
    }
    if args.dry_run:
        print(redact(body))
        return
    task = api("POST", "/contents/generations/tasks", body)
    print(f"task {task['id']} submitted, waiting...")
    while True:
        time.sleep(10)
        res = api("GET", f"/contents/generations/tasks/{task['id']}")
        if res["status"] in ("succeeded", "failed", "cancelled", "expired"):
            break
        print(f"  {res['status']}")
    if res["status"] != "succeeded":
        sys.exit(f"task {res['status']}: {json.dumps(res.get('error'), ensure_ascii=False)}")
    tag = "fast" if args.fast else "final"
    dest = download(res["content"]["video_url"], OUT / "videos" / f"{args.target}_{tag}_{int(time.time())}.mp4")
    tokens = (res.get("usage") or {}).get("total_tokens", 0)
    cost = tokens / 1e6 * (PRICE_VIDEO_FAST if args.fast else PRICE_VIDEO)
    log({"kind": "video", "target": args.target, "file": str(dest.relative_to(ROOT)), "tokens": tokens, "cost_rmb": cost})
    print(f"{dest}  ({tokens} tokens, ~{cost:.2f} RMB)")


def cmd_cost(_args):
    if not LOG.exists():
        print("nothing spent yet")
        return
    entries = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line]
    for kind in ("image", "video"):
        rows = [e for e in entries if e["kind"] == kind]
        print(f"{kind}: {len(rows)} generated, ~{sum(e['cost_rmb'] for e in rows):.2f} RMB")
    total = sum(e["cost_rmb"] for e in entries)
    print(f"total: ~{total:.2f} RMB (~{total * 4.4:.0f} TWD)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("image", help="generate keyframe / character sheet variants")
    s.add_argument("target")
    s.add_argument("-n", type=int, default=4)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_image)

    s = sub.add_parser("approve", help="mark an image or clip as the chosen one for a target")
    s.add_argument("target")
    s.add_argument("file")
    s.set_defaults(func=cmd_approve)

    s = sub.add_parser("video", help="generate a clip from the approved keyframe")
    s.add_argument("target")
    s.add_argument("--fast", action="store_true", help="use the cheaper draft model")
    s.add_argument("--duration", type=int)
    s.add_argument("--resolution", default="1080p")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_video)

    s = sub.add_parser("cost", help="show spend so far")
    s.set_defaults(func=cmd_cost)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
