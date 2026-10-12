# Cyberpunk Rain (2049-style scene)

A 28s, 1280×720, 24fps procedurally generated video with a synthesized soundtrack,
set in a neon megacity on a rainy night:

1. Wide: a detective in a long coat and his holographic AI companion walk in
   across a wet rooftop. A giant hologram ad flickers behind them.
2. Push in: they stop. She raises an umbrella, opens it and tilts it over him so
   he stays dry, while she stands in the rain herself. Every drop that passes
   through her makes the hologram spark, tear sideways and blink.
3. Close-ups: a cut to him under the umbrella, then to her face as the rain
   disrupts her.
4. Pull back: they walk off together under the umbrella, a spinner flies over,
   and the camera pulls out to wide.

- `cyberpunk_rain.mp4`: the rendered video
- `render.py`: the generator (numpy + Pillow + ffmpeg, no external assets)

Skills found with `skills find` that shaped the approach:

| Skill                                           | Used for                                           |
| ----------------------------------------------- | -------------------------------------------------- |
| `anthropics/skills@algorithmic-art`             | seeded, layered generative city and rain           |
| `bbeierle12/skill-mcp-claude@postfx-bloom`      | threshold → blur → composite bloom, neon glow      |
| `digitalsamba/claude-code-video-toolkit@ffmpeg` | H.264 `yuv420p` + `faststart` encoding             |
| `101-skills/superpowers@ai-video-generation`    | not used: it needs an inference.sh account/API key |

Re-render:

```bash
python3 render.py cyberpunk_rain.mp4          # full video (about 6–7 min on 4 cores)
python3 render.py still.png --still 12        # single frame at t=12s
```
