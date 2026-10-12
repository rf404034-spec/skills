# Cyberpunk Rain (2049-style scene)

A 22s, 1280×720, 24fps procedurally generated video with a synthesized soundtrack:
a detective in a long coat and his holographic AI companion stand on a wet rooftop
in a neon megacity at night. It rains, a spinner flies past, and a giant hologram
ad flickers. She raises an umbrella, opens it and tilts it over him so he stays
dry, while she stands in the rain herself. Every drop that passes through her
makes the hologram spark, tear sideways and blink.

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
python3 render.py cyberpunk_rain.mp4          # full video (about 4–5 min on 4 cores)
python3 render.py still.png --still 12        # single frame at t=12s
```
