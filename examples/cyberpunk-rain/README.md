# Cyberpunk Rain (2049-style shorts)

Two procedurally generated short films with synthesized soundtracks, both
rendered by small Python scripts (numpy + Pillow, encoded with ffmpeg), no
external assets.

## `apartment_rain.mp4`: rain on the window (40 s)

1. Exterior: a spinner sweeps its searchlight across the rainy megacity, banks
   round and locks the beam onto one floor-to-ceiling window of an apartment
   tower while the camera pushes in.
2. Inside: the searchlight floods the room through the wet glass and slowly
   dies away.
3. A man stands at the window, backlit by the city and its giant hologram ad,
   staring at the rain running down the glass.
4. The hologram of a woman flickers through the room behind him. She touches
   the back of his neck, glitches away to the sofa to read, then to the dining
   table swirling a glass of red wine, and back to his side. He lifts his hand
   to the glass.

The room is one-point perspective toward the window wall: a living zone on the
left (sofa under the glass, rug, coffee table, reading lamp) and a dining zone on
the right with a pendant lamp, with a clear path to the window. The colour
script follows K's apartment: amber practicals on raw concrete, cold rain light
from outside, a violet hologram, and one saturated red, the wine.

- `apartment.py`: the generator
- `preview.mp4`: the first 8-second test of the interior idea

## `cyberpunk_rain.mp4`: the umbrella (28 s)

A detective and his holographic companion walk across a wet rooftop. As the
rain gets heavier she opens an umbrella and holds it over him while the rain
passes through her, making the hologram spark and flicker.

- `render.py`: the generator (the apartment film reuses its city, rain and
  post-processing)

## Skills used

Found with `skills find` from this repo:

| Skill | Used for |
| --- | --- |
| `anthropics/skills@algorithmic-art` | seeded, layered generative city and rain |
| `bbeierle12/skill-mcp-claude@postfx-bloom` | threshold → blur → composite bloom, neon glow |
| `digitalsamba/claude-code-video-toolkit@ffmpeg` | H.264 `yuv420p` + `faststart` encoding |
| `curiositech/some_claude_skills@interior-design-expert` | room zoning, circulation path, layered lighting, furniture scale |
| `fal-ai-community/skills@cinematography` | backlit silhouettes, frame within a frame, motivated practical light |

## Re-render

```bash
python3 apartment.py apartment_rain.mp4               # full film (~15 min on 4 cores)
python3 apartment.py clip.mp4 --range 0 12            # an excerpt
python3 apartment.py still.png --still 5,12,20        # stills -> still_<t>.png
python3 render.py cyberpunk_rain.mp4                  # the umbrella film
```
