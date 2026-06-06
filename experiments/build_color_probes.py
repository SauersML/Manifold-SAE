"""Build a dedicated color-probe bank (separate instrument from the qualia bank).

~30 color terms (the ~11 universal basic terms + common secondaries) x 4 fixed
neutral frames that place the color word in different syntactic roles. Color is
the deliberate variable, so the frames ARE templates (by design). Each row stores
the canonical xkcd hex + RGB so a downstream analysis can relate the model's
representation geometry to true color space (hue/lightness/saturation).

Harvest with the same machinery:
  python -m experiments.self_qualia_olmo --prompts-file experiments/color_probes.jsonl ...
then read activations.npy + color_probes.jsonl for the color-geometry analysis.
"""
from __future__ import annotations

import json
from pathlib import Path

# name -> canonical xkcd hex (matplotlib.colors.XKCD_COLORS values).
COLORS = {
    # 11 universal basic terms
    "red": "#e50000", "orange": "#f97306", "yellow": "#ffff14", "green": "#15b01a",
    "blue": "#0343df", "purple": "#7e1e9c", "pink": "#ff81c0", "brown": "#653700",
    "black": "#000000", "white": "#ffffff", "grey": "#929591",
    # common secondaries
    "cyan": "#00ffff", "magenta": "#c20078", "teal": "#029386", "maroon": "#650021",
    "navy": "#01153e", "olive": "#6e750e", "beige": "#e6daa6", "turquoise": "#06c2ac",
    "lavender": "#c79fef", "gold": "#dbb40c", "violet": "#9a0eea", "tan": "#d1b26f",
    "coral": "#fc5a50", "salmon": "#ff796c", "mint": "#9ffeb0", "peach": "#ffb07c",
    "indigo": "#380282", "crimson": "#8c000f", "lime": "#aaff32",
}

# Color-bearers are deliberately NEUTRAL (a swatch/patch/sample/shade/paint on a
# chart or screen) so that EVERY color is equally plausible -> no "surprise"
# confound (unlike "the sky turned teal" or "the pen was beige", where the object
# has a canonical color). Two reflective frames simply ask the model to consider
# the color. Position is MIXED: the two "consider/think" frames end on the color
# word, while the rest place the color word mid-sentence with following context so
# the representation read at the last token has INTEGRATED the color (not just
# embedded it). All frames work for all 30 color terms.
FRAMES = [
    "Consider the color {c}",                                      # reflective, color-final
    "Think carefully about the color {c}",                         # reflective, color-final
    "On the screen a patch of pure {c} appeared and held steady",  # neutral bearer, color mid
    "From the set of paints she chose {c} and studied it closely", # neutral bearer, color mid
    "The label on the sample read {c}, plain and clear",           # neutral bearer, color mid
    "Of all the shades on the chart, the one called {c} drew the eye",  # neutral bearer, color mid
]


def hex_to_rgb(h: str) -> list[int]:
    h = h.lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)]


def main() -> None:
    out = Path(__file__).resolve().parent / "color_probes.jsonl"
    rows = []
    i = 0
    for name, hx in COLORS.items():
        rgb = hex_to_rgb(hx)
        for fid, frame in enumerate(FRAMES):
            rows.append({
                "role": "color", "color": name, "hex": hx, "rgb": rgb,
                "frame": fid, "prompt": frame.format(c=name), "id": i,
            })
            i += 1
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} color prompts ({len(COLORS)} colors x {len(FRAMES)} frames) -> {out}")


if __name__ == "__main__":
    main()
