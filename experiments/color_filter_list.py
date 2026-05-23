"""xkcd color names with strong non-color connotations.

These names either don't function as color adjectives in our 28 template
sentences (e.g., "She slipped into a {x} silk dress"), have strong
disgust/object/brand associations that override the color reading, or
are typos/joke words. Filtering them out should sharpen the supervised
R² because each remaining centroid is averaging "model understood as
color" forward passes rather than mixing in "model parsed as object".

Compiled by reading through all 949 xkcd entries and flagging by
criterion. Categories:
  • disgust / bodily-fluid / waste — overrides any color reading
  • negative descriptive adjectives — warp the meaning
  • brand / proper-noun references
  • toxicity / danger connotations
  • misspellings / joke words
  • culturally loaded names
  • too-vague generic descriptors

Keep: salmon, coral, olive, mustard, lemon, peach, apricot, cinnamon,
banana, mango, melon, pumpkin, eggplant, lavender, mint, mocha, sepia,
ochre, terracotta, sage, mauve, etc. — established food/object-derived
color terms. Also keep "watermelon" by judgment (xkcd's intent is the
pink-red shade), "hospital green" (institutional shade), "macaroni and
cheese" (yellow-orange food shade that does function as adjective).
"""

# Sorted by xkcd index for review/debug.
COLOR_FILTER = {
    # ----- Disgust / bodily fluids / waste -----
    "booger", "booger green",
    "snot", "snot green",
    "puke", "puke green", "puke yellow", "puke brown", "baby puke green",
    "barf green",
    "vomit", "vomit green", "vomit yellow",
    "shit", "shit green", "shit brown",
    "baby shit brown", "baby shit green",
    "poo", "poo brown", "poop", "poop green", "poop brown",
    "baby poo", "baby poop", "baby poop green",
    "diarrhea",
    "piss yellow",
    "bile",

    # ----- Body-trauma / blood overrides -----
    "bruise",
    "dried blood",
    "blood",                    # standalone "blood" — too literal
    # KEEP "blood red", "blood orange" — these are canonical color names

    # ----- Negative descriptive adjectives that warp the reading -----
    "ugly brown", "ugly blue", "ugly pink", "ugly purple",
    "ugly yellow", "ugly green",
    "nasty green",
    "icky green",
    "gross green",
    "sickly yellow", "sickly green",
    "sick green",
    "weird green",
    "murky green",

    # ----- Brand / proper nouns / characters -----
    "windows blue",             # Microsoft Windows
    "barney", "barney purple",  # Barney the dinosaur
    "kermit green",             # Kermit the Frog

    # ----- Toxicity / danger connotations -----
    "poison green",
    "toxic green",
    "radioactive green",

    # ----- Misspellings / joke words -----
    "burple", "blurple",        # purple-blue portmanteaux
    "perrywinkle",              # misspelling of periwinkle
    "toupe",                    # misspelling of taupe
    "liliac",                   # misspelling of lilac
    "light lavendar",           # misspelling of lavender

    # ----- Culturally loaded / problematic -----
    "indian red",               # historical pigment name, loaded

    # ----- Too-vague generic descriptors (not really color names) -----
    "dark",                     # just "dark", no axis
    "pale",                     # just "pale"
    "bland",                    # descriptor, not color
}


def filter_colors(colors):
    """Drop entries whose name is in COLOR_FILTER. Return (kept_colors, kept_indices)."""
    kept = []
    kept_idx = []
    for i, c in enumerate(colors):
        name = c[0] if isinstance(c, tuple) else c
        if name in COLOR_FILTER:
            continue
        kept.append(c)
        kept_idx.append(i)
    return kept, kept_idx


if __name__ == "__main__":
    # Sanity report — print what would be filtered from the live xkcd list.
    import sys
    sys.path.insert(0, "/Users/user/Manifold-SAE/experiments")
    from color_geometry import load_xkcd_colors
    colors = load_xkcd_colors()
    kept, kept_idx = filter_colors(colors)
    dropped = [c for c in colors if c[0] in COLOR_FILTER]
    not_found = COLOR_FILTER - {c[0] for c in colors}
    print(f"xkcd has {len(colors)} colors")
    print(f"filter list has {len(COLOR_FILTER)} entries")
    print(f"dropped from xkcd: {len(dropped)}")
    print(f"kept: {len(kept)}")
    if not_found:
        print(f"NOT FOUND in xkcd (typos in filter?): {sorted(not_found)}")
    print()
    print("Dropped (with hex):")
    for name, r, g, b in dropped:
        print(f"  {name:30s}  #{r:02x}{g:02x}{b:02x}")
