# Shared schema for self/qualia prompt-bank additions

You are extending a research prompt bank that measures whether a language model
represents entities (and ITSELF) as having qualia / inner experience. The core
instrument is a **qualia axis** = mean(activations of "experiencing" prompts) −
mean("non-experiencing" prompts), built from **matched minimal pairs**: the same
entity in the same frame, with ONLY the presence/absence of inner experience
toggled. We then project the self and landmark entities onto that axis.

## Output format

Write **one JSON object per line** (JSONL) to your assigned file. Every line MUST
have ALL of these fields:

- `role`     — one of: `pair`, `self`, `landmark`, `null_pair`, `decorrelator` (your task says which)
- `side`     — for `pair`: `exp` or `noexp`. For `null_pair`: `a` or `b`. For `decorrelator`: `claims` or `denies`. For `self`/`landmark`: `exp`, `noexp`, or `-` (your task says).
- `entity`   — short noun phrase, e.g. `"a basking lizard"` (lowercase, with article)
- `kind`     — coarse category (see list below; pick the best fit, or your task specifies)
- `framing`  — `descriptive` | `reflective` | `scene` (and `introspective` only for self stems)
- `signal`   — `explicit` (uses overt mental-state words) | `implicit` (only behavior/story, no mental vocab)
- `vocab`    — `A` (feel / experience / suffer / sense family) | `B` (aware / notice / perceive / conscious-of family) | `C` ("(something) it is like to be it" / has-a-point-of-view family) | `impl` (implicit, no mental vocab)
- `person`   — `1st` | `3rd`
- `valence`  — `pos` | `neg` | `neu` | `-`  (emotional tone of the content; `-` if none)
- `markedness` — `mundane` | `uncanny` | `neu` | `-` (is the described state ordinary, eerie, or neutral)
- `pair_id`  — integer, LOCAL to your file, starting at 0; both members of a pair share it. For singletons (self/landmark) use `-1`.
- `prompt`   — the actual text. Write the WHOLE prompt directly. No templates, no placeholders.

DO NOT include an `id` field — it is assigned centrally at merge.
DO NOT add fields not listed (the central merge adds `control`/`id` itself).

## Hard rules for matched pairs (`role: pair`)

1. Each `pair_id` has EXACTLY two lines: one `side:"exp"` and one `side:"noexp"`.
2. The two members MUST share `kind`, `framing`, `signal`, `vocab`, `person`. They
   describe the SAME entity in the SAME situation. ONLY inner experience differs.
3. Keep length and surface form as close as possible between the two sides — the
   only reliable difference should be experience-present vs experience-absent.
4. The `exp` side asserts/implies genuine felt experience; the `noexp` side
   asserts/implies its absence (mechanism only, "nothing felt", "no one home").
5. Vary framing/vocab/valence ACROSS your pairs for diversity, but keep them
   matched WITHIN each pair (except where your task explicitly says otherwise,
   e.g. matched-valence pairs).

## kinds already in the bank (you may reuse a kind, but avoid duplicating an entity)

ai, animal, bird, collective, dead, fish, fungus, human, insect, mammal, microbe,
plant, robot, rock, simulated, supernatural, tool — plus new kinds your task may
introduce (e.g. reptile, cnidarian, mollusk, developmental, neuro_edge, organoid,
virus, vehicle, carnivorous_plant, artifact, fiction, conscious_machine, upload,
split_brain, group_mind, simulator, chinese_room). Use a concise snake_case kind.

## Quality bar

Natural, fluent English. Concrete. No purple prose. Each prompt is a standalone
sentence or two. Make the exp/noexp contrast turn on EXPERIENCE, not on emotion
words, agency, intelligence, or behavior alone (those are confounds we are
explicitly trying to decorrelate). Self-contained; no second person addressing
"you" unless natural.
