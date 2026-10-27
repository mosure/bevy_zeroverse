# Mezzanine, stair and guardrail diversity

Mezzanines are sampled as recorded construction programs in eligible tall rooms.
They share one plan across mesh construction, floor support, camera/furniture
clearance, ceiling headroom, live schematics and manifest export. Proposal and
construction-detail random streams are independent. Admission is bounded to 40
proposals; architectural constraints can change the accepted distribution.

## Parameter space

| Component | Sampled dimensions and relationships |
|---|---|
| Placement | Both X/Z run axes, both ascent directions, continuous footprint position and stair opening position along the deck edge |
| Deck | 2.70–3.42 m elevation, 0.15–0.29 m slab, 2.1–3.7 m depth and 2.9–6.0 m breadth, subject to room fit |
| Frame | Wood, steel or concrete; square, round or steel I-section columns; 4/6/8 deck supports, variable section widths/depths and support bays |
| Stairs | 0.95–1.65 m width, 0.275–0.35 m nominal tread depth, continuously sampled riser target; solid, closed-stringer and open-stringer construction |
| Landings | Optional inline landing, varying riser index and 0.85–1.55 m additional run; framed landings have grounded supports and a cross-beam |
| Finishes | Separate frame, deck, tread, inlaid anti-slip nosing, guard frame, infill and handrail materials; concrete frames use solid stairs |
| Guardrails | Glass, pickets, slats or opaque panels; round/square posts and handrails; 1.03–1.22 m height, 0.70–1.45 m nominal post pitch and bounded interior-bay variation |
| Raised-floor edges | The same guardrail programs cover exposed level changes, replacing a separate fixed rail recipe |

Uniform rise within a flight, connected stair/deck surfaces, supported landings
and coordinated framing are retained deliberately. Independent noise in every
structural member would create implausible buildings rather than useful variety.
The deck remains rectangular; stair paths are straight with optional inline
landings. Turning flights, spiral stairs and cantilevered decks are not included.

Corner posts are shared between rail spans. Cross-beams and longitudinal ties
meet without overlapping their underside faces. Stair nosings replace a region
of the tread cap instead of overlaying a coplanar surface. Clearance reserves
remain conservative between open risers and guard infill; exact rendering and
visibility annotations use the actual triangles.

## Export and replay

`Mezzanine::construction` and `EnvelopeProgram::tier_guard` retain dimensions,
materials and per-installation seeds. Archives without these optional fields use
the retained construction path. The construction revision is `envelope=7`;
each dataset records its complete generator and capture identity for replay.

Dataset coverage reports include `mezzanine_stair_axis`,
`mezzanine_stair_direction`, `mezzanine_stair_structure`,
`mezzanine_support_section`, landing presence, component materials, guard infill,
and numeric dimensions, heights, bays and support counts. `mezzanine_stair_tread_m`
excludes the additional landing run. Equivalent `tier_guard_*` measurements are
recorded only when that room has exposed tier guards.

## Local qualification

The opt-in `mezzanine_population_qualification` Rust test writes
`out/mezzanine_diversity/report.json`, a twelve-cell actual-mesh gallery and a
local HTML review page. It samples 512 eligible tall-room fixtures and 256
consecutive ordinary Mixed room manifests. Every admitted mezzanine room receives
full CPU geometry validation. The companion tests exercise legacy replay,
both ascent directions and axes, guard-post deduplication, stair/deck connectivity,
headroom, beam joints, finite normals/UVs, winding, triangle budgets, conservative
bounds and independent ray-to-tread support agreement.

The gallery uses CPU flat shading, with opaque glass for mesh inspection. Mesh
hash uniqueness and parameter coverage are diagnostics, not estimates of
perceptual capacity, proof against memorization, or downstream training results.
CPU construction timing is recorded separately from scene generation; it does
not measure GPU rendering or concurrent-training throughput. Qualification logs
and current measurements remain local alongside the report.

The 512-fixture run covered all twelve stair-construction/infill combinations,
all four axis/direction combinations and 172 inline landings. Deck areas ranged
from 6.51 to 21.53 m², stair widths from 0.953 to 1.649 m, and deck elevations
from 2.706 to 3.417 m. The 256-room Mixed cohort admitted twelve mezzanines,
including three landings; all twelve passed complete geometry validation.
These accepted-room counts expose the effect of room fit rather than assuming
that every available construction appears equally often.

Mesh construction measured 0.192 ms median, 0.287 ms p95 and 0.387 ms maximum
across the fixtures, with 5,532 median and 9,480 maximum triangles. Measurements
use the optimized CPU test build and serial fixture study, with a two-job
build budget while production training remains active; they are a bounded local construction study,
not a controlled whole-renderer throughput comparison. Strict workspace Clippy,
WASM viewer compilation with human motion, publication diagrams, scene metrics,
schematics and architectural regression tests passed.

A further camera regression passed 96 mixed room/ceiling rigs across camera
counts 1–12 and eight ordinary room seeds, including mezzanines at seeds 3 and 7.
It checks placement, motion/lens roles, ceiling mounts, replay and serialization.
The completed checks comprise 146 distinct passing tests plus the local mesh
population study. Hardware rendering and downstream generalization were not
re-qualified in this pass.
