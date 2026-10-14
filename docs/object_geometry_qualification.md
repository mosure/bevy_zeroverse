# Object construction qualification

The current local implementation is generator 40 / capture-v58, source
`92932d0959232f1bbc47312cde5d1feb1010ba7f10d35bfc6a597ebc5b32a7c5`.
Package versions remain unchanged. This work has not been published.

## Construction changes

The review concentrated on builders where changing a seed previously changed
little beyond material or a few dimensions. It adds independent construction
controls rather than multiplying nominal style labels:

| Objects | Realized construction controls |
| --- | --- |
| Keyboard, mouse | Split key fields, row/column counts, stagger, wrist rest; curved fitted mouse shells, waist/crown, roof-following buttons, wheel/trackball |
| Books, notebooks, notepads, pencils | Unequal bound volumes, covers/pages/spines, binding and straps, pencil cross sections and tips |
| Floor lamps, sconces | Support topology, shade profile/section, offsets, body taper and proportions; a shared program locates both the rendered emitter and native/GI light |
| Printer, storage box, bag, coat rack | Rounded/tapered housings, fitted covers and handles, tray proportions, support and hook arrangements |
| Headphones, speakers, fans, stationery equipment | Cup/band and driver proportions, support mechanisms, reels, tray stacks/posts and compartments |
| Bottles, cans, phones, microphones | Neck/shoulder/heel profiles, bezels and lenses, bases and continuous gooseneck paths |
| Whiteboards, projection screens, shelves | Frame/tray/mount construction, roller versus framed screens, drop and support positions |
| Rugs, floor outlets, drawer units, service plates | Fitted outline/corner treatments, hinged lid proportions, shared cable bounds, drawer height/handle and gang variation |

The reusable rounded housing loft has independent caps, smooth side normals and
two-dimensional side UVs. The review also corrected protruding mouse button/base
geometry, disconnected printer/storage fittings and coplanar tray joints.
Render and collision/illumination consumers retain the same program inputs.
Expensive shape descriptors are confined to explicitly invoked audit tools.

## Current category evidence

The formal Rust publication pipeline owns the audit, native studio renderer,
checksum-bound atlas, project-page grids and paper figures/tables. `refresh`
checks generator and qualification-tool identities before reusing results;
`verify` checks the self-contained exported artifacts. It does not depend on
manually ordered Python publication steps.

All 64 furnishing/prop categories are audited on 256 consecutive construction
seeds: **16,384 assemblies**. Category envelopes, pose and finish seed remain
fixed. Six 24-by-24 orthographic first-surface depth maps describe shape. Three
thresholds (0.005, 0.015, 0.030) expose the sensitivity of ordered greedy coverage.
Each category has six deterministic spread-selected native RGB/normal examples,
all seed-level inputs, parameter marginals, mesh cost and nearest-shape distances.

Selected current coverage at threshold 0.015:

| Category | Representatives / 256 | Median triangles |
| --- | ---: | ---: |
| Floor lamp | 256 | 2,416 |
| Keyboard | 235 | 7,020 |
| Books | 232 | 2,160 |
| Headphones | 213 | 2,624 |
| Mouse | 210 | 1,040 |
| Coat rack | 205 | 3,280 |
| Microphone | 152 | 1,076 |
| Floor outlet | 72 | 668 |
| Water bottle | 67 | 3,104 |
| Rug | 66 | 384 |
| Storage box | 55 | 452 |
| Printer | 49 | 776 |
| Whiteboard | 26 | 646 |

The local matched comparison is retained at
`out/object_diversity/local_comparison_final.json`; it is excluded from the public
snapshot. It reaches at least twice the measured coverage in 29 of 63 valid
comparable categories. The baseline cable fixture was empty and is excluded.
This does **not** establish a 2–4x gain for every category. Tables/chairs already
approach saturation at this sample size. Notice boards, closed notebooks, power
strips and cans retain low coarse-shape counts at fixed dimensions. Near-planar
details can fall below descriptor resolution. These are correlated program
draws, not controlled one-factor interventions; conditional parameters may be
inactive. Coverage is neither SigLIP volume nor downstream training utility.

## Correctness and platform checks

- 360 active indoor tests passed; 23 explicitly ignored tests were not run.
- 48 publication unit/integration tests passed, including malformed atlas data,
  byte/identity binding and camera-family integrity checks.
- Native and WASM Clippy passed with warnings denied; workspace formatting passed.
- All 16,384 object meshes passed finite attributes, normal/winding, index,
  nondegenerate triangle, semantic and reserved-envelope checks. Separate tests
  cover deterministic reconstruction, size extremes, texture coordinates,
  emitter placement, actual contacts and coincident planar faces.
- 512 layouts passed; the first 16 also received full actual-mesh audits.
- 32 consecutive rooms produced 256 calibrated views with RGB, depth, normal,
  position, semantics and exact co-visibility, across two trajectory endpoints.
- All 2,304 camera groups passed (64 rooms, three baselines, counts 1–12).
  Native camera qualification added 312 images; factor qualification added 88.
- All six studio examples per category were rendered with one persistent native
  device. RGB and normal examples are exported losslessly. Selected construction
  plates, the unfiltered room contact sheets, browser controls and new paper
  pages received visual inspection.

Evidence is under `out/object_diversity/`; the current self-contained publication
is bound by `www/project/publication.json`. The primary room, semantic contracts,
render modes and native quality defaults remain intact. WASM was compile/lint
checked for this change; this is not a new browser GPU runtime qualification.
Finite mesh checks do not certify every hidden self-intersection or physical
assembly. Photographic realism remains unproven.

## Performance observations

The opt-in object audit times the actual builder, excluding its depth descriptor.
Representative current medians are 0.252 ms for a keyboard, 0.076 ms for a mouse,
and 0.107 ms for a floor lamp. Triangle counts and p95 build costs are exported
for every category. These fixed-envelope microbenchmarks do not establish a
whole-room speedup or equal cost across the scene distribution.

The native completed-capture run recorded all 128 consecutive rooms (seeds
400–527), with the first 32 treated as warmup. Three 512-by-512 views, six channels
(RGB, depth, position, normal, semantic, co-visibility), human density 0.25,
GPU materials, shadows, SSAO and 1,024-ray / three-bounce GPU GI were retained.
The 96 measured rooms completed at **0.844 rooms/s, 2.533 views/s**; median/p95
capture latency was **1.046 / 2.241 seconds**. File encoding is excluded;
preparation and readback are included. See
`out/object_diversity/throughput_qualified/{summary.json,scenes.jsonl,device.csv}`.

This RTX PRO 6000 Blackwell run shared the device with an independent workload;
aggregate board activity averaged 96.4%. It is a concurrent-workload observation,
not an isolated or matched throughput-regression experiment. Publication work
had finished before the run. All capture records and the summary were written,
but the command wrapper subsequently ended with SIGTERM; a separate eight-room
full-quality process/shutdown probe exited normally in 17.77 seconds. The
earlier 64-room run also exited normally. Do not call the 128-room process a
clean-exit qualification.

Measured process RSS grew from a 5.39 GiB first-quarter mean to 6.67 GiB in the
last quarter (6.90 GiB peak). This short run does not establish a memory plateau
or unlimited-process stability. The expanded geometry passes the structural
checks above; these performance observations leave long-run residency and an
isolated throughput comparison as separate qualification limits.
