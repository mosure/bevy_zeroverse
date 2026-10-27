# Cabinet and clock construction qualification

General storage and wall clocks use seed-stable construction programs with
independent finish streams. This removes repeated proportions and placement
priors without treating necessary product relationships as defects.

## Cabinet contracts

- Continuous compartment widths and shelf/drawer intervals; some products retain
  aligned shelves. Mixed fronts select their drawer bay independently of parity.
- Coherent or contrasting timber, metal and polymer/laminate substrates for
  frame, shelves and fronts. Finish changes preserve geometry and operating poses.
- Sampled base height, hardware position/span and slat spacing. Handles attach
  to the door skin; sliding tracks remain ahead of internal dividers and shelves.
- The widest actual door determines the static opening reserve. Rigid joint
  construction uses that same rest geometry and hinge. Existing clearance
  planning continues to qualify optional motion.
- A full horizontal top supports the declared cabinet footprint. Uprights end
  below it, keeping competing top faces separate.
- Open bays reuse the bookcase's bounded contents program. Books, stacks,
  storage boxes and vases fit the actual bay, rest on its shelf, and use varied
  supported transforms. The obsolete parallel bay-content implementation is removed.

The implementation is split into `storage/program.rs`, `cabinet.rs` and
`fronts.rs`; `bookcase/contents.rs` provides the shared content construction.

## Clock contracts

The wall sampler includes 0.20–0.46 m heights, square or 0.80–2.35 aspect ratios,
and 0.032–0.080 m depth reservations. Wide products favor digital faces; analog
products keep time-consistent hour, minute and optional second hands. Taper,
corner roundness, bezel, hand proportions, index density and digital typography
vary. Digital strokes fit the actual rounded dial, including corner clearance.

The rear case plane is shared with wall-mount placement. Casing/dial materials
use the independent finish identity instead of integer seed modulo. Existing
material atlases and semantic classes are reused.

## Measurement protocol

The ignored `storage_clock_regularity_population` test records 64 fixed-envelope,
fixed-finish meshes per category. Shape coverage uses six 24×24 first-surface
depth maps and union-pixel normalized L1 distances at 0.005, 0.015 and 0.03.
It excludes colors, triangle ordering and mesh tessellation from the diversity
measurement. The same study validates 80 complete rooms across all 20 concrete
profiles, with three cameras, density 0.65 and human density 0.15.

Routine tests cover independent metric extremes, actual top-support ray hits,
contents containment/separation/contact, accepted loose-content position/yaw,
material/geometry independence, wall attachment and dial support. Motion,
semantic, coplanar-face, catalog, camera-placement and rejection regressions are
checked separately. Construction parameters and accepted physical dimensions are
available in the audit report; dataset metrics include cabinet bay width and
clear height, substrates, clock aspect, roundness and case depth.

Results are local under `out/storage_clock_regularity`. The gallery is CPU flat
shading of actual meshes. Timings describe bounded CPU preparation on the current
machine, not GPU rendering or training throughput. Shape coverage is not a count
of independent training samples; downstream generalization and fresh PBR/browser
runtime behavior require separate qualification.

## Recorded local results

Generator 91 / capture-v116 passed 51 distinct native tests, strict workspace
Clippy (CUDA IPC/WGPU fusion and SigLIP CLI/pipeline enabled), formatting and the
WASM viewer compilation with `web,human_motion`. The tests include 384 cabinet
and 384 clock meshes at independent metric extremes, full-catalog mesh checks,
31-category fixed-envelope coverage, cabinet motion, coplanar-face detection,
semantic vocabulary, historical rejected rooms and metric export.

| Fixed-envelope category, 64 seeds | Shape representatives at 0.005 / 0.015 / 0.03 | CPU build median / p95 | Triangle median / maximum |
|---|---:|---:|---:|
| Cabinet | 51 / 40 / 27 | 0.065 / 0.279 ms | 2,228 / 19,980 |
| Clock | 63 / 62 / 57 | 0.168 / 0.183 ms | 5,720 / 5,800 |

All 80 complete rooms (seeds 0–3 for each of 20 profiles) passed layout and mesh
validation. With three cameras, CPU manifest preparation measured 25.7 ms median,
171.5 ms p95 and 833.1 ms maximum. This is a bounded coverage cohort on a shared
machine, not a matched throughput comparison.

The accepted non-neighbor objects include 13 cabinets and 36 clocks. Their
observed cabinet bay widths span 0.258–0.801 m; clear bay heights span
0.191–0.476 m. Five frame/shelf/front substrate combinations are present. Clock
aspect ratios span 0.815–2.246, with reserved depths 0.0367–0.0799 m. Because the
same four base seeds are reused across profiles, these observations are
correlated; their small counts do not qualify global style probabilities.

The local `qualification.json` records source and test-binary hashes, exact
commands, counts, timing summaries and limitations. `report.json` retains each
sample's construction program and accepted transform. `index.html` provides
the geometry gallery with its sampled parameters. No study artifacts are
automatically added to the published page or paper.
