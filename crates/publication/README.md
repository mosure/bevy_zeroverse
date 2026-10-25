# bevy_zeroverse_publication

The canonical Rust publication pipeline for the bevy_zeroverse project page and
whitepaper. It is a separate, GPU-free workspace crate. The generator shares its
wire contract, camera colors and Turbo depth palette through `bevy_zeroverse_capture`.

From a Linux bevy_zeroverse checkout:

```sh
cargo run --locked -p bevy_zeroverse_publication -- plan
cargo run --locked -p bevy_zeroverse_publication -- sync
cargo run --locked -p bevy_zeroverse_publication -- verify-current
# Registry publication uses the same current-source gate:
cargo run --locked -p bevy_zeroverse_publication -- publish --package bevy_zeroverse --dry-run
```

`sync` owns the complete order: build native tools, capture the main population,
audit camera counts/baselines, render the camera subset, derive room metrics from
the main captures, qualify fisheye sampling/geometry/lighting, measure preparation
and completed capture, refresh factor/object/door/material atlases, build the page
and paper, and verify/install the complete bundle. It never reads private
before/after studies. `publication.toml` defines bounded populations; the default
is a 512-room structural audit, 32 rendered rooms, 64 camera-audit families at
counts 1–12 and baselines 0/0.5/1, a prespecified two-family rendered camera
subset, and six fisheye rooms. Population sizes and different measurement
resolutions are reported separately. Quality flags are not reduced for CI.

A shared writer lock prevents overlapping publication jobs. Completed stages
are reused only for identical source, version, dependencies, tools and recipe,
with every cached file rehashed. `--recapture` forces fresh captures. Incomplete
stages retain diagnostics and cannot be deployed. Scientific or typesetting
failures restore prepared input evidence; output installation has its own
rollback. An interrupted input transaction leaves an explicit journal under
`out/publication/input-transaction` and blocks the next writer until restored.

`verify-current` is the read-only deployment gate. It needs no renderer, GPU,
LaTeX, Python or `out/` captures. It rejects retained evidence, stale source/tool
identities, incomplete annotations, modified artifacts, invalid membership
masks, inconsistent denominators, broken page links and mismatched PDF sources.
It checks the automatic run receipt as well as the page/paper attestation.

The Pages workflow runs on main pushes, published releases and manual dispatch.
Hosted Ubuntu uses Mesa Vulkan software rendering; the motion-enabled WASM
viewer builds independently in parallel. A second job verifies the exact
SHA-named artifacts before deploying them together. Only current main can
deploy; superseded/failed runs leave the previously deployed site intact. The
paper is produced in this same workflow, not by a separate stale-PDF job.
Software-render timing is explicitly labeled with the actual adapter and must
not be read as GPU training throughput. These workflows activate after the
changes are pushed to main.

Local rendering is the supported alternative: run `sync` on a Linux GPU host,
then include the resulting `www/project`, `tex`, configured `docs/evidence`
directories and `docs/evidence/publication_run.json` in the source revision.
CI still verifies all bytes against that revision; it skips rendering only when
the current-source gate passes. Capture-tool or renderer changes invalidate this
shortcut. A cache hit never relaxes freshness or annotation checks. Required
local tools are Rust, Vulkan, `latexmk`/pdfLaTeX and Poppler (`pdftoppm`); the
checked-in Anny reference supplies static people, without motion-model downloads.

`refresh` remains the lower-level assembly command for already prepared current
studies; it does not generate missing camera/lens/timing prerequisites. `rebuild`
regenerates presentation from an attested bundle. Legacy `verify`/`rebuild` can
read an explicit source-bound `[retained_evidence]` archival policy, but `sync`,
`verify-current`, registry `publish` and automatic deployment reject it. Such
archives cannot masquerade as current measurements.

The paper's capacity section contains analytical occupancy examples, generated
automatically by `refresh`/`rebuild` from tested known probability laws. To inspect
these formulas independently, without rendering scenes:

```sh
cargo run --locked -p bevy_zeroverse_publication -- capacity-theory --output out/capacity_theory
```

This command does **not** estimate the generator's population capacity. Its
receipt declares `generator_measurements: false`. The implemented configuration-bound
capacity API and empirical qualification protocol are described in
[`docs/procedural_capacity_plan.md`](../../docs/procedural_capacity_plan.md);
they are not implemented by the analytical illustration command.

Use `capacity-qualify --plan PLAN --records JSONL --output RECEIPT` to analyze a
complete explicit audit, then `capacity-report --receipt RECEIPT --output DIR`
for a local current-state HTML report, SVG/PNG figure and paper table. See
[`docs/capacity.md`](../../docs/capacity.md) for the generator adapter and native
capture workflow. These commands do not publish or start training.

`capacity-embedding-index` freezes the complete rendered cohort, cached-model
weight hash and kernel bandwidth before optional `indoor_embed` inference.
`capacity-embedding-report` verifies those bindings and generates a room-weighted
RBF spectrum, Vendi/participation scores and nearest-pair gallery. These are
sample diagnostics, not population-capacity estimates. The exact kernel has a
512-room limit and never silently subsamples an oversized cohort.

`capacity-paper --receipt RECEIPT... --output DIR` builds local source-checked
paper sections and figures. For the canonical publication pipeline, optionally
set `capacity_receipts = ["docs/evidence/your-receipt.json"]` in the recipe (up to
four separately labeled populations). Refresh rejects stale generator identities
before capture and binds receipt/artifact hashes into the attestation. Removing
the configured receipts removes the generated section and figures. No evidence
is installed or published merely by running the local capacity commands.

The recipe is `publication.toml`; authored templates are in this crate's
`templates/`. Measured results and selected examples come from the dataset,
not manually copied numeric prose. Figure rendering uses bundled fonts, native
Rust image codecs and SVG rasterization. Previews use lossless image encoding;
depth previews use display-only Turbo colorization of the existing 8-bit axial
metres/15 PNGs, on a fixed logarithmic 0–15 m scale. Scalar capture files and
CLI/data-generation defaults stay unchanged.
Membership remains an exact 16-bit mask with separate 8-bit 0/1 validity.
The page and paper describe the current generator and its measured cohorts.
Every cohort must match the attested capture identity. In the default policy
that identity must also match the compiled generator. Historical studies,
release comparisons and proposed implementation plans stay outside this bundle.
Camera-count and baseline studies use the same identity gate as the room gallery.

The object atlas audits all 100 furnishing/prop categories on 256 construction
seeds each at fixed metric envelopes. The Rust refresh command builds
`audit_objects` and `review_objects`, checks the real triangles, selects six
depth-map-spread examples per category and captures them with one persistent
native renderer. The shipped atlas binds per-seed programs, three coverage
thresholds, nearest-shape distances, mesh counts, build timings and lossless
RGB/normal examples. Category data loads on demand in the project page.
Generator and qualification-tool hashes gate cache reuse; changed geometry
cannot reuse old studio images. The paper is generated from the same atlas.

The project page has one geometry explorer for objects and architectural doors.
Cabinet categories expose geometry variants and opening/hardware studies in that
same region. Channels and semantic legends follow the selected study; the 256-seed
shape audit and 64-seed assembly checks keep separate denominators.

For an explicit local audit:

```sh
cargo run --example audit_objects -- --samples 256 --output out/objects/audit
cargo run --example review_objects -- --audit out/objects/audit/audit.json --output out/objects/renders
cargo run -p bevy_zeroverse_publication -- object-study \
  --audit out/objects/audit/audit.json --renders out/objects/renders --output out/objects/atlas
```

These descriptors measure fixed-envelope surface variation, not independent
parameter counts, perceptual embedding volume or downstream training gains.
Displayed examples are selected for spread; all 256 samples remain in the
statistics. Near-planar content and subpixel details can remain indistinguishable.

The material atlas uses the same pipeline. It covers all 36 architectural/object
surface roles, with three floor types and a dedicated rug program giving 39 rows.
Each records 256 consecutive recipe seeds. Six shared illustration seeds generate 234 native
PBR swatches, matched rendered normals, and level-zero base-color, tangent-normal
and roughness maps. Seed four deliberately includes frosted interior glass;
the selected grid is not a frequency estimate. Every handle comes from the
production material factory, including finish-slot selection, texture structure,
tint, metric UVs and optical properties. The swatches use the CPU reference
map baker and a persistent native renderer; this is not a GPU throughput test.

A separate 256-room audit builds actual primary-room objects and checks seeded
finish assignments. Object-category/surface/slot histograms show where each
material is used. Humans, architecture and neighboring objects are outside this
object-assignment denominator. The page exposes conditional parameter marginals
and the paper includes every material row. Atlas verification rejects missing
roles, inconsistent seed bindings, stale compiled tool sources, changed images
and incorrect summary denominators. Nothing is added to the runtime generator.

```sh
cargo run --example qualify_materials -- --output out/materials/raw
cargo run -p bevy_zeroverse_publication -- material-study \
  --input out/materials/raw --output out/materials/atlas
```

`refresh` automatically creates/reuses `out/publication/materials` using generator
and compiled study-tool hashes. `rebuild` and CI verify the self-contained atlas
without rerunning the GPU study. Recipes, maps and bindings describe a correlated
procedural space, not a product of independent parameters or a measured semantic
embedding volume. See [the material protocol](../../docs/material_qualification.md)
for validation scope and interpretation.

Refresh/rebuild additionally require `latexmk`, `pdflatex` (including the usual
LaTeX packages) and Poppler's `pdftoppm`. LaTeX is the document typesetter; Rust
owns source generation, dependency closure, warnings, provenance and staging.
The publisher does not estimate photographic realism or downstream utility.

An explicit retained-evidence release can pin updated capture tools in
`retained_evidence.current_tool_sha256`. Only registered capture-tool paths are
accepted. Their current bytes must match the pin; the original tool hashes,
measurements, images and paper stay bound to the original capture attestation.
This does not qualify the updated tool or generator using older captures.

`publish --package <name>` prepares the latest page/paper before calling Cargo.
Under the default policy it refreshes stale inputs or rebuilds an already-current shipped bundle, verifies
the installed result, then invokes `cargo publish --locked`. `--dry-run` never
uploads and permits local dirty-checkout packaging; actual uploads retain Cargo's
clean-checkout requirement. Publish the shared capture crate before dependent
packages. Direct external `cargo publish` invocations cannot be intercepted; use
this entry point for repository releases.

`www/project/publication.json` is the commit marker and machine-readable
attestation. It binds renderer inputs, publisher/templates, capture sources,
current cohort identities and every managed page/paper artifact. A crash during
installation cannot yield a partially installed bundle that passes this gate.
Tests cover the failure modes that previously allowed mixed gallery cohorts.

## Paired room studies

`compare-indoor` qualifies two complete, explicitly identified cohorts and builds
an independent local before/after gallery without replacing the project page or paper.
Study outputs must stay below ignored `out/`, outside `out/publication/`.
Reports and receipts carry `visibility = local_only`; the release pipeline
rejects them as snapshot inputs. They are not deployment artifacts.
The default protocol is 64 consecutive rooms, three 512×512 views, one timestep,
furnishing density 0.65, human density 0.25 and 1,024 diffuse-GI rays.

```sh
cargo run --locked -p bevy_zeroverse_publication -- compare-indoor \
  --before out/study/before --after out/study/after --output out/study/report \
  --before-population out/study/before-population \
  --after-population out/study/after-population
cargo run --locked -p bevy_zeroverse_publication -- verify-study \
  --output out/study/report
```

Every seed, camera, RGB/annotation preview, exact co-visibility membership/validity
mask, schematic, actual-mesh audit and input identity is retained and checked.
Primary architecture, base lighting samples and camera controls must match.
Updated viewpoint priors, added fixture illumination and collision-safe camera
replanning are disclosed; the views are not assumed pixel-matched.
Optional population inputs must match each renderer's compiled identity and
controls. No sample is dropped because it is sparse, dark or has a close neighbor.

The first invocation produces `embedding-index.json`. Run the optional native
`indoor_embed` tool against it, then repeat `compare-indoor` with
`--embeddings out/study/embeddings` to bind the exact tensor and model checksums.
Reports include cross-room nearest pairs, equal-weight room-centroid variance,
and paired-room bootstrap intervals. Seed uniqueness and products of marginal
entropies are not estimates of perceptual procedural-space size.

Outputs are staged and installed atomically. Failed qualification preserves the
previous report, and `verify-study` detects modified or missing gallery bytes.
Study generation supports 2–256 rendered rooms and a single timestep; use
`--recipe protocol.toml` for a smaller qualification fixture or a larger cohort.


## Calibrated fisheye evidence

The lens gallery and lens chapter have their own current-source receipt. They
extend the project page and downloadable whitepaper without relabeling the
architectural cohort's capture provenance. Both paper components remain bound
by artifact hashes. Nothing is deployed by these commands.

```sh
cargo run --example validate_fisheye -- --cpu --output=out/fisheye/current
cargo run --example validate_fisheye -- --fixture
cargo run --example validate_fisheye -- --fixture --narrow
cargo run --example validate_fisheye -- --bench --rooms=6 --seed=400 --output=out/fisheye/current
cargo run -p bevy_zeroverse_publication -- fisheye-study --captures out/fisheye/current --analytic out/fisheye/analytic.json
cargo run -p bevy_zeroverse_publication -- verify
```

The qualification recipe requires all six consecutive rooms, three views and two
timestamps at 512×512, with full-quality RGB and aligned annotation previews.
Set `--isolated` on the capture command only when competing GPU training has
actually been paused. Runs outside this recipe remain local diagnostics. Renew
the source-bound retained-evidence policy deliberately when generator source
changes; `rebuild` refuses stale inputs and never silently recaptures or rewrites
an old measurement.

At the canonical evidence destination, `fisheye-study` invokes the transactional
page/paper rebuild: the installed gallery and its publication attestation change
together. A previous chapter must still pass its recorded hashes before renewal;
the replacement must match the current generator. A custom `--output` prepares
evidence only. `rebuild` can refresh presentation from existing qualified inputs.

`fisheye-performance --input <benchmark-directory> --output <capture-directory>`
verifies the `current`, `pinhole`, `rgb-current` and `rgb-pinhole` subdirectories
before producing a matched throughput report. Each uses six rooms starting at
400 and three 512×512 views; the first two use all seven modes and two timestamps,
the RGB pair uses `--rgb-only --steps=1`. Pinhole controls require
`--pinhole --matched`, and all four require `--isolated` only after competing GPU
work has actually paused. This checks exact scene manifests and camera poses,
not merely seed equality. It records first-room latency separately.

The optional `lighting.json` input comes from `validate_fisheye
--lighting-fixture`. Its final production-WGSL composite continuity checks must
pass; raw face-boundary failures remain explicitly failed diagnostics in the
receipt and paper. The study receipt binds both optional reports, and the
generated lens chapter includes their measurements and limitations.
