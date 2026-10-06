# bevy_zeroverse 0.31

Core and FFI **0.31.1**, with Burn wrapper **0.14.1**, include a seeded furnishing
repair on top of the qualified full-quality capture and CPU-preparation optimizations. Capture contract **0.1.1**,
publication tool **0.1.2** and SigLIP2 **0.1.1** are unchanged. The renderer contract
is capture-v48; the scene generator remains 29.

Map-owned field caches, exact material arithmetic reuse, bounded native atlas
parallelism and overlapping material/transport/mesh preparation reduce repeated
CPU work. Ground-truth extraction assembles indices directly and reuses exact
unchanged instance matrices. Cached readiness ID lists still check live GPU
assets and pipelines every frame. Shared capture arrays and combined native
attachment mapping reduce allocation and mapping overhead. The normal/GGX mip
contract, geometric detail, shadows, GI and requested annotations are retained.
GPU material baking is not implemented.

The ordinary CLI and LiveDataset paths retain automatic bounded preparation;
consumers need no new scheduling controls. Position planes use float32 under
capture-v48, with explicit precision-aware decoding, sensor transforms and
full-pixel annotation checks. O-voxel pipelines and pooled buffers have explicit
renderer/device ownership; primary-room geometry and annotation bounds remain
unchanged.

The release audit also exposed rare exhausted camera searches in sparse,
concave rooms and at full human occupancy. Bounded recovery searches run only
after the original proposals fail, retaining the same clearance, visibility,
trajectory length and multi-view overlap requirements. Previously successful
camera proposals and their random sequence remain unchanged.
Recovery retains up to four spatially diverse partial groups. Trapped pairs use
body-relative reference framings, ceiling-relative views and two frontiers of at
most four accepted prefixes, plus a transient incoming prefix and working vectors.
Storage scales with the requested camera count. Recovery first tries to finish
retained pairs before replacing their reference. Twenty-four nearby proposal slots
vary lateral spread, path length and lens framing within the authored bounds. Each
candidate-parent attempt, including reused paths, consumes the existing budget;
reference probes and the intact-pair branch are charged against it. A retained group's total is at most 384 times the number of nonreference
views. Explicit overlap-mixture strata are never redrawn or relaxed.

[Throughput qualification](generation_cpu_efficiency.md) records twelve matched
local runs, 896 room captures, and exact replay of 908 RGB/annotation/metadata
files. Its incremental improvements are measured against the preceding accepted
local optimization snapshot, not the last published crate. It explicitly reports
flat pool throughput, moving-transform overhead, and the retained ancestor RGB
exception. Local WGPU patches are excluded from registry packages; published
packages are separately qualified against registry WGPU. The requested 2× gain,
unlimited-process stability and downstream training benefit remain unproven.

[Release qualification](evidence/release_031/README.md) binds 13 local and
five registry-only gates to the frozen optimized source: 40,000 density configurations,
4,096 full-occupancy configurations, recovered-camera RGB and annotation
captures, native GPU fixtures, motion-enabled WASM compilation and strict
Clippy. The normalized core archive passed unit, GPU and Burn temporal tests
with registry WGPU 29.0.4. These counts are configurations, with seeds repeated
across density settings; they do not imply 44,096 distinct seeds.

[The portability bridge](evidence/release_031/ci_portability/README.md) records
the final test-only correction found by macOS CI. Arithmetic replay checks NaN
classification, while all non-NaN results and bit copies remain exact. The bridge
proves unchanged production inputs, preserves the earlier runtime qualification,
and records a fresh 368-test core suite, formatting and strict workspace Clippy.

Crate archives exclude model assets. Occupied scenes require the deployed
`assets/burn_human` fixture: pass its parent directory with `--asset-root` to
`zeroverse_gen`, or set `LiveDatasetConfig.asset_root`. Direct library tests use
`BEVY_ASSET_ROOT`; packaged-source qualification records the external fixture
checksums separately from archive contents.

The canonical Rust publisher regenerates the page, paper and qualification
cohorts from the final release source. Historical camera reference studies retain
their original identities. Regenerated media does not establish photographic
realism or ten-million-sample learning utility.

## Rust compatibility

PreparationTimings and AnnotationAlignment gain public diagnostic fields.
Exhaustive Rust struct literals must initialize them; use Default where available.
Both wrappers require core 0.31.1. This minor release makes those source-level
changes explicit. Existing material recipes, camera programs and numeric semantic,
flow and co-visibility contracts retain their behavior. Capture identities and
position precision distinguish the new export contract from older captures.

## Patch 0.31.1

Seed `43218880` could exhaust workstation repairs after accepting lounge seating
from a mixed-activity program, leaving no primary activity surface. A bounded
sofa-relative coffee-table recovery now completes existing seating groups only
after ordinary repair exhausts. Floor support, portal, ceiling, pillar and
collision checks are retained. Ordinary successful scenes and their random
streams remain unchanged.

Regression coverage includes deterministic replay, real mesh validation, three
cameras, all twelve combinations of furniture density `0/.35/.65/1` and human
density `0/.25/1`, and unchanged complete-room recovery. Native qualification
checks six full-quality 512×512 views across two timesteps, with RGB, depth,
normal, position, semantic and co-visibility outputs. A 256-seed neighborhood
audit retains the same placement and camera acceptance thresholds.

[Patch qualification](evidence/release_0311/README.md) binds the seed capture
and neighborhood measurements to the 0.31.1 generator identity.
