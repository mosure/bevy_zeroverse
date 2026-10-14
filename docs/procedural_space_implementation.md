# Procedural space implementation and qualification

This is the current execution record for the [roadmap](procedural_space_roadmap.md).
The working source identity is
`1aab7c4b05a950f005240c75b216e55777e32359fb23c2ebd2636e597d36e4b5`:
generator 41, metrics schema 19, capture engine v62, semantic taxonomy 4.
Crate versions remain 0.40.0 / 0.24.0. These local changes have not been committed,
pushed or published. The starting checkout is
`b938bb277cab0effb8e1c5fc60d209aeb097a6ae`.

The implementation and qualification work covers all twelve execution phases.
That does **not** mean every promotion gate passed. Physical transport accuracy,
long-run host-memory stability, a 2× completed-throughput gain and improved
reconstruction learning remain unmet or unproven. The local matched embedding
study does not establish a perceptual-volume increase.

## Delivery and evidence boundaries

| Phase | Implemented or exercised | Result and limit |
| --- | --- | --- |
| 1. Unified qualification | Atomic running/complete/failed capture receipts, requested settings and source identity; current-only Rust cohort aggregation and strict membership/graph validation | Failed requests and distinct sampling populations remain visible; old or incomplete captures cannot enter the current publication |
| 2. Architecture and functional assemblies | Shared envelope/topology, closed soffits and local headroom, offset returns, portal/window systems, transactional meeting assemblies, support and operational-clearance contracts | 10,000 actual constructed rooms pass; this is a bounded connected-polygon grammar, not arbitrary architectural CAD |
| 3. Objects and use states | Open notebooks with supported curved pages, fitted speaker cones/surrounds, bounded noticeboard details; conditional object-program audits | Adversarial dimensions and deterministic mesh tests pass; dormant parameters no longer count as realized object variation |
| 4. Materials and filtering | GPU paint/accent/ceiling and neutral knit recipes; independently varied metal/chrome/plastic structures; controlled detail scaling; coupled normal/roughness mips | 135 CPU/GPU material recipes pass all-map/mip comparisons; material realism remains view/illumination dependent |
| 5. Lighting, glass and reflections | GPU cosine/GGX environment convolution with CPU oracle, bounded spatial native probes, explicit unrefracted glass annotation contracts; calibrated Cycles comparison | Convolution parity passes; complete-room transport agreement fails, particularly for direct illumination through glazing |
| 6. Humans, motion and context | Anatomically partitioned eye surfaces; current fitted clothing/hair, botanical/exterior programs; native/browser ARDY and collision-fallback checks | Motion and aligned annotations work; 7/10 selected actors receive admitted motion. Humans remain visibly stylized |
| 7. Reconstruction cameras | Optional connected overlap graph, bounded admission/fallback diagnostics, primary-region rendered graphs, metric baselines and triangulation angles | 7,103/7,104 requested CPU camera sets admitted; 312 camera-count/baseline render views pass. Default reference policy remains unchanged |
| 8. Camera/sensor contracts | Seedable off-center pinhole projection, exact K/frustum/subviews, calibration/flow/visibility/reference integration; UI/URL round trips | Projection and both glass-policy oracles pass. Jitter defaults to zero; distorted optics, rolling shutter and ambiguous multi-time pixels remain deferred |
| 9. Useful diversity | Seed-matched 64-room SigLIP2 study, fixed pooled basis and room bootstrap; structural/object/material distributions | No significant measured volume gain. No downstream reconstruction improvement is claimed; active external training is preserved |
| 10. Preparation critical path | GPU map and environment filtering, bounded CPU preparation and render residency; profiling separated from completed capture | Parallel preparation is useful; an isolated 2× end-to-end gain was not demonstrated under shared-device contention |
| 11. Actual consumers and web | Completed WGPU/CUDA tensor parity, retained sample lifetime, indexed replay, same-device optimizer probes; hardware Chrome static/motion/legacy scenes | No image roundtrip is required on GPU routes. Browser long tasks and host-memory growth remain explicit limitations |
| 12. Larger audit and publication | 10,000 manifests, 10,000 mesh constructions, matched 64, current-only 512-room campaign, 1,000-room stability run, current page/paper regeneration | Finite correctness checks pass where stated below; the memory-plateau gate fails |

## Source, protocol and measurement conditions

Final-source evidence lives under `out/development_041/validation/`.
Earlier intermediate results are retained under separate directories and do not
qualify the final source. The 64-room comparison, embedding neighbors and its
before/after viewer stay local. The distributable page and paper contain only
current-source captures and instantaneous metrics, using the Rust publication
pipeline and its atomic attestation.

Hardware: NVIDIA RTX PRO 6000 Blackwell, 96 GB; Vulkan driver 610.43.02;
Intel Core i9-12900K, 24 logical CPUs. The build uses Rust nightly-2026-08-31,
Bevy 0.20, Burn 0.22 and registry WGPU 30.0.1. No vendor patch or alternate
allocator is a production dependency.

An independent 250k-step CUDA training job and four downstream JIT capture
workers initially occupied most of the GPU. That workload dropped off during
the earlier timing series, without intervention by this task; its training
completion status was not assessed. Subsequent consumer probes ran at low
background occupancy and include their own small optimizer workload. Per-run
board telemetry records the distinction. CPU preparation,
completed room capture, tensor transport and optimizer completion are separate
measurements. The controlled reference and matched studies are finite diagnostics,
not estimates of ten-million-sample learning utility.

## Geometry, materials and annotation contracts

The exact primary envelope, floor/roof features and local soffit heights feed
construction, camera clearance, ray queries and O-voxel membership. The annotated
primary reconstruction AABB excludes exterior backdrop geometry. Offset returns
can occupy multiple eligible side edges while retaining the shared neighbor wall
and at least 78% of the original footprint area. Footprints remain simple connected
polygons without interior holes; neighbor rooms and mezzanine decks remain bounded
rectangular assemblies.

A regression discovered in the actual construction audit, seed 578, placed a
short/wide speaker surround below its housing. Cone/surround radii now jointly
respect width, height, edge clearance and pair separation. The regression and
256 seeds at three adversarial aspect ratios pass. Notebook page curvature,
opening, cover thickness and ink remain within supported bounds. Recessed
noticeboard notes and divider checks cover extreme dimensions. These fixes retain
existing semantic ownership and collision requirements rather than relaxing them.

The eye fix partitions the existing Anny eye triangles into exclusive sclera,
iris and pupil regions. It preserves triangle winding, area, UVs and interpolated
normals instead of layering coincident ellipsoids over the eyeball. This repairs
ownership and a pale iris ring; it is not a claim of photographic human rendering.

GPU material recipes preserve CPU reference evaluation, linear/sRGB conventions,
all mip levels and joint normal/roughness filtering. Chrome avoids quantized
micron-scale normal perturbations; polish and brushing remain scattering controls.
The detail factor affects relief on the supported substrates without changing
geometry labels. Default arithmetic and independent RNG domains retain replay.

Static HDR radiance tracing remains CPU work. Its cosine and GGX convolutions
execute once on the GPU and are readiness-gated before capture. Native rendering
has at most three box-projected zone probes plus the global environment. WASM
uses the global map because localized probes require texture-array capabilities.
The bounded cache includes cancellation/error cleanup and does not retain every
room's textures. It is not dynamic reflection of moving humans.

The geometry annotation path uses float32 positions/depth and consistent semantic
ownership. `surface` treats glass as an annotated surface; `through` skips clear
or transmissive panes while retaining frames and opaque frit. Through-glass depth,
normal, position, semantic, flow and co-visibility describe straight geometric
rays. They are not exact inverse correspondences for refracted or diffusely
frosted RGB. Sensor-only appearance effects do not modify geometric attachments.

## Reconstruction camera and overlap qualification

`crates/capture/src/reconstruction.rs` derives a versioned view graph from integer
visibility counts. Validation checks denominators, directionality, graph edges,
components and overflow. Native qualification exports each camera/time graph
for the complete frame and for the exact primary reconstruction region separately.
The primary mask uses native float32 positions, the polygon and the local roof.
Per-pair output includes metric baseline, shared-pixel count and pixel-weighted
triangulation P05/P50/P95 using 0.5-degree bins. A pair without shared geometry has
null angle statistics, not an artificial zero.

The optional `connected` placement policy admits stable peer connections across
five trajectory times. It permits a chain of overlapping views without requiring
every camera to overlap the same reference view. A bounded search retains all
collision, path, visibility and baseline predicates. Any stricter reference-policy
recovery is explicitly counted. Default reference-connected behavior is unchanged.

Off-center projection is an actual render projection, including view frusta and
subviews, not an annotation-only K edit. The optional principal-point jitter is
seeded independently and bounded to 0.2 image fractions per axis. Jitter zero
preserves the centered-pinhole contract. Existing downstream consumers that assume
centered K must opt in only after their own camera adapters support it.

The CPU audit covers 7,104 requests across 1–12 cameras, multiple baselines and
aspects 0.625, 1 and 1.6. There is one recorded rejection: reference policy,
portrait aspect, seed 4, baseline 1, twelve cameras. The connected-policy cohorts
also use principal-point jitter, so their acceptance is not a controlled isolated
comparison against the reference cohorts. A separate native study renders 312
views across counts 2/4/8/12 and baselines 0/0.5/1 with two rooms and two times.
Current schematics, pose distributions and co-visibility figures are generated
from those exact captures.

## Current rendered cohorts

The current-only campaign uses full Auto quality, 256 GI rays, three 512×512
cameras, two times, RGB/depth/normal/position/semantic and numeric co-visibility:
256 consecutive rooms, 128 prespecified factor-stratified rooms from an independent
2,048-room population, and 128 held-out consecutive rooms. These are 512 distinct
rooms / 3,072 views. They are not pooled into an IID estimate: selected strata and
consecutive populations have separate denominators and receipts.

The publication recipe additionally retains its 32-room, four-camera, two-time
640×400 explorer with 1,024 GI rays, aligned annotations and one integrated
co-visibility control for every room. Its 512-room manifest population is a
separate structural sample, not 512 rendered rooms. The larger rendered-cohort
receipt names its own protocol. Camera-count, object-program, material-PBR and
factor-control galleries are distinct current-source diagnostics.

The Rust `cohort-study` path verifies completed source identities, every delivered
image/mask hash, seed uniqueness, view/time denominators, numeric reconstruction
graphs and null-angle semantics before emitting compact publication evidence.

| Rendered cohort | Rooms | Minimum semantic classes/view | Valid source pixels shared with any peer | Primary-human rooms with person pixels |
| --- | ---: | ---: | ---: | ---: |
| Consecutive 0–255 | 256 | 5 | 74.69% | 172/172 |
| Prespecified strata from 20,000–22,047 | 128 | 4 | 73.75% | 78/78 |
| Held-out 40,000–40,127 | 128 | 6 | 74.54% | 90/90 |

All 1,024 camera/time groups have connected primary-region graphs at 10%
minimum bidirectional overlap, with no empty primary views or zero-shared pairs.
Median pair baselines by cohort are 1.95 / 1.89 / 1.85 m; median pair-median
triangulation angles are 21.25 / 22.25 / 21.25 degrees. The worst per-view P99
reprojection error is 0.000713 pixels; worst P99 depth/position disagreement is
3.82e-6 m. Person visibility means at least one person somewhere in a room's
six views, not visibility of every actor. Other envelope zones and the neighboring
room may contain unseen people; those counts remain separate in the receipt.

Only the current-source receipt enters `publication.toml`; the matched comparison
cannot be copied into the deployable cache by that command.

## Physical transport reference

Complete-room native exports and Cycles references share geometry, materials,
lights, exposure and off-center cameras. References use Blender 5.2.2, 1,024
samples, twelve bounces, no denoiser and an independent repeated seed. A calibrated
radiometry fixture and export-contract tests qualify the comparison. Geometry
ray checks show no hit mismatch and submillimetre P99 depth disagreement. Reused
reference images are permitted only after exact optical-input hashes match.

| Room | Native/reference mean linear luminance | Relative luminance MAE | Median absolute exposure difference |
| --- | ---: | ---: | ---: |
| Seed 3, sunlight fixture | 2.882 | 2.091 | 0.855 stops |
| Seed 6 | 1.050 | 0.270 | 0.798 stops |

Independent reference-repeat block errors are approximately 0.00250 and 0.00133;
reference mean-luminance ratios are 0.9996 and 1.0004. Reference sampling noise
therefore does not explain the observed transport discrepancy. Glazing is omitted
from the approximate diffuse transport and does not cast the corresponding tinted
or frosted direct-light shadow. Local reflections do not fix that energy path.
The physical-accuracy promotion gate **fails**. A global exposure adjustment was
not used to hide scene-dependent error, and photographic realism is not claimed.

## Human motion and browser qualification

Native ARDY qualification uses the upstream cached model loaders, 60 frames,
40 history frames, ten diffusion steps, batch two and two attempts per selected
actor. Seeds 3/6 are fixed diagnostic rooms; seeds 15/16/27 are the first eligible
backward-walk, side-step and skip selections from the planning population. This
selection tests behavior coverage rather than estimating a random acceptance rate.

Seven of ten selected actors receive admitted motion. One actor has no fitting
plan; two generated plans are rejected for collisions, including a skip with
2.7 cm floor penetration. Rejected actors keep static geometry. All five rooms
complete calibrated two-view/five-time motion and annotation checks. There is one
model load across all five rooms and seven generation batches. An earlier failed
run is retained: a local mirror lacked lazily fetched vocabulary pages. The
complete retry uses the official CDN/cache, not a changed prompt set or weakened
collision tolerance.

Hardware Chrome tests cover every indoor render mode, both new camera controls,
eight scene regenerations with URL-seed updates, optional motion over two scenes,
and CornellCube/Object/Room/SemanticRoom. GPU validation reports no errors. The
static browser path makes zero motion-model requests. Regeneration reuses the
same inference device and model manifests. At fixed cameras, the static frame
pair changes zero pixels; enabled motion changes 280 pixels above 4/255, a
visibility check rather than a pose-quality score.

Browser regeneration takes 6.4–12.3 seconds after startup in the contended run;
individual long tasks reach 442–918 ms. Cooperative scheduling and successful
rendering do not establish interactive frame-budget compliance. Browser preparation
remains cooperative work on the browser thread, not parallel worker execution.

## Diversity, performance and stability limits

The local paired study embeds 384 RGB images per cohort with cached, shard-verified
SigLIP2 base-patch16-224. Room-weighted cross-room nearest cosine distance changes
by −0.00000988; its 95% paired-room bootstrap interval is
[−0.000362, 0.000312]. Pooled-PCA volume ratios at ranks 4/8/16 are approximately
1.006 / 0.992 / 0.997, with intervals including one. This is neither a significant
volume increase nor a ten-million-sample non-repeat guarantee. Whole-view embeddings
may also miss localized eye or notebook improvements. Better visual details and
more configurable parameters must not be reported as demonstrated training gains.

Current CPU preparation of 32 rooms after four warmups measures 1.52 prepared
rooms/s with one in flight and 3.32 with four. A sequential prepare averages
654.8 ms: about 61.4% is the parallel material/transport/mesh stage, 16.2% static
HDR radiance tracing and 10.4% camera generation. The paper includes these relative
critical-path timings and a pipeline diagram. Summed parallel task times are not
presented as elapsed wall time.

Matched completed capture timings vary substantially as independent training
contention changes. They do not establish a 2× generation gain. GPU parity tests
verify WGPU and Vulkan/CUDA transport, retained sample lifetime, allocation sharing,
indexed deterministic replay, both glass policies, exposure isolation and exact
CPU/GPU-material geometry annotations. Same-device optimizer probes measure actual
backward/Adam completion separately from tensor delivery; they are not downstream
reconstruction-model experiments.

The isolated 128-iteration, three-view tensor probe measures 91.4 iterations/s
with GPU→CPU→GPU image traffic and 1,617.9 iterations/s retaining GPU tensors on
the same device/queue. Both complete the same scalar reduction. The latter avoids
24 MiB of host image traffic per iteration; the approximately 17.7× ratio applies
only to tensor transport/consumption, excluding rendering and scene preparation.

Completed 32-room probes at three 512×512 views measure 0.915 rooms/s for CPU
transport, 0.936 for shared WGPU and 1.708 for Vulkan/CUDA. These sequential probes
are integration timings, not an isolated 2× improvement. With the same-device
Adam/backward consumer, CPU material synthesis measures 1.532 and 1.550 rooms/s
in the outer ABBA runs, versus 0.969 and 0.969 for GPU materials in the inner runs.
Each run includes 96 measured optimizer steps plus warmup. This measured GPU
material configuration is slower; correct GPU baking is not by itself a
throughput improvement. Its shader scheduling/readiness costs remain an
optimization target. No claim of concurrent large-model training follows from
these later probes: the earlier external workload had already dropped off.

The 1,000-room, three-view 512×512 sustained capture finishes with no capture
failures and a stable 98 compiled render pipelines. Observed asset/GPU cache
counts are bounded. **Host-memory plateau nevertheless fails:** post-warmup RSS
has a 4.10 MiB/room fitted slope; even the final 250-room window has 1.61 MiB/room
RSS and 1.84 MiB/room live-heap growth. Peak RSS is 9.84 GiB.

A separate 256-room jemalloc allocation-profile diagnostic attributes most growing
live allocation to NVIDIA/Vulkan command-buffer construction. Estimated driver
allocation rises from roughly 1,336 MiB after 27 rooms to 2,995 MiB after 251;
application/other allocation remains around 1,490–1,730 MiB. The registry WGPU Vulkan
command allocator pools encoders and resets command pools without releasing their
retained storage. This identifies a retention path, not proof that a particular
cache policy is safe to change. The alternate allocator is diagnostic-only;
production dependencies and devices are unchanged. No driver-memory bound or
unlimited-process claim is made. Restarting a shared device could invalidate
retained training tensors and is not an acceptable silent workaround.

## Tests and remaining gates

Final-source checks include 509 core tests passing (26 explicit hardware/diagnostic
ignores), strict native workspace Clippy, strict WASM-plus-motion Clippy, 15 Cycles
contract tests, 135 full-mip material comparisons, three HDR reflection-field
comparisons, seven actual GPU consumer tests, 10,000 actual-mesh constructions and
10,000 independent three-camera manifests with zero invalid seeds. Publication
unit/contract tests additionally protect provenance, current-only receipts,
cohort arithmetic, conditional object parameters and canonical page generation.

Before promoting broader claims or changing training defaults, the remaining
acceptance conditions are: correct tinted/frosted-glass transport against the
reference; a sustained host-memory plateau; uncontended repeated completed-capture
and concurrent-consumer timing; improved interaction/garment/hair realism in
controlled views; and a fixed-budget downstream reconstruction evaluation on
held-out scene families and real validation data. Distortion, rolling shutter,
full cloth simulation, arbitrary CAD topology and renderer patches retain their
separate evidence requirements. No active downstream training is changed by this
local execution.
