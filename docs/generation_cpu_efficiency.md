# Full-quality indoor generation: current CPU and readback work

The qualified pre-release local implementation adds bounded exact glaze preparation, direct
index assembly and reuse of unchanged instance matrices, cached readiness asset
IDs, and shared image-copier arrays. It retains shadows, GI, full material detail,
geometry and requested annotations. Matched three-view 512×512 capture throughput
measures **+4.42% for RGB** and **+0.73% for six-mode capture**. Three-renderer throughput
is effectively flat (**−0.17%**), with **1.18% higher CPU seconds/room**. RGB and
six-mode single CPU seconds/room fall **1.57% and 0.76%**. This is incremental
progress; the requested 2× improvement and downstream generate-to-train speedup
remain unproven. All saved RGB, annotation and metadata files match the accepted
local baseline byte for byte. The [qualification receipt](evidence/cpu_efficiency/receipt.json)
binds source, builds, tests, measured runs and exact replay.

## Qualified implementation

**Prepare fired ceramic once per map.** Glaze preparation shares one actual
allocation-capacity budget of 1MiB across speckle and crack halos, periodic fields
and the melt deposit. Frequencies, footprints and fixed pigment decodes are
computed once. Missing cache entries retain scalar evaluation. Nearest-feature
iteration, interpolation order, both intermediate sRGB encodes/decodes, tint,
mix, clamps and final quantization remain intact. Only the selected glaze branch
prepares, including authored recipes containing unused mineral or paint programs.
Exceptional authored parameters and nonfinite coordinates retain the original
scalar expression. Copied-original fixtures compare float/NaN bits, original
panic type/message where malformed scalar inputs panic, and every consumed
color/normal/ORM map and mip byte.

**Write geometry indices once and reuse exact matrices.** Ground-truth extraction
borrows live mesh index spans until extraction ends, then writes them directly in
the original sorted batch and entity order. U16, U32 and nonindexed meshes retain
the same offsets, topology hashes and valid prefixes on failure. Instance buffers
reuse an inverse-transpose only when every input matrix float bit matches,
including signed zero; transforms and semantic IDs still update each extraction.
Validation, failure order, live mesh events, skin capture/joint refresh, flow,
visibility, culling and layer invalidation retain the original behavior. Spans
are temporary and matrix reuse occupies the existing instance buffer.

**Retain expected readiness IDs between asset changes.** Expected mesh, image and
material ID lists rebuild when the current scene or actual main-world asset
change ticks change, or when future assets change/promote. Live prepared assets,
material bindings and pipeline state are still checked every render frame.
An earlier room cannot release a new room's capture; capture-settling fences
remain in place.

**Share immutable copier arrays.** Image-copier clones share ordered source-image
and staging-buffer arrays through `Arc` slices. Recycling still creates fresh
request/completion state. Plane order, mapping after queue submission, failure
propagation and packet publication after complete unpacking/unmapping are
unchanged. Staging storage and logical copy bytes do not increase.

Retained earlier optimizations include exact mineral color-decode carry; combined
native attachment staging with the original single/oversized/WASM fallback;
four-band native atlas evaluation; exact cellular and periodic halos;
unit-interval wrapping with exceptional-coordinate fallback; overlapping material,
transport and mesh preparation with ordered bindings/tangents; unused parent-map
omission; exact color-mip memoization; in-place normal/GGX moments; owned transport
and probe classification; root-only BVH parallelism; bounded future GI;
render-settling fences; mapping-only waits and borrowed GT decoding. WASM retains
serial atlas preparation and its existing readback path.

Existing bounds remain: paint shares 851,968 bytes per map; the largest cellular
table is 602,176 bytes; mineral and new glaze fields each have a 1MiB per-map
budget; textile tables have a 64KiB logical budget. The process-wide 64-square HDR
filtering stencil occupies approximately 21.47MiB. One full future asset room and
one additional GI-only room share a 256MiB logical upload budget, with current
capture priority. Four shared worker threads can receive help from joining
callers. These are neither a strict process CPU cap nor a total RAM/VRAM ceiling.

## Matched measurements and provenance

Twelve fresh ABBA runs cover RGB, six-mode single renderer and
six-mode three-renderer workloads. Single runs use 64 rooms/eight warmups; pool
runs use three 32-room lanes/four warmups each. The corpus covers 896 room
captures and 2,688 views across 96 seeds, 200–295, including warmups. Steady single
timing excludes warmups. Pool timing starts after every lane finishes warmup and
ends at the first lane's final capture, using each version's own completed count.

All workloads use three 512×512 views, automatic CPU lookahead three, native Auto
quality, 2048 shadow maps, 256 GI rays/probe, three diffuse bounces, SSAO, bloom,
refraction and full detail. Workloads request RGB or Color/Depth/Normal/Position/
Semantic/CoVisibility. Motion, O-voxel, GPU timestamps and encoding are excluded.
Scene preparation is included; validation and JSONL writes are excluded from
steady capture timing. CPU seconds cover complete command lifetimes.

| Workload | Baseline views/s | Current views/s | Change | CPU seconds/room |
|---|---:|---:|---:|---:|
| RGB, one renderer | 11.231 | 11.728 | +4.42% | 1.860 → 1.831 (-1.57%) |
| Six modes, one renderer | 10.700 | 10.779 | +0.73% | 1.905 → 1.890 (-0.76%) |
| Six modes, three renderers | 16.842 | 16.813 | -0.17% | 2.497 → 2.526 (+1.18%) |

Baseline/current repeat ranges are RGB, one renderer **10.987–11.487 / 11.709–11.747** views/s,
six modes, one renderer **10.678–10.722 / 10.771–10.787** views/s,
six modes, three renderers **16.490–17.204 / 16.495–17.166** views/s. Full-lifetime throughput
changes are **+4.00%**, **+0.55%**, **-0.06%** for RGB, six-mode single and pool, respectively. Steady NVML activity is RGB, one renderer **48.43→53.03%**,
six modes, one renderer **56.06→58.92%**,
six modes, three renderers **96.16→96.47%**.
Maximum measured per-process RSS is **3,969,396KiB baseline / 3,929,204KiB current**. All **448 paired timing records** match geometry,
people, camera count, annotation precision/copy bytes and GI counters; transport
bytes do not increase. NVML includes desktop and other processes; it measures
whole-device activity, not occupancy or per-process utilization. Two repeats are
bounded diagnostics, not statistical production assurance.

Current six-mode single preparation averages **304ms** materials,
**170ms** transport, **122ms** meshes,
**76ms** environment and **376ms** for the combined
material/transport/mesh window. These elapsed stages overlap and must not be added
as exclusive CPU costs.

Both builds use optimized development profiles and the existing WGPU package
patches on RTX PRO 6000 Blackwell/Vulkan, with 24 logical CPU cores. Baseline is
the previous accepted local source
`d24f8324c8244944c1692b48130710398058075de7085cc6fa3b2efaaf82bad8`;
its binary is
`c496a69c709bbb0ab7de2522d8a396361a7d84a693a60d0b56924b1251b47b5b`.
The measured pre-release round8 source is
`01dbb8f2f7ff7a6c58a36bcc8f51532c32ab4eae47557f7cdef62de1fef9c46c`,
414 capture inputs and 271 Rust files. Its measured binary is
`d3b57e2b331806abf5fd57b4aabe65aa2495eb70d30f0b435701d6389a4b7afc`.
Final formatting and the malformed-input oracle correction preceded this
freeze, successful checks, fresh replay and benchmark build.

## Diagnostics, rejected candidates and qualification

The allocation-inclusive prepared-glaze CPU diagnostic measures **1.538× at 256**
and **1.555× at 512**, including field setup/evaluation/drop and excluding map/mip
construction, workers, GPU and capture. Twelve full ground-truth CPU diagnostic
records measure **1.077–1.102× fresh**, **1.086–1.124× reused**, and
**1.067–1.115× unchanged** extraction. Changing transforms measure
**0.969–0.991×**, a small regression in that isolated case. These results measure
CPU kernels/extraction, not capture or training throughput.

A fixed-origin reflection cache was slower in both isolated CPU cases and was
removed. Earlier camera activation changed four strict RGB files: 17 channels at
16 glass-edge pixels, maximum delta 0.00018310546875. Restoring conservative
activation reproduced all 108 files exactly in the same five-view configuration;
the underlying HDR/transmission mechanism remains unidentified. These historical
experiments remain rejected, as do producer overlap and ready-packet consumption.
GPU material/geometry baking remains unimplemented.

The measured pre-release snapshot passed **357 core tests, zero failures and 22 ignored**; **eight native
GPU fixtures**; **four Burn generator fixtures**; **one unique production
LiveDataset temporal fixture**; strict Clippy; motion-enabled WebGPU compilation;
and the final benchmark build. Browser runtime was not exercised. Fresh raw
replay matches **908 files / 3,445,511,069 bytes**, across **44 room captures /
140 views** in three overlapping groups: **36 distinct seeds / 116 distinct views**.
All RGB, annotation and metadata replay files are exact against this baseline.

The historical ancestor's separate **17-channel RGB exception at 16 pixels,
maximum delta 0.00018310546875** remains an evidence boundary. This round's exact
replay does not retroactively resolve it. Local exactness does not establish
photographic realism, active ARDY/browser throughput, downstream training speed
or unlimited-process memory stability. This qualification was captured before release closeout, with no commit, push
or publication performed during its measurements. Release provenance and
registry-package checks are recorded separately.
