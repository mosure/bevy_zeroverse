# Procedural capture throughput plan

Reviewed October 3, 2026 against Zeroverse commit
`077fe924d1a0c3098d8497e7cbd21327840b242a` and the downstream Gekko renderer-pool
study. This is an implementation and qualification plan, not a new performance
result. No generator, active job, dataset, or release was changed during review.

The priority is to shorten the host-side gaps between useful GPU submissions.
Implement observable, bounded overlap within a renderer before increasing process
count. Target **at least 14.36 validated rooms/s, or 43.08 views/s**, against the
reported 7.18 rooms/s configuration, with exactly the same scene and output
contract. This is a qualification target; the evidence does not yet establish
that another 2× is attainable.

## Evidence and workload boundaries

The downstream study compared identical 640-room populations, three 512×512 views
per room. One renderer produced 4.59 rooms/s; three produced 7.18; four fell to
6.13. The three-renderer result matched all 640 baseline archive checksums.
The two- and four-renderer candidates did not match every archive, despite passing
content checks. Each setting was measured once, in order 1, 2, 4, 3. Repeatability
and the causes of the nonselected candidates' differences remain unqualified.

Crucially, the downstream capture wrapper selects **Portable quality, zero human
density, RGB/depth/position, one timestep, no O-voxel, and CPU prefetch depth three**.
Portable disables diffuse GI, shadows, SSAO, bloom, and specular transmission.
`indoor_gi_rays = 64` does not enable GI in this profile. GI, human inference, and
shadow optimizations cannot explain a speedup for this workload. Keep a separate
Auto-quality benchmark with humans and all annotations to protect the broader
engine; do not compare its rates directly with the Portable workload.

The saved single-renderer Nsight trace covers 32 rooms. Its GPU event union is
1.364 seconds in a 5.116-second first-to-last-event interval, leaving 3.752 seconds
without GPU events. Recomputing the event union from the read-only SQLite trace:

| Gap threshold | Number of gaps | Total gap time | Fraction of all gap time |
| --- | ---: | ---: | ---: |
| At least 1 ms | 196 | 3.718 s | 99.1% |
| At least 10 ms | 41 | 3.054 s | 81.4% |
| At least 50 ms | 21 | 2.678 s | 71.4% |
| At least 100 ms | 18 | 2.462 s | 65.6% |

These are queue-work coverage measurements, not SM occupancy or an attribution
of gaps to specific functions. Long preparation/transition gaps are a strong
hypothesis; correlated spans must identify their owners. The trace's 11.27-second
device teardown was distorted by instrumentation and must not be used to justify
persistent workers. Whole-device activity and board energy include desktop work.

The downstream source and evidence are local to `~/repos/burn_gekko`:

- `docs/studies/pilot-30-capture-throughput.md` and
  `.data/experiments/30-capture-throughput/comparison.json`.
- `.data/experiments/30-capture-throughput/renderer-profile.json` and
  `renderer-trace.sqlite` in the same directory.
- `crates/burn_gekko_capture/src/main.rs` for the actual render contract.
- `crates/burn_gekko_data/src/capture/render.rs` for per-shard process creation.

The existing [Zeroverse throughput report](generation_throughput.md) measures a
different, Auto-quality workload with existing checkout WGPU patches. Its stage
times are useful clues, not current production bottleneck measurements. All new
qualification must use published WGPU dependencies, matching the downstream
renderer, and record the executable hash, source, features, compiler and profile.

## What the code currently serializes

| Boundary | Current behavior | Consequence to investigate |
| --- | --- | --- |
| CPU lookahead | `preparation/pipeline.rs` holds tasks and reserved handles; no future assets reach the live stores | A hit can still be an unfinished CPU task. Measure ready hits separately from task hits |
| Room promotion | `procedural_indoor/mod.rs::regenerate` despawns the old scene, spawns entities and commits assets only after preparation completes | Upload and render preparation begin after promotion |
| Readiness | `io/image_copy/readiness.rs` stamps a scene identity but scans all live assets and all waiting pipelines | Naïvely inserting future assets could stall the current capture |
| CPU scheduling | Geometry, mesh conversion, materials and GI BVH have separate pools capped at four threads, in addition to Bevy pools | Three processes can oversubscribe useful CPU resources; Portable does not use the GI pool |
| Capture completion | `sample.rs` waits for all cameras, expands float32 planes and constructs qualification metadata on the ECS thread | CPU completion work can delay the next room's submissions |
| Readback | `io.rs` already uses asynchronous mapping and reusable buffers | Replacing it with another asynchronous API is not itself an optimization; cross-room overlap is missing |
| Output | `generator.rs` overlaps a writer with capture using a bounded rendezvous channel | Measure writer backpressure before enlarging the queue |
| Shards | Gekko invokes a new process for each 64-room shard; `LiveDataset` permits one process configuration | Device setup, shader warmup and caches are repeated |

Native geometric modes already share a float32 ground-truth pass. Do not propose
combining separate depth/normal/position renders as if they were still serialized
in this path. The depth/position workload currently also reads normal/semantic
attachments: optimizing their transfer is a later, measured candidate, not a
reason to change output precision or annotation validation.

## Implementation sequence

### 1 Measure the critical path through the public capture API

Add opt-in `CaptureTelemetry` and a versioned JSONL sidecar, shared by the normal
Burn generator and `indoor_bench`. Keep runtime timing out of scene identity,
dataset tensor bytes and content hashes. Each event needs monotonic timestamps,
request ID, seed, scene epoch, process/thread ID, stage, byte counts and queue
occupancy. Expose the same completed-request diagnostics to library callers.

Reuse `PreparationTimings` and `CaptureProgress`, then add the missing boundaries:

1. Request enqueue/dequeue, CPU task queue/start/finish, ready-hit versus pending-hit.
2. Geometry, texture generation/mips, tangent generation, camera placement and,
   when enabled, GI setup; task CPU time separately from elapsed wait time.
3. Asset insertion, deferred ECS application, extraction, mesh/image upload,
   pipeline compilation, GI dispatch, and the reason each readiness check waits.
4. Warmup versus useful camera frames, GPU submission start/end, per-pass GPU
   timestamps, and room-associated GPU idle gaps.
5. Copy encoded/submitted, map registration/completion, row packing, plane
   expansion, metadata computation, and sample-channel delivery.
6. Sensor transforms, encoding, compression, filesystem writes, writer wait,
   downstream finalization and ordered commit.

Current preparation counters include worker joins and exclude deferred ECS work,
GPU baking and uploads. Do not add overlapping stage durations and call the sum
wall time. Report exclusive blocked time on the critical path, per-thread CPU
seconds, p50/p95/p99 room latency, queue occupancy, allocations, RSS and VRAM.
Count useful submitted captures separately from warmup/empty updates.

**Deliverable:** a timeline for the exact Portable profile and an Auto profile,
identifying the stages responsible for most long gaps. Instrumented traces explain
causality; repeated uninstrumented runs establish speed. Telemetry must be bounded,
optional and cheap when disabled. Web builds retain CPU counters and explicitly
report unsupported GPU counters.

### 2 Reduce CPU cost and coordinate thread budgets

First compare the sealed renderer's actual build profile with an optimized release
build. The downstream manifest specifies dependency opt-level 2 in dev, whereas
this repo's dev dependencies use 3; inspect build provenance rather than assuming
the deployed binary used either profile. Compare artifacts at identical settings.

Introduce a process runtime configuration separate from the scene recipe, with a
total preparation budget and explicit Bevy/render/export allocations. Replace the
independent preparation pools with a bounded executor or coordinated permits.
Account for the scope caller and nested work; do not block all executor threads
waiting on child tasks in the same exhausted pool. Capture/render progress must
have CPU time reserved. Configure before global pools are initialized and report
the effective values; reject incompatible reconfiguration rather than ignoring it.

Test a small thread-budget matrix around one and three renderers. Include CPU
run-queue delay and the workstation's mixed performance/efficiency cores; aggregate
CPU percentage alone cannot distinguish contention, serial code and bandwidth.
Do not simply divide 24 logical CPUs equally or count four pools as four cores.

Use the stage profile to select CPU kernels: tangent construction, procedural
texture/mip construction, camera collision/overlap sampling, geometry assembly,
and annotation expansion. Retain seeded operation order, identical geometry and
all finite-value checks. Share immutable intermediates where inputs are identical;
bound caches by bytes and complete input identity. Do not reduce texture detail,
triangle counts, camera search quality or generation diversity.

**Deliverable:** lower CPU seconds per completed room and fewer preparation stalls
under the three-renderer load. A faster isolated renderer that slows the pool is
not a production win.

### 3 Move completion work off the render critical path

Separate an immutable captured packet from its CPU conversion into `Sample`.
Snapshot matrices, AABB, scene manifest, camera order, semantic attachments,
physical time, frame/flow epoch and all qualification inputs when capture is
issued. Move plane expansion and metadata work to a bounded CPU completion stage,
preserving output order. GPU map callbacks should do only the safe copy/unmap
handoff, not substantial validation or conversion.

Start with at most two completion packets. Retain each staging allocation until
mapping and ownership transfer finish; if reusing camera targets, prove the copy
was queued before those targets are overwritten. Do not publish a partial room.
Associate errors with the original request even if later work has started, stop
ordered publication at that request, and keep cancellation bounded.

Investigate unnecessary normal/semantic transfers only after measuring copy and
decode cost. Preserve every requested channel, float32 precision, first-surface
policy, geometric validation and external export formats. Specializing requested
outputs must not silently remove validation of the renderer's geometry.

**Deliverable:** CPU completion for room N overlaps useful work for N+1 without
mixing annotations, camera metadata or errors across rooms.

### 4 Add one bounded future GPU asset slot

Use an explicit room lifecycle:

```text
CPU queued -> CPU ready -> upload pending -> GPU assets ready
           -> active capture -> readback pending -> CPU completion -> retired
```

Begin with one active room and one future GPU asset set. The existing CPU queue
may hold further rooms, but capacity must cover all CPU/GPU/completion residency
and obey byte budgets, not just room counts. Keep this opt-in until qualified.

The prerequisite is **readiness scoped to the assets and pipelines required by a
specific scene epoch**. Store immutable required mesh/image/material IDs. Future
assets or unrelated compilation must not hold up the current room; current-room
missing assets must still prevent capture. Track pipeline dependencies explicitly,
including pipelines discovered only after entity extraction. Asset upload alone
does not imply that a room is capture-ready.

Commit next-room assets early and allow Bevy to prepare them while the current
room renders or maps. Initially keep future entities, lights, cameras and physics
out of the active ECS scene. This avoids cross-room shadow, lighting, culling and
semantic contamination. Promote the ready asset set atomically with its manifest,
camera program and fresh capture epoch. Retire old assets only when their GPU and
readback owners have finished using them.

Uploads use supported Bevy/WGPU paths. Overlapping scheduling on one device does
not guarantee independent hardware copy/graphics execution. Measure submission
gaps and current-room latency; future uploads need a budget so they cannot starve
the active capture. Test staging with existing CPU prefetch depth three before
increasing either queue depth.

For Auto quality, GI currently has singleton request/prepared/readiness resources.
Make those per-slot before pre-baking a future room, or leave GI promotion-only
in the first implementation. Motion inference stays lazy and per-request; rooms
with no motion configuration must not load models. Reuse caches/loaders when that
feature is enabled. WASM starts with the existing cooperative path; enable GPU
staging there only after browser-specific residency and readiness qualification.

**Deliverable:** the largest transition gaps shrink while scene distributions,
RGB, geometry and every requested annotation remain correct.

### 5 Keep workers alive across compatible shard requests

Add an owned `CaptureSession` and a versioned worker protocol rather than relaxing
`LiveDataset`'s global configuration assertion. Preserve the existing one-shot API
as a wrapper. Separate fixed device/features/assets configuration, validated scene
recipe, request seed range, and output destination. At a request boundary, drain
old work, validate the next recipe, reset temporal/capture state and invalidate
lookahead when its full key changes. Carry explicit seed ranges across compatible
shards so tail draining does not discard already authorized lookahead.

Start with successive seed ranges for one fixed recipe. Then support the eight
production appearance/baseline recipes through validated idle-boundary updates or
a bounded recipe-aware worker cache. Never create eight unbounded devices merely
to retain eight recipes. Keep the chosen maximum renderer count separate from
CPU finalizers and total in-flight shard capacity.

Use request IDs, result/failure receipts, cancellation, EOF shutdown, ordered
commits and atomic output completion. Gekko keeps responsibility for shard repair,
quarantine and resume; the renderer reports errors instead of selecting replacement
seeds. Record new executable provenance and preserve original dataset split and
scene-family membership.

Measure uninstrumented startup, first-room and shard-boundary time before ranking
this change. Persistence amortizes setup and shader caches; it does not by itself
fix steady-state per-room gaps. Initially enforce an orderly recycle threshold
between shards and test long-run memory before making unlimited lifetime a default.

**Deliverable:** comparable cold/warm shard timings and bounded long-lived worker
state, with production recovery behavior preserved.

### 6 Change render scheduling only where the timeline justifies it

After bounded overlap works, inspect expensive extraction, queue building,
draw submission and repeated empty/warmup frames. Advance a capture when its
specific dependencies complete; do not repeatedly run full app/render schedules
solely to check a CPU task. A progress-driven runner must still service Bevy
executors, uploads, GPU polls, cancellation and timeouts.

Keep required temporal settling for shadows/post-processing/flow. Replace a fixed
frame delay only with an equivalent tested readiness condition. The 50 ms request
receive timeout wakes on arrival and is not a fixed per-room delay. The asynchronous
capture path is distinct from the legacy blocking prepass/O-voxel paths.

Only investigate cross-room render batching or draw/asset arenas if submission
work remains dominant. Multiple visible rooms in one ECS world require isolation
of lights, environment, shadows, exposure, temporal history and annotations; this
is a larger correctness project than pre-uploading immutable assets. No custom
WGPU fork, zero-copy CUDA bridge, or reduced render quality is a prerequisite.

## Qualification and stop rules

Keep production generation untouched during investigation. Run controlled
performance experiments in an isolated measurement window after production is
idle or separately authorized to pause. Do not compile or launch competing GPU
benchmarks and then treat those results as a clean comparison.

1. **Baseline and noise:** repeat the current Portable 640-room screen at least
   three times. Record A/A tensor/hash agreement, stage timings, effective CPU
   budgets, cold/warm time, memory, telemetry coverage and actual provenance.
2. **Small screen:** use a fixed 64-room subset covering all eight recipes and
   32 Auto-quality rooms for each isolated change. Include seeds 202 and 1,013,005
   in a separate geometric regression set. Reject correctness failures immediately.
3. **Matched acceptance:** compare the surviving candidate against the current
   three-renderer baseline on the same 640 rooms, in counterbalanced repeated
   runs. Report validated committed rooms/s, CPU seconds/room, latency tails,
   useful GPU coverage, RSS/VRAM and board joules/room. Report per-recipe results;
   do not hide a slow or failing recipe in the average.
4. **Output equality:** compare tensor bytes, calibration, poses, seeds, manifests,
   object/semantic attachments and complete requested annotations. Archive hashes
   are decisive when metadata is identical; for a new executable, separate known
   provenance/runtime fields explicitly and compare decoded payloads exactly.
   Investigate any nondeterminism with A/A controls. Do not widen tolerances merely
   because a candidate is faster or drop difficult seeds.
5. **Lifecycle:** exercise cancellation during preparation/upload/map/export,
   partial failure, config changes, queue saturation, stale scene epochs, resume,
   first-frame correctness, co-visibility, optical flow/motion and static O-voxel
   constraints. Test native and WebGPU fallbacks. Run strict Clippy and focused
   tests before the broad render qualification.
6. **Endurance:** after a winning change, run at least 10,000 consecutive rooms
   through persistent workers and alternating recipes, within an explicit budget.
   Compare equal population blocks after warmup; bound live CPU/GPU allocations,
   completed-but-retained requests, cache bytes and asset counts. Report measured
   memory slopes and recycle behavior, not an unlimited-stability claim.

Measure startup removal, CPU optimization, staging and persistence independently
before combining them. Rank each by its removable critical-path fraction, not by
GPU utilization alone. Preserve a candidate only when its repeated end-to-end
gain exceeds measured noise and its resource costs are acceptable. Stop broad
parameter sweeps once a bottleneck has moved; re-profile the combined candidate.

Success is increased **validated dataset throughput at the same quality**. A
higher utilization number caused by redundant rendering, busy-waiting, more
shaders or excess speculative work is a regression. If 2× is not reached, report
the measured gain and the remaining critical path rather than claiming completion.
