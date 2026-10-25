# Capture scheduling and demand

The ordinary `LiveDataset::next_sample` path automatically prepares at most three
known consecutive successors (four is the diagnostic maximum). CPU work is shared
by a bounded preparation pool. Future assets may upload during current readback; future material/GI dispatch
still waits until all current capture copies are submitted. Current-room readiness excludes future assets.

After delivering a sample, the native headless runner can preflight one declared
successor while the consumer trains or encodes. Capture cameras remain disabled:
this prepares assets, lighting and requested motion, without rendering an extra
sample. It consumes a lookahead slot and stops at a ready room. One-shot, indexed,
interactive and browser requests do not run this idle preflight. The final sample
of an epoch has no successor budget. Capture readiness, shadows, GI, geometry and
annotation precision remain unchanged. Native indoor captures with stable shading
use an exact completed-render acknowledgement instead of redundant settling
frames. With SSAA this validates both the high-resolution source and resolved
output, their dimensions and scale, plus the scene/camera/timestep identity.
The copy path still requires the current resolve/readback epoch. Unsupported
camera effects retain the conservative settling policy.

Camera activation precedes Bevy's camera, visibility and directional-shadow
preparation. After capture scheduling, a camera may only be deactivated until
the next update. This preserves complete per-view cascades when individual
views finish their copies at different times, while still suppressing duplicate
draws during readback.

Native headless captures also suspend Bevy's independent point/spot shadow
root schedules when no extracted camera is active. Disabling a camera alone
does not suspend these retained light views: they otherwise continue dispatching
mesh preprocessing while scene preparation or readback is pending. The gate
restores the roots immediately after the camera driver, preserving their GPU
resources and full shadows for the next capture. Asset uploads, shader
preparation, material synthesis, GI, annotation and copy systems remain enabled.
Interactive renderers and unrelated auxiliary render schedules are unaffected.
For captures qualified by the exact render fence, an unready asset/pipeline
barrier also suspends the RGB driver, geometric pass and lens/SSAA resolves for
that update. Nothing from that update could pass the existing copy barrier.
The prepared views and their ordering are retained; the next ready update draws
normally. Temporal/fallback warmup is preserved.

Application updates are not rendered frames. For stable native indoor captures,
one requested timestep normally needs one RGB render per pinhole camera, one
shared geometric-annotation pass per view, and an SSAA resolve when configured.
Depth, normal, position and semantics share the geometric pass. Co-visibility
uses those results; it does not render each camera pair separately. Fisheye RGB
uses its angular source views and resolve, so one logical camera is not one
raster view. Temporal effects outside the qualified capture path may still need
settling. `indoor_bench --gpu-timings` records actual render-pass spans separately
from application updates. The `capture/work` diagnostics count driver frames,
scheduled camera views (including fisheye source views),
`unready_camera_frames_skipped`, and `idle_auxiliary_views_skipped`. These
cumulative counts require neither GPU timestamps nor synchronization; they are
scheduling counts, not GPU durations or
draw calls. Annotation-only requests currently retain the ordinary camera path;
the one-frame qualification is not a claim that every possible output/effect
combination executes the theoretical minimum number of GPU passes.

GPU material baking records one recipe per command encoder and submits the
finished buffers in recipe order. The shared scratch workspace, every PBR map
and every mip level are preserved. This bounds the size of each native command
pool's retained workload; it adds no GPU wait or CPU image readback.
When all cameras use direct drawing, point/spot shadows also use direct drawing
with GPU mesh preprocessing. They retain the complete shadow geometry and do
not consume culling bounds left over from a previous scene or draw mode.
GPU irradiance probes enter the lighting scene only after their bake is encoded
and their parent room has passed transform propagation. Capture readiness also
waits for that publication; no extra device fence or probe readback is required.

## Demand-controlled archives

`bevy_zeroverse_burn::generator::run_chunk_generation_controlled(config, control)`
uses the canonical validator, sensor processing, atomic writer and one-writer
rendezvous. `CaptureControl` starts with zero credits. Credits count physical rooms,
including all requested views, timesteps and modalities; they are cumulative
within this run, independent of `sample_offset`. Requests/events identify the
absolute attempted capture index and exact seed.

```rust,no_run
use std::sync::Arc;
use bevy_zeroverse_burn::generator::{GenConfig, run_chunk_generation_controlled};
use bevy_zeroverse_burn::generator::admission::{CaptureControl, CaptureEvent};

let mut config = GenConfig::indoor("out/rooms", 128);
config.seed = Some(4700);
config.chunk_size = 1; // one credit -> one atomically published room
let control = Arc::new(CaptureControl::with_observer(|event: CaptureEvent| {
    // Forward to a short, nonblocking metrics/ledger handler.
}));
// Retain a clone in the consumer's inventory controller.
control.grant_through(4);
run_chunk_generation_controlled(config, control)?;
# Ok::<(), anyhow::Error>(())
```

The consumer calls `grant_through(total)`, `hold(true/false)` and `cancel()` from
another thread. `snapshot()` reports credits, admissions, terminal completions and
the waiting request. Stage events distinguish waiting, capturing (including
readiness/render/readback) and exporting. Completion events identify published,
rejected, failed and cancelled requests, with demand-wait, capture and export durations.
One control belongs to one run; controlled generation requires explicit indoor
seeds, one worker and one room per output transaction. The default stops on an
invalid room. Optional bounded rejection recovery is described below; it never
recycles seeds or publishes a rejected scene.
Keep that run open across inventory waves with credits/hold. Completing or
cancelling it stops the process-owned live renderer; subsequent capture fails
immediately rather than waiting on a dead renderer. A new run requires a fresh
process. Do not share a controlled run with unrelated live capture consumers.

Waiting uses a condition variable outside the capture transaction mutex. It has
no capture deadline. Already dispatched CPU jobs and the archive writer continue;
**idle app updates, future uploads and future GI dispatch are disabled**. Work
already submitted during an admitted capture can finish. Existing speculative
residency is bounded to one complete future asset set plus a second probe, under
the 256 MiB staging cap. This mode does not increase the lookahead depth or transfer
Bevy-owned handles between processes.

Cancellation immediately wakes a waiting request. An admitted transaction cannot
be revoked: it completes under the ordinary capture watchdog and is exported
before shutdown. Writer failure wakes blocked admission; all scoped writer jobs
are joined, and the controlled background renderer is stopped/joined. Each
requested room has exactly one terminal event. A failed transaction that cannot
publish receives a failed completion; previously published archives remain valid.
Observers must not block or panic. Inventory policy, credits, stdin/network
protocols, memory targets and training scheduling belong to the consumer.

## Explicit scene rejection recovery

`CaptureControl::with_rejection_policy(RejectionPolicy::Continue { kinds,
max_total, max_consecutive })` opts into bounded recovery. Both positive limits
and the allowed typed categories are required. `Abort` remains the default.
For example, allow only camera-search exhaustion with at most 16 rejected attempts
in the run and at most two consecutively:

```rust,no_run
use bevy_zeroverse::sample::failure::SceneRejectionKind;
use bevy_zeroverse_burn::generator::admission::{CaptureControl, RejectionPolicy};
let control = CaptureControl::new().with_rejection_policy(RejectionPolicy::Continue {
    kinds: vec![SceneRejectionKind::CameraPlacement],
    max_total: 16,
    max_consecutive: 2,
})?;
# Ok::<(), anyhow::Error>(())
```

Supported categories are missing required program equipment, insufficient usable
program capacity, generated placement-clearance conflicts, and exhausted camera
placement search under valid constraints.
`MissingProgramAnchor` includes a kitchen whose bounded service-run search could
not admit its required equipment. Every manifest integrity check runs before
that decision; partial or malformed kitchen assemblies remain fatal errors.
`PlacementClearance` includes a ceiling fan whose mount or swept rotor clearance
fails after construction. Fan identity/support, nonfinite transforms and all
other manifest integrity checks remain fatal and take precedence over recovery.
Fan admission reserves the maximum possible area-rug thickness before the rugs
are placed, preserving the full 2.1 m headroom when later floor supports appear.
This sampling change is identified by generator revision 67; replay requires
the recorded generator identity, not just the seed. It does not change the
retirement protocol or opt consumers into skipping rejected rooms.
Mixed ceiling/room camera search separates optical visibility from physical
clearance. Ceiling installation triangles participate in visibility; conservative
installation reservations still protect mounting bodies and rail sweeps. The
bounded search retains alternative prefixes and varies placement order without
changing assigned camera indices, lenses, mount roles or motion requirements.
Group spread is checked on the complete rig; pairwise distance and motion checks
also apply during search. `camera_diagnostics.prefix_backtracks` records reuse of
an alternate prefix. Exhausted searches remain typed `CameraPlacement` rejections;
no incomplete camera set is published and no constraints are silently relaxed.

Configuration, manifest integrity, device, shader, readback, annotation, archive,
and unclassified failures remain fatal. No error-text matching is used.

The runner acknowledges recovery only after retiring the failed scene, cancelling
its owned preparation/staging state, dropping old capture attachments and queuing
the next exact seed. It retains the renderer/device and monotonically increasing
readback epochs. Retirement performs no rendering, speculative upload or GPU wait.
The next admitted capture must pass all ordinary geometry, readiness and annotation
checks. Controlled idle waits remain CPU-only.

Each rejected attempt emits `CaptureOutcome::Rejected { rejection, continued }`
with its original seed, category and reason. Rejections beyond either limit, or of
an unlisted kind, stop the run; their terminal receipt still records the rejection.
Cancellation after retirement returns without admitting another scene. A rejection
returns its unpublished reservation: `AdmissionSnapshot::returned` grows, grants
remain cumulative, and admission compares `admitted - returned` with `credits`.
Previously submitted exports complete before a rejection receipt is delivered;
normal successful capture/export overlap is preserved.

`CaptureRequest::sequence` is the attempted capture index. `Completed.publication`
contains the actual closed archive/directory path and compact output index only
after successful publication. A rejected attempt has no publication. Consumers
must use these explicit fields rather than assuming capture indices equal archive
indices. `samples` limits attempted seeds, so an explicitly recovering finite run
can export fewer rooms; zero remains an unbounded stream. For resumption after
rejection, the consumer's seed ledger supplies the next `sample_offset` separately
from the next `chunk_offset`. Archive counting alone cannot recover skipped seed
history. These semantics do not change ordinary fail-fast CLI generation.

## Diagnostics and CPU budgets

`LiveDataset::next_sample_with_telemetry()` returns a sample and optional
`CaptureTelemetry` under the same transaction lock. It contains the room seed,
`PreparationTimings`, cumulative prefetch hit/ready/staged/discarded counters,
idle-preflight counters and readback byte/count totals. The controlled generator
attaches these diagnostics to its terminal event. They are not serialized into
deterministic sample identity and do not synchronize the GPU.

`PreparationTimings.wall` partitions elapsed CPU preparation time, including
queueing. Parallel branch/job sums are overlapping service durations, not total
elapsed time or CPU core time. Capture time includes readiness and readback;
export time covers encoding and atomic output, excluding admission waits.

Applications running several renderer processes may call native
`preparation::configure_preparation_threads(threads)` before the first indoor job.
The validated range is 1–64; the safe default is `min(available_parallelism, 4)`.
Changing an initialized process pool fails. Browser preparation remains cooperative.
`indoor_bench --preparation-threads N --gpu-timings` supports bounded diagnostic
comparisons. GPU timestamps require adapter support; asynchronous diagnostic
windows can overlap and arrive late. Material synthesis, GI, rendering, SSAA
resolve and readback spans are observations, not an additive per-room partition.
No timestamp query or GPU synchronization is added to ordinary capture.

`cargo run -p bevy_zeroverse_burn --example capture_pipeline -- --help` exposes a
qualification harness for ordinary, controlled and live-consumer paths. Compare
identical seeds/settings at 256 and 512, including RGB-only and multi-step labels.
A generator-only speedup is not proof of improved concurrent training throughput;
measure trainer updates, input stalls, memory and energy with the actual consumer.

For sustained memory qualification, `indoor_bench` records RSS, live/free glibc
heap bytes on Linux, live assets, WGPU registries and native GPU allocations for
each completed room. Use a long regenerated-room run, rather than extrapolating
startup growth; allocator arenas and driver pools retain high-water allocations.
For example, `cargo run --bin indoor_bench -- --scenes 1024 --warmup-scenes 64
--cameras 3 --width 512 --height 512 --co-visibility --asset-root .` measures full
native quality with static people and all geometry labels. Compare late windows
and replay matching seeds before accepting a memory or throughput change. A
finite qualification run is not a guarantee of unlimited-process stability.
