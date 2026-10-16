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
annotation precision remain unchanged.

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
absolute output index and exact seed.

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
failed and cancelled requests, with demand-wait, capture and export durations.
One control belongs to one run; controlled generation requires explicit indoor
seeds, one worker and one room per output transaction. The generator stops on
invalid rooms; it never silently recycles or skips their seeds.
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
