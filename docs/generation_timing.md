# Indoor pipeline timing

`indoor_bench` measures completed captures. `indoor_prepare_bench` measures CPU
preparation through the **same `prepare_indoor` entry point** used by the viewer
and native lookahead. Its default prepares production GPU material recipes; it
does not execute their GPU bakes. `--cpu-materials` is an explicit CPU oracle.
Neither CPU preparation nor rendering initializes motion models when motion is
unspecified.

## Accounting

`PreparationTimings.wall` partitions a single future's lifetime into disjoint
monotonic intervals: layout, cameras, validation, motion planning, assemblies,
material selection, parallel assets, transport binding, reflection environment,
lighting requests, browser mesh tail, group binding, and bookkeeping. These are
elapsed wall durations, including queue delays/cooperative yields, **not CPU core
seconds**. The material/transport/mesh fork is counted once. Existing branch
counters remain diagnostic; do not sum them or different rooms into a wall total.
Per-category geometry job counters record count, sum and maximum builder service
time, excluding the queue before execution. All timings stay out of manifests,
annotations and deterministic sample identities.

Main-world `asset_insertion_seconds` includes command recording and insertion,
not deferred command application or GPU upload. GPU maps, GI, optional ARDY,
pipeline compilation, settling, rasterization and readback are downstream of CPU
preparation. Capture waits for their existing readiness gates. Event-loop timing
buckets label the blocker observed **before** the update; they are not exclusive
CPU/GPU kernels. GPU diagnostic intervals may nest and arrive asynchronously.

Use `stream_scenes_per_second` / `stream_views_per_second` for delivered throughput.
Their denominator includes validation, JSONL diagnostics and sample/asset drops:
prefetch continues during those operations. Legacy `scenes_per_second` measures
only capture spans and can overestimate delivery throughput. Initialization,
warm-up, saved image planes, tensor interop, compression and concurrent training
must be identified separately. Throughput with `--save-samples` is diagnostic.

## Reproduce the paper protocol

Use an optimized native build and a fresh output directory. Record the hardware,
compiler/profile, competing workloads and board-level telemetry. Do not silently
pause training or describe a shared device as isolated. Run these workloads
sequentially; compilation or another local benchmark would confound the study.

```sh
cargo build --bin indoor_prepare_bench --bin indoor_bench
cargo run --bin indoor_prepare_bench -- --rooms 64 --seed 400 --concurrent 1 --output out/timing/cpu1.json
cargo run --bin indoor_prepare_bench -- --rooms 64 --seed 400 --concurrent 4 --output out/timing/cpu4.json
cargo run --bin indoor_bench -- --scenes 80 --warmup-scenes 16 --seed 384 --cameras 3 --width 512 --height 512 --co-visibility --output out/timing/capture1
cargo run --bin indoor_bench -- --scenes 80 --warmup-scenes 16 --seed 384 --cameras 3 --width 512 --height 512 --co-visibility --output out/timing/capture2
cargo run -p bevy_zeroverse_publication -- timing-study --preparation out/timing/cpu1.json out/timing/cpu4.json --captures out/timing/capture1 out/timing/capture2 --environment out/timing/environment.json --output out/timing/paper
```

The environment JSON requires human-readable `hardware`, `build`, and `contention`
strings. The report validates source identities, full-quality settings,
consecutive populations, timing partitions, denominators and warm-up exclusions.
It writes `pipeline_receipt.json`, LaTeX, and reproducible SVG/PNG figures. Install
these `pipeline_*` files in `tex/generated` for local paper compilation. No
project-page files are generated. The full publication pipeline regenerates this
paper section from the receipt and rejects stale measurements instead of carrying
them into a later generator snapshot. Original raw logs can remain local; the
receipt preserves measured rows, configuration, provenance and input checksums.

Preparation is split into assembly/material selection, typed mesh group plans,
shared bounded workers, staging/residency, lookahead, cooperative execution, and
timing modules. Independent jobs own their random streams and reserve asset
identities before completion. Tests compare parallel geometry and deferred mesh
realization with ordered references, test speculative cancellation/promotion, and
reject misleading timing receipts. Numeric annotations and same-seed rendered
replay remain necessary after changes to this boundary.
