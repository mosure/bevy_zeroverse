# Configurable GPU capture and CUDA process transport

This API is introduced in `bevy_zeroverse 0.45.0` and `bevy_zeroverse_burn 0.29.0`.
Local GPU checks and an independent downstream training study are recorded
separately below, with the source snapshot and scope of each result.

## Capture policy and training color

`CaptureSpec::from_scene(config).resolve()` validates one immutable scene policy.
`render_modes` is authoritative; an empty list resolves through `render_mode`.
RGB-only requests skip the Zeroverse geometric annotation camera, uploads and
MRT. Partial annotation requests retain paired MRT dependencies, but copy only
requested canonical planes. The full `indoor()` preset remains annotated.

`GpuView` planes are now `Option<Tensor<3>>`. Unrequested data is absent, never a
fake zero map. `plane(Plane)` is a checked accessor. `depth()` and
`semantic_ids()` return optional single-channel views. Depth is metric camera-Z:
request `DepthFormat::Linear` and `z_depth=true`; other depth formats are rejected
for this transport. Normal/semantic and world/depth remain documented packed
planes, not palette-colored labels.

Raw output remains tone-mapped **linear RGBA float32**. Training pipelines that
expect archive-style RGB must explicitly request **sRGB RGB float32**:

```rust,no_run
use bevy_zeroverse::render::RenderMode;
use bevy_zeroverse_burn::{generator::GenConfig, gpu::{
    CaptureSpec, CaptureRuntimeOptions, ColorOutput, GpuLiveDataset,
}};
let mut config = GenConfig::indoor("unused", 0);
config.render_modes = vec![RenderMode::Color];
let runtime = CaptureRuntimeOptions::from_gen(&config);
let spec = CaptureSpec::from_gen(&config)?.resolve()?; // sRGB RGB by default here
let mut rooms = GpuLiveDataset::builder(spec).runtime(runtime).build()?;
let sample = rooms.next_sample()?;
let rgb = sample.stack(bevy_zeroverse_burn::gpu::Plane::Color)?;
// [views, height, width, 3], timestep-major, on rooms.device().
// Do not apply linear_to_srgb again in the training loader.
# Ok::<(), anyhow::Error>(())
```

For a Bevy config directly, select `.color_output(ColorOutput::SrgbRgbF32)` on
`CaptureSpec`. `SrgbRgbaF32` preserves alpha without encoding it. A cached CubeCL
kernel fuses row depadding, RGBA-to-RGB packing and the IEC sRGB transfer, on
both CUDA and WGPU. It runs **after** linear-light SSAA. It clamps finite RGB to
[0,1], matching the archive oracle; it does not tone-map twice, quantize to 8-bit,
normalize each image or alter any annotation. `metadata.color_encoding` describes
the delivered tensor; `gpu_tensor_contract.source_color` records the input.

Nonfinite values are preserved rather than hidden by clamping. Explicit
`sample.validate_finite()` reduces on the GPU and reads one boolean; it fails on
nonfinite data and adds a completion fence. Qualification and training acceptance
should use this guard where their validation policy requires it. Normal capture
does not conceal a host image readback inside preprocessing.

`GenConfig::scene_config()` is shared with archive generation. GPU sensor
transforms, O-voxels and schematic/archive export are explicitly rejected by
`CaptureSpec::from_gen`; use their existing APIs. Scene geometry, lighting,
shadows, glass, SSAA, appearance, humans and motion are not reduced for transport.

## Explicit demand and preparation

`GpuCaptureBuilder::build_session()` creates a `GpuCaptureSession` instead of an
automatic consecutive stream. `plan_requests()` reserves up to four actual future
seeds for CPU preparation, including lane-strided seeds; it grants no rendering
credit. `try_submit()` bounds pending, active and completed-but-unreceived rooms.
`try_receive()` and `receive_timeout()` leave outstanding requests intact on a
timeout. Each completion carries its sequence and exact seed.

An optional shared `CaptureControl` supplies cumulative room credits, hold and
cancel. GPU delivery emits `CaptureOutcome::Delivered`, distinct from durable
archive publication. Typed construction rejections, including `PlacementClearance`
for the known floor-furniture/ceiling-clearance failure, return credits only after
scene retirement and follow the existing bounded `RejectionPolicy`. Transport
failures are not relabeled as scientific rejections. Recovery defaults to `Abort`;
opt in with `RejectionPolicy::Continue`, allowed kinds and finite total/consecutive
limits. Validation is never relaxed, and malformed scene data remains fatal.
Already admitted work may finish after a hold; waiting for demand does not spend
a capture timeout.

`new(config)` and `new_cuda(config)` remain consecutive-stream conveniences.
They honor `runtime.capture_timeout`. A receive timeout preserves the outstanding
request; a consumed terminal rejection is returned to the caller and advances the
consecutive cursor without silently retrying that seed. Explicit sessions retain
caller ownership of request sequencing.
For existing CUDA training devices use `.cuda_on(device)`; UUID/context mismatch
fails closed. The current in-process path validates the selected Vulkan adapter;
on machines with several adapters, select the matching Vulkan adapter explicitly.
`cuda_ipc` always checks each producer against the receiver's existing CUDA device.

## Supervised Linux CUDA workers

Enable `bevy_zeroverse_burn/cuda_ipc`. Default and WASM dependency graphs contain
no CUDA IPC dependencies or global patches. The supervisor retains child process
ownership, seed ledgers, checkpoint state, replacement and ordering policy.
If the consumer uses Burn's fused backend, also enable
`bevy_zeroverse_burn/gpu_tensor_fusion` so the returned tensor backend matches:

```toml
features = ["cuda_ipc", "gpu_tensor_fusion"]
```

- A fresh exec'd child calls `gpu::ipc::GpuCaptureServer::serve(UnixStream, spec,
  runtime, lane)`. The `zeroverse_gpu_worker` executable accepts `--config` JSON,
  optional `--runtime` JSON, `--lane`, and either an inherited `--socket-fd` or
  `--connect` Unix socket path. Never fork and continue using initialized GPU state.
- The trainer calls `CudaCaptureReceiver::attach(existing_device, spec, lanes,
  runtime, control)`. All peers must have the same UID and policy; device UUIDs,
  protocol/generator/package identity, descriptor lengths and ranges are validated.
  Set `runtime.expected_generator_source` to pin a specific renderer source hash;
  the actual engine provenance is recorded regardless of that optional pin.
- `plan_requests(lane, requests)` supplies bounded lookahead. `try_admit(lane,
  request)` returns false on hold, exhausted credit or bounded backpressure.
  Receive any lane's completion with `try_receive` or `receive_timeout`. Restore
  the scientific training order in the downstream ledger, not arrival order.
- After consuming a failed/drained lane's completion, `replace_lane()` retires
  its imports before attaching a replacement. It preserves request sequence
  history and other lanes. Resolution/policy changes require a new drained
  receiver; delivered tensors remain valid. Credit/checkpoint ownership is not
  silently transferred to the new session.

Each lane registers **one dedicated Vulkan arena and one binary semaphore** via
SCM_RIGHTS. The arena reserves two raw rooms to cover active and retiring camera
bindings. Memory/semaphore FDs are exported/imported once per session, not per
view. Captures reference nonoverlapping camera/time ranges with session,
incarnation, request and generation identities. The entire room uses one managed
Burn destination reservation and one transfer-stream completion fence, even for
40 views. Padding removal and color conversion stay on the GPU in one kernel for the
entire room. `sample.stack(Plane::Color)` returns that contiguous allocation;
individual view tensors share it without separate color copies.

The exporter releases queue-family ownership and completes its signal before
sending `Ready`. Thus an ingress worker never waits for an unsignaled semaphore
from a producer that died before readiness. After the CUDA wait and device copy
complete, the receiver acknowledges release; Vulkan then reacquires ownership.
A duplicate/stale acknowledgement cannot recycle a slot. Receiver-owned tensors
survive subsequent captures and producer termination. Imported allocations and
managed destination reservations survive partial-copy errors until GPU reads
retire. Cleanup never resets or synchronizes the entire training context.

Receive deadlines bound the control plane, not arbitrary driver calls. A timed-out
lane is quarantined; replacement is refused until its import worker retires.
`shutdown(timeout)` reports incomplete retirement instead of hanging the training
thread. Destructors wake sockets/queues and do not join an unresponsive driver.
The downstream supervisor may terminate only its own renderer process. A physical
GPU fault can still affect training on that same GPU.

`CudaCompletion.timings.cuda_import` reports receiver-side allocation,
allocation-dependency fence, copy enqueue, transfer fence, and color/layout pack
intervals, plus copied bytes/views and total import time. These are host wall
intervals around existing operations; instrumentation adds no GPU synchronization
or image readback. Pack time measures compilation/submission and metadata work,
not asynchronous kernel execution. Producer times remain separate: `cuda` is the
in-process producer bridge and is normally absent for IPC. Older producer receipts
without `cuda_import` remain readable; the receiver supplies its own measurements.

## Budgets and qualification

Runtime defaults bound preparation to three future rooms, the in-process queue
to two admitted/unreceived rooms, raw capture to 512 MiB per renderer, and aggregate
registered IPC arenas to 1 GiB per receiver. Limits reject excessive policies;
they never lower resolution or quality. Already delivered Burn tensor inventory
is owned and budgeted downstream. Do not convert a 4 GiB host inventory target
into an implicit 4 GiB GPU cache alongside a training model.

`queue_capacity` is a global room bound for an **in-process** capture session,
not per camera. IPC automatically permits one outstanding/unreceived room per
connected lane and sizes its completion queue to the lane count. Thus four IPC
lanes can run concurrently without overriding `queue_capacity`; cumulative
`CaptureControl` credits and the downstream inventory budget still apply.

| RGB-only physical room | Raw linear RGBA | Delivered sRGB RGB | Registered arena/lane |
| --- | ---: | ---: | ---: |
| 8 cameras × 5 times × 256² | 40 MiB | 30 MiB | 80 MiB |
| 8 cameras × 3 times × 512² | 96 MiB | 72 MiB | 192 MiB |

Pure tests cover descriptor bounds, overflow/overlap, duplicate and stale leases,
incarnation replacement, ancillary FD ownership/framing, authoritative outputs,
policy hashes, strided/repeated-seed lookahead and admission accounting. GPU tests
separately cover the analytic sRGB oracle, rendered archive parity, RGB-only
annotation work, existing full annotation/SSAA parity and retained outputs.

The bounded real-process harness accepts the actual production shapes:

```sh
cargo build -p bevy_zeroverse_burn --features cuda_ipc --bin zeroverse_gpu_worker --example gpu_ipc_qualification
# Run only after coordinating hardware use with any live training process.
cargo run -p bevy_zeroverse_burn --features cuda_ipc --example gpu_ipc_qualification -- \
  --worker target/debug/zeroverse_gpu_worker --lanes 1 --side 256 --cameras 8 --steps 5 --rooms 2
# Repeat with four lanes; then two lanes, 512, eight cameras and three steps.
```

This harness checks sRGB shape/encoding, finite results, zero RGB-only annotation
uploads, receipts and retained tensors after cross-thread use and worker shutdown.
Its boolean/reduction readbacks are validation telemetry, not host image transport.
Full forward/loss/gradient/optimizer replay and refreshed inputs in the downstream
CUDA graph must be qualified in the training project before production adoption.
Removing PCIe image traffic alone does not prove a training throughput gain.

Direct foreign-allocation Burn tensors remain conditional on an upstream owner,
readiness and last-use contract. This implementation intentionally uses an owned
CUDA destination copy and published dependencies.

## Local validation record

The [machine-readable receipt](qualification/gpu_process_capture.json) records
this pass on an RTX PRO 6000 Blackwell. GPU checks passed for the analytic sRGB
oracle, RGB-only rendered/archive parity, and WGPU/CUDA full-annotation parity
with SSAA=2 and both glass policies. Calibration and human poses matched;
retained tensors survived regeneration and renderer destruction.

CPU validation passed 47 adapter tests, two portable wire-contract tests, and
four core regressions for exact lookahead and typed rejection versus fatal
integrity failures. The CUDA IPC/training/fusion and WGPU-only feature combinations
passed Clippy with warnings denied, and the core WebGPU library passed its WASM build
check. Cargo reports an existing future-incompatibility notice in upstream
`burn-cubecl-fusion 0.22.0`.

The process probes passed for **14 rooms / 428 views**: one lane with padded
129² views and replacement, four lanes with 8 cameras × 5 times × 256², and two
lanes with 8 cameras × 3 times × 512². Replacement preserved request history,
rejected an incomplete handshake, and retained delivered tensors. GPU use was
coordinated with the trainer; every child exited before production resumed.

These are correctness probes. The zero host-image-byte field is a transport
contract assertion, **not measured PCIe traffic**. The tests also deliberately
read reference images or validation scalars. These local probes do not measure
concurrent training or long-run stability; the downstream study below supplies
separate training and profiler evidence.
The updated harness records the full requested policy and actual per-room
sampling metadata, asserts effective SSAA=2, and labels unmeasured traffic
explicitly. Its optional `--kill-during-capture` fault probe was not run in this
window; the drained-producer replacement probe was run.

## Downstream training qualification

The [downstream receipt](qualification/gpu_process_capture_training.json) records
Burn Gekko experiment 121, inspected from its local report on 2026-10-18. It used
an immutable pre-release snapshot of this API (manifest
`82fc787236cc6b52da8b9baabf292aaf7089293b3c1baefd02a2e8cd667a17a8`), the same
model/optimizer starting state, scientific settings, and SSAA=2. Benchmark arms
used independently reserved fresh scenes; these are operational throughput
comparisons, not matched-seed learning comparisons.

| Shape | Producers | Archive s/update | IPC s/update | Pair throughput gain |
| --- | ---: | ---: | ---: | ---: |
| 256², batch 10, 8 cameras × 5 times | 4 | 1.1601 | 1.0555 | 9.90% |
| 512², batch 4, 8 cameras × 3 times | 2 | 2.7418 | 2.5863 | 6.01% |

The clean 256² IPC window contains 512 steady updates; the archive and both
512² windows contain 256. All four windows reported zero input waiting. Two
producers at 256² were insufficient (26% waiting); three achieved 1.0949 s/update.
The downstream tensor inventory was bounded to 2 GiB, excluding renderer arenas,
temporary import buffers and active training allocations.

Untimed parity audits at both resolutions matched RGB and encoder features
exactly, with gradient differences around 3.6e-9 (repeat-backward noise), zero
numerical parameter/optimizer differences, and 918 optimizer tensors checked.
Separate checkpoint replay verified exact serialized optimizer records and the
next update. Each resolution passed a 32-room holdout check. Killing an owned
active producer failed visibly and released owned processes/GPU memory in 1.88 s.

A separate warm trainer trace removed the 62,914,560-byte image upload: total
host-to-device bytes fell from 62,931,032 to 19,064 (99.97%). Remaining uploads
were mask/runtime metadata. The trace observed 80 device-to-device copies
totalling 0.353 ms; it does not cover concurrent Vulkan producer traffic or imply
zero device copies. Profiler timings are excluded from the throughput windows.

Renderer RSS still rose across the four quarters of the 512-update run, from
18.24 to 22.69 GiB in aggregate. This is **not** a memory-plateau or unlimited-run
stability qualification. The recorded snapshot predates the receiver timing
instrumentation above; those additions were checked separately without another
GPU run. Published-package portability and current production state are separate
from this source-snapshot qualification.
