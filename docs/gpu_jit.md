# Same-device JIT training

Enable `bevy_zeroverse_burn/gpu_tensor` and retain one `GpuLiveDataset`. Use its
`device()` for your Burn model, optimizer and tensor operations:

```rust,no_run
use bevy_zeroverse::{app::BevyZeroverseConfig, scene::ZeroverseSceneType};
use bevy_zeroverse_burn::gpu::GpuLiveDataset;

let mut rooms = GpuLiveDataset::new(BevyZeroverseConfig {
    scene_type: ZeroverseSceneType::ProceduralIndoor,
    num_cameras: 3, width: 512., height: 512., playback_steps: 1,
    ..Default::default()
})?;
let training_device = rooms.device().clone();
let sample = rooms.next_sample()?;
// sample.views[0].color is Tensor<3>, [height, width, RGBA].
// Construct the model on training_device; no CPU image conversion is needed.
# Ok::<(), anyhow::Error>(())
```

`GpuLiveDataset::indoor()` supplies a three-view 512x512 single-timestep preset.
An omitted seed starts a random, recorded consecutive stream. Explicit configs
retain their requested timesteps and annotations.
Occupied rooms need the AnnyBody assets in `assets/burn_human`. Set
`BEVY_ASSET_ROOT` to the project directory containing `assets` before constructing
the dataset, just as for native capture. Missing assets return a capture error.

One persistent capture worker automatically prepares one future GPU sample
while the consumer trains. Requests and results are bounded; retained samples
belong to the consumer. `sample_seed(seed)` drains any queued lookahead before
indexed capture and preserves the consecutive stream's cursor. Dropping the
dataset joins its worker; returned tensors remain valid afterwards.
For Burn's data loader, call `rooms.into_dataset(samples_per_epoch)`. Its
`device()` is the same training device. This is a live stream: indices bound
an epoch, and every `get` returns a new room, including across epochs.

The renderer keeps native quality, shadows, diffuse indirect lighting and scene
lookahead. It copies the existing RGBA32Float render attachments directly into
Burn-owned storage on the shared wgpu queue. Capture publishes after queue
submission, without mapping image data or waiting for GPU completion. Burn work
on that queue observes the copies in order. Allocator leases remain alive until
the render copies finish, including when a sample is dropped early. Samples
retained by a consumer cannot be overwritten by subsequent rooms.

`GpuSample` separates small CPU metadata from GPU images; it is deliberately not
a CPU archive `Sample`. View order is timestep-major. Row padding is removed
from each tensor's logical shape. Outputs preserve the raw raster contract:

| Tensor | RGBA channels |
|---|---|
| `color` | Tonemapped linear RGB, alpha |
| `world_depth` | World XYZ, metric camera Z depth; depth zero denotes background |
| `normal_semantic` | Encoded geometric view normal, integral NYU/Hypersim class ID |
| `optical_flow` (requested) | Forward pixel dx/dy, source validity, target visibility |
| `co_visibility` (requested) | Camera membership bits, peer count, source validity, zero |

Flow is shifted from the raster pass to the source-frame convention, matching
CPU archives. The terminal source has no successor and has zero/invalid flow.
`depth()` and `semantic_ids()` return single-channel tensors; other derived
representations can be computed on the training device. Glass policy, camera
calibration, room bounds, object boxes and pose metadata remain explicit.
This transport currently supports native procedural indoor scenes. Use the
CPU archive API for O-voxel export, filesystem datasets and browser capture.

If your model enables Burn fusion, enable `gpu_tensor_fusion` for its allocation
bridge. The published human inference 0.2.0 render bridge uses
an unfused primitive: combined fusion and optional `human_motion` need upstream
fusion support. Static human geometry does not load the motion models.

Run the matched 512x512, three-view benchmark (full quality in both paths):

```sh
cargo run -p bevy_zeroverse_burn --features gpu_tensor --example gpu_jit_bench -- --scenes 64
cargo run -p bevy_zeroverse_burn --features gpu_tensor --example gpu_jit_bench -- --gpu --scenes 64
```

The benchmark reads a small consumer reduction to measure completed tensor work,
like a training loop reading its loss. It does not synchronize the whole shared
device, which can also include unrelated future-room work. Production capture
does not impose this per-sample completion fence. CPU
construction and training can still dominate end-to-end throughput; a removed
host round trip alone is not evidence of a twofold speedup.

Native room preparation already overlaps bounded future rooms and parallelizes
geometry, material maps and tangents. Lighting trees have a separate bounded
pool. Increasing thread counts alone can slow full captures; CPU-stage timing
and completed tensor throughput must be measured separately. WASM keeps its
cooperative serial path. Scene quality and GPU readiness checks remain intact.

To isolate preparation from GPU rendering, uploads and tensor transport:

```sh
cargo run --bin indoor_prepare_bench -- --rooms 64 --concurrent 4 --output out/preparation.json
```

This diagnostic prepares full native geometry, PBR maps and GPU-bake inputs,
including static people at density 0.25 and three square-aspect cameras. Its
reported rate is **CPU-prepared rooms/s**, not completed rendered rooms/s. One
room warms shared assets before timing. `--fingerprints` records diagnostic mesh,
map and manifest identities for comparison and suppresses the rate because hashing adds work.
Set `--asset-root` when the project assets live elsewhere. Timings for individual
stages overlap and must not be added as exclusive CPU work. The normal JIT path
also emits per-seed preparation timings when
`RUST_LOG=bevy_zeroverse::preparation=debug` is enabled, without changing dataset
metadata.

The [earlier WGPU qualification receipt](qualification/jit_transport.json) records
matched 64-room runs for generator 30, hardware, quality and timing variation. One controlled
CPU-affinity pair measured 2.98 GPU versus 2.37 CPU rooms/s; unrestricted repeats
varied substantially. These measurements do not establish a repeatable twofold
end-to-end speedup. Scene preparation and capture still dominate the consumer
reduction. The receipt includes all repeated measurements, not only the fastest
run. These are historical measurements, not a throughput promise for the current
generator or CUDA transport. CPU affinity is a benchmark control, not a library
tuning requirement.

## Copy and device contract

The native `gpu_tensor` API imports Bevy's **existing** wgpu instance, adapter,
logical device and queue into Burn. It does not create a second logical device.
Use `rooms.device()` for training; constructing an unrelated `Device::wgpu(...)`
and calling `to_device` is outside this same-device contract.

The render attachments are GPU textures; Burn tensors are linear GPU buffers.
These are different storage representations. Each attachment therefore takes one
GPU texture-to-buffer copy directly into the Burn allocation, not a readback and
re-upload. At 512x512, RGBA rows and plane offsets are aligned, so extracting each
view/plane shares that allocation. Odd widths can require an additional GPU
packing kernel to remove row padding. No image bytes cross the host boundary.
Eliminating the texture-to-buffer copy would require a different rendering/output
representation, not simply importing a texture as a normal Burn tensor.

The archived/CPU `LiveDataset` API intentionally returns host images and does
perform readback. It is appropriate for archives, not the lowest-copy JIT path.
`GpuSample.metadata` contains calibration/scene metadata with empty image vectors;
`gpu_tensor_contract.transport` records the active transport and copy/fence contract.

## Opt-in Linux Vulkan → CUDA

The [0.37.2 / adapter 0.21.0 qualification receipt](qualification/gpu_interop.json)
records annotation parity, allocation lifetimes, fusion and feature isolation.

Enable `bevy_zeroverse_burn/cuda_interop` and use:

```rust,no_run
# #[cfg(all(feature = "cuda_interop", target_os = "linux"))]
# fn example() -> anyhow::Result<()> {
use bevy_zeroverse::{app::BevyZeroverseConfig, scene::ZeroverseSceneType};
use bevy_zeroverse_burn::gpu::GpuLiveDataset;
let mut rooms = GpuLiveDataset::new_cuda(BevyZeroverseConfig {
    scene_type: ZeroverseSceneType::ProceduralIndoor,
    num_cameras: 3, width: 512., height: 512., playback_steps: 1,
    ..Default::default()
})?;
let training_device = rooms.device().clone(); // CUDA, matched to Vulkan by UUID
let sample = rooms.next_sample()?;            // ordinary owned Burn CUDA tensors
# Ok(())
# }
```

`GpuLiveDataset::indoor_cuda()` provides the same three-view 512x512 preset as
`indoor()`. Both constructors automatically bound future capture and retain
full native scene quality. Keep the dataset alive across batches and shards;
initialize the model and optimizer on the returned `device()`. Existing CUDA
models must use the matching device/context; use a checkpoint when restarting a
training process onto this transport rather than moving image tensors through
host memory. Measure complete training updates, input waiting and memory use
before choosing CUDA over WGPU: the extra GPU copy and stream fences can offset
the removed host transfer.

This experimental native path uses published Bevy 0.20, wgpu 30, Burn 0.22 and
CubeCL 0.11 APIs. **There are no dependency patches, vendored GPU dependencies,
or CPU fallback.** CUDA/Vulkan dependencies and initialization callbacks are
optional and Linux-only; the default and WASM graphs remain unchanged. The
CUDA toolkit selection follows CubeCL rather than pinning a toolkit version or
forcing its linking mode. The threaded JIT dataset API itself remains native-only. Unsupported Vulkan adapters,
missing external-memory/semaphore support, mismatched UUIDs/contexts and failed
captures return errors rather than silently transferring through host memory.

The producer uses dedicated device-local Vulkan buffers exported as opaque FDs,
imports/maps those allocations in the matching CUDA primary context, and pools
these allocations. Vulkan queue-family release/acquire barriers and an exported
binary semaphore order the two APIs. Fresh allocations are zeroed on the GPU
before their first import into wgpu. File descriptor ownership, imports and
allocator reservations are retained explicitly until work completes.

Current CubeCL does not expose foreign-allocation ownership or an external
readiness-event hook for ordinary CUDA tensors. Consequently this path uses:

1. Bevy texture → shared Vulkan/CUDA buffer (GPU copy).
2. Shared CUDA mapping → Burn-owned CUDA allocation (GPU copy).
3. Producer-side allocation/transfer stream fences before exposing the tensors.

There is no GPU → CPU → GPU **image** round trip. This is **not zero-copy** and is
not fully asynchronous with respect to the producer CPU. It does not impose a
CUDA device/context-wide synchronization. One future scene is still prepared
by the bounded worker while the consumer trains, and consumers can retain tensors
across regenerations, worker shutdown and other CUDA streams. A failed partial
batch poisons its bridge to prevent unsafe buffer/semaphore reuse.

Fully asynchronous zero-copy CUDA tensor import would require an upstream
CubeCL allocation lease and stream-readiness API. A Cargo `[patch]` cannot be
made conditional on a feature, so no global patch has been added to approximate
that missing contract. Benchmark this opt-in path on the intended hardware;
removing host transfers alone does not establish higher training throughput.

```sh
cargo test -p bevy_zeroverse_burn --features cuda_interop --lib cuda_transport -- --ignored --test-threads=1
cargo run -p bevy_zeroverse_burn --features cuda_interop --example gpu_jit_bench -- --cuda --scenes 64
```

API references: [NVIDIA external resource interoperability](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html#external-resource-interoperability),
[wgpu HAL buffer import](https://docs.rs/wgpu/30.0.1/wgpu/struct.Device.html#method.create_buffer_from_hal),
[Bevy Vulkan initialization callbacks](https://docs.rs/bevy_render/0.20.0/bevy_render/renderer/raw_vulkan_init/struct.RawVulkanInitSettings.html).
