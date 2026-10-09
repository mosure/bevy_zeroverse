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

The [current qualification receipt](qualification/jit_transport.json) records
matched 64-room runs, hardware, quality and timing variation. One controlled
CPU-affinity pair measured 2.98 GPU versus 2.37 CPU rooms/s; unrestricted repeats
varied substantially. These measurements do not establish a repeatable twofold
end-to-end speedup. Scene preparation and capture still dominate the consumer
reduction. The receipt includes all repeated measurements, not only the fastest
run. CPU affinity is a benchmark control, not a library tuning requirement.
