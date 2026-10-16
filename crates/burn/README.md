# bevy_zeroverse_burn

Full-quality procedural indoor datasets for Burn 0.22 and Bevy 0.20.

For native JIT training, retain one `gpu::GpuLiveDataset` and construct your
model and optimizer on its `device()`. A persistent worker prepares one future
sample automatically while training runs. Scenes retain shadows, indirect
lighting, procedural materials and float32 geometric annotations.

Low-resolution RGB defaults to 2× SSAA: Auto (`ssaa_factor = 0`) applies below
1280×720 pixels, including the three-view 512×512 preset. The GPU area resolve
preserves final tensor shapes and transfer size. Numeric geometric labels remain
at the exact final pixel-center rays. Set `ssaa_factor = 1` to disable it, or
2/3/4 to force a factor. `sample.metadata.sampling` records the resolved policy.

| Backend | Feature | Three-view 512x512 preset | Transfer |
| --- | --- | --- | --- |
| WGPU | `gpu_tensor` | `GpuLiveDataset::indoor()` | One texture-to-buffer GPU copy on Bevy's existing device and queue |
| CUDA (Linux NVIDIA) | `cuda_interop` | `GpuLiveDataset::indoor_cuda()` | Vulkan external GPU memory, then one additional copy into Burn CUDA storage |

Neither path transfers image bytes through the CPU. CUDA currently fences
producer streams before returning; it is experimental, not zero-copy, and does
not guarantee a throughput improvement over WGPU. It verifies GPU UUIDs and
allocation contexts and fails on unsupported hardware instead of using a host
fallback. All dependencies are published; no patches or vendored forks are needed.

```toml
[dependencies]
bevy_zeroverse_burn = { version = "0.28", features = ["gpu_tensor"] }
```

```rust,no_run
use bevy_zeroverse_burn::gpu::GpuLiveDataset;

let mut rooms = GpuLiveDataset::indoor()?;
let training_device = rooms.device().clone();
let sample = rooms.next_sample()?;
// sample.views[0].color: float32 [height, width, 4], tonemapped linear RGBA.
// Construct model and optimizer on training_device; keep tensors on that device.
# Ok::<(), anyhow::Error>(())
```

For CUDA, enable `cuda_interop` and replace `indoor()` with `indoor_cuda()`.
`new(config)` and `new_cuda(config)` accept explicit camera, scene and annotation
settings. `into_dataset(samples_per_epoch)` adapts either stream to Burn's
dataset interface. If the model uses Burn fusion, also enable
`gpu_tensor_fusion`. Static people do not initialize motion inference models.

Set `BEVY_ASSET_ROOT` to the directory containing `assets/burn_human` for rooms
with AnnyBody people. Missing assets return an error. CPU archives, filesystem
datasets, O-voxels and browser captures use the existing export APIs.

The [tensor contract and benchmarks](https://github.com/mosure/bevy_zeroverse/blob/main/docs/gpu_jit.md)
describe channel layouts, glass policy, flow, co-visibility, retained tensor
lifetimes, copy counts and measurement limitations. Check completed training
steps and input starvation when evaluating throughput; GPU capture submission
time alone does not measure completed generation or model training.

For inventory-controlled archive generation, use
`generator::run_chunk_generation_controlled(config, Arc<CaptureControl>)`.
Cumulative credits admit whole rooms; hold/cancel use interruptible waits outside
the capture mutex. Controlled waits allow dispatched CPU work and export to finish
without idle GPU preflight. The ordinary live path automatically preflights one
known successor while the consumer is busy. See the
[capture scheduling contract](../../docs/capture_admission.md) for migration,
terminal receipts, diagnostics and optional process CPU budgets.
