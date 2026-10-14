# GPU dependency sources

The workspace uses published Bevy 0.20.0 and Burn 0.22.0 with a shared registry
WGPU 30.0.1 dependency graph. There are no WGPU path patches or vendored GPU
crates. Local builds and published packages resolve the same GPU sources.

| Path | Dependency contract |
| --- | --- |
| Native viewer and capture | Registry Bevy and WGPU |
| WebGPU viewer | Registry Bevy and WGPU, without native CUDA interop dependencies |
| Burn `gpu_tensor` | Registry Burn/CubeCL sharing the Bevy WGPU device |
| Burn `cuda_interop` | Opt-in Linux interop using published Bevy `raw_vulkan_init`, Burn, CubeCL, Ash and Cudarc APIs |

The workspace's path dependencies are its owned crates under `crates/` and the
root generator. CUDA interop does not require a patched WGPU checkout. See the
[Burn integration guide](../crates/burn/README.md) for backend configuration and
transfer behavior.

Inspect the resolved native and browser dependency graphs with:

```sh
cargo metadata --locked --all-features --filter-platform x86_64-unknown-linux-gnu
cargo metadata --locked --no-default-features --features web,human_motion \
  --filter-platform wasm32-unknown-unknown
cargo tree --locked -i wgpu
```

The [archived WGPU 29.0.4 audit](evidence/wgpu_dependency_audit/review_29.md) and
its patch notes explain older measurements. They do not describe dependencies
required by the current generator or qualify current throughput and long-run
memory stability. Historical patch sources remain available in Git history.
