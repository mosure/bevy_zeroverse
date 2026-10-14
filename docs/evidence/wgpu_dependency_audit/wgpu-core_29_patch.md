> Historical patch notes for WGPU 29.0.4. This fork is no longer used.
> Current builds use registry WGPU 30.0.1; see the [dependency guide](../../wgpu_dependency_review.md).
> The original sources are preserved in [Git history](https://github.com/mosure/bevy_zeroverse/tree/91c5c96361970cc38867e2e78f27813fb92d833c/third_party).

# Local patch to published wgpu-core 29.0.4

Source: https://crates.io/crates/wgpu-core/29.0.4 (MIT/Apache-2.0).

For `MemoryHints::MemoryUsage`, retain at most 64 idle HAL command encoders.
Performance and Manual hints keep upstream behavior. Native capture selects
MemoryUsage; interactive rendering keeps Performance.

The upstream allocator retains every historical peak of concurrent encoders.
A 2,627-scene diagnostic retained 2,665 HAL encoders despite only a small live
command-buffer registry. Their temporary arrays and idle native command pools
continued to raise host memory after the large indirect-validation allocation
source had been removed. Capping the idle cache complements the Vulkan HAL
command-pool retirement patch; neither cache is allowed to retain arbitrary
historical peaks.

`InnerCommandEncoder::drop` calls `reset_all` before returning its encoder to
this pool. Excess completed encoders are dropped outside the allocator lock.
This does not limit active encoders, wait on the GPU, change validation flags,
alter rendering equations, or drop submitted work.

Root Cargo patches are not inherited by downstream crate consumers. The
checkout uses both patches until upstream offers equivalent bounded retention.
