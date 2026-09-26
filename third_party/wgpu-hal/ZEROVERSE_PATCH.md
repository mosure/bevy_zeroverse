# Local patch to published wgpu-hal 29.0.4

Upstream source: https://crates.io/crates/wgpu-hal/29.0.4 (MIT/Apache-2.0).
Bevy and Burn remain on their published releases. No shader or rendering equation
is changed by this patch. All non-Vulkan backends and `MemoryHints::Performance`
retain upstream behavior.

Heaptrack on NVIDIA 610.43.02 found nearly 1 GB of live host allocations from
`vkAllocateCommandBuffers` after 128 regenerated indoor scenes. Upstream allocates
16 command buffers at a time per encoder, caches all of them, and resets pools
without releasing their driver storage. A diverse sequence distributes costly
passes across the entire encoder pool even after the underlying assets are freed.

With `MemoryHints::MemoryUsage`, allocate command buffers singly. On every completed
reset of an encoder, retire its idle command pool and cached temporary image views.
A 32-reset cadence still retained substantial historical command storage in the
least-reused encoders and did not meet the continuous-process memory growth gate.
A `RELEASE_RESOURCES` reset alone did not reclaim driver command metadata in the
measured NVIDIA configuration and was rejected after a regression trial.
The replacement pool is allocated before retiring the old one; allocation failure
falls back to the ordinary reset. The existing `reset_all`
safety contract guarantees that all submitted buffers have finished; this adds
no queue wait, device idle, extra submission, or synchronization point.

A second profile isolated approximately 120 KB/scene of retained NVIDIA
allocations in indirect-draw validation. Private descriptor pools did not fix
this and were rejected. Native capture cameras now use Bevy's supported direct
draw path with GPU preprocessing. Validation remains enabled. This application
policy does not change this HAL patch or interactive cameras.

The application selects MemoryUsage for image capture workers. Remove this local
patch when an upstream release provides equivalent bounded command retention.
Runtime evidence and limitations are recorded in the indoor qualification report.

The accompanying wgpu-core patch caps the outer cache at 64 idle encoders under
MemoryUsage. Retiring an encoder's native pool alone does not bound the number
of idle encoder objects and their temporary arrays.

Root Cargo patches are not inherited by downstream published-crate consumers;
consumers need both patches until equivalent upstream fixes are available.
