# Generated-scene rejection contract

The CPU preparation boundary distinguishes an unusable sampled room from a
broken capture producer. It does not weaken geometry or camera validation.

| Failure source | Result |
| --- | --- |
| External densities, dimensions, camera count or JSON settings | Fatal `CaptureError::Failed` before construction |
| Fresh generated layout, appearance, camera placement, validation or motion planning | `CaptureError::Rejected`, with seed and stage diagnostic |
| An already typed rejection | Original category, reason and retirement state |
| Loaded-manifest integrity, renderer, GPU, readback, IPC or retirement | Fatal/separate; never routed through the generated-scene conversion |

Previously untyped generated failures use the existing `PlacementClearance`
category. There is no message allowlist. In particular, a valid explicit motion
request referencing an actor absent from the generated room is a scene-local
failure; malformed motion configuration is a fatal input error. Human and indoor
object motion settings are both parsed before generation. Interactive editor-only
scenes retain their reference camera when no capture views are requested; a
headless capture requires 1–256 views and a finite positive raster aspect.

`preparation::prepare_indoor_typed` exposes this distinction to CPU preparation
consumers. Its rejection has `retired: false`. The existing `prepare_indoor`
string-returning API remains available. Neither function retires a live renderer
transaction. The native capture runtime must cancel owned preparation, discard
speculative leases, clear scene/readback state and advance the consumed attempt
before reporting `retired: true`. Retirement failure stops the lane. Admission
credits, cancellation and rejection policy remain controlled by the existing
capture API; this change adds no rejection-count limits or automatic retry loop.

Retirement cancels queued human-motion work and invalidates the running epoch
without a GPU wait. It clears scene clips while preserving the plugin's report
resource and model cache. This prevents an idle motion runtime from panicking
on the update after a rejection, and prevents late inference from updating a
successor's scene.

The conversion lives only in `preparation/pipeline/generated.rs`. Direct manifest
validation and the render/transport paths are unchanged. Seed streams, accepted
manifest generation, geometry/material construction and rendering are unchanged;
the generator version remains 120. Capture-engine identity changes to distinguish
the producer's error-handling contract.

## Validation

The focused CPU suite checks malformed and out-of-range external settings,
including indoor motion, the valid historical seed 66242909, deliberately invalid
generated data versus direct manifest validation, all existing rejection
categories, stage diagnostics and wire round trips. It also checks fatal retirement
of a mismatched transaction without destroying the current scene.

`tests/capture_recovery.rs::generated_cpu_failures_retire_without_mutating_retained_samples`
uses real native rendering and readback. Two valid motion conditions that cannot
be realized in their sampled rooms must produce retired rejections without
publishing a sample. Captured RGB, calibration and annotations retained across
those rejections must survive unchanged; a valid successor and repeated capture
must match. The test supports hardware Vulkan or a bounded software Vulkan
adapter. Software results do not qualify CUDA IPC, GPU memory stability, or
concurrent-training performance.

Release receipts, an unpatched registry consumer, and accepted-manifest/mesh
comparisons with the preceding published source are kept in `out/release_0635/`.
No downstream workspace, checkpoints or training controller are modified.
