# Mixed camera recovery: seed 71712897

The published 0.63.0 generator validates this room's layout and meshes, but the
production eight-view policy exhausts camera placement after finding six views.
The downstream typed rejection is `CameraPlacement`, not invalid geometry.
The archived production settings match
`cameras/ceiling/tests/mixed_training_camera_settings.json`: eight moving
512×512 views, baseline 0.5, mixed mounting and lens families, no people,
density 0.65, with neighboring-room placements permitted. Slot 6 is the ceiling
fisheye; all eight requested motion roles remain unchanged.

## Cause and correction

Recovery reuses individually valid trajectories across camera slots. Several
slots can therefore have the same first reference-compatible path. Those
singleton domains cannot form a valid rig because their cameras coincide or
lack relative motion. Searching singleton domains for every reference before
widening any of them consumed the calibrated-visibility budget before useful
alternatives could be assembled.

The solver now widens a reference's candidate domains before advancing to the
next reference. Each reference receives at most 512 new visibility evaluations
per recovery pass, within the existing shared budget and the existing reserves
for subsequent passes. Previously qualified visibility decisions remain reusable
after the budget is exhausted. Assignment/backtracking and every admission
oracle are unchanged.

This changes neither room geometry/materials nor requested camera settings.
It does not shorten paths, substitute lenses, reduce camera count, weaken
clearance/spacing/overlap, or increase the total search budgets. Successful
ordinary placement still bypasses this fallback. Generator 117 and
`capture-v142;ceiling_camera=7` identify the changed camera recovery behavior.

## Qualification

Evidence and commands are in `out/seed_71712897/`. The CPU regression checks
complete eight-camera placement, the original lens/motion assignments, full
mesh validation, swept path clearance, calibrated overlap at five normalized
check times (0, 0.25, 0.5, 0.75, 1), configured intrinsic support, geometry/appearance preservation,
deterministic replay, and bounded work counters. A small synthetic constraint
fixture reproduces singleton-domain starvation without depending on a room
seed. Neighboring seeds are checked separately.

The test scope is CPU placement, geometry and calibration. It does not qualify
new rendered RGB/annotation comparisons, GPU throughput or training behavior.

The 123 ceiling/mixed-camera tests and seven baseline-contract tests pass,
including 96 baseline/rig combinations over baselines 0, 0.5 and 1 with 2, 4,
8 and 12 views. The explicit 16-room neighboring cohort (71712890–71712905)
passes full geometry and camera validation. All 15 previously successful rooms
retain exactly equal serialized camera configurations; the previously rejected
room now has all eight cameras and replays deterministically.

On this shared Linux host, optimized CPU placement for seed 71712897 takes
2.14–2.33 seconds, versus 10.62–10.90 seconds before rejection in the published
version. Mean placement-attempt time over the 16-room sample is 0.457 seconds
(0.970 seconds in the published control, including its rejected room). These are
bounded CPU placement observations, excluding mesh/material preparation,
rendering and readback, not an end-to-end GPU throughput claim.

Strict workspace Clippy (all targets, CUDA IPC/WGPU fusion features), the WASM
library check (`web,human_motion`), formatting and diff checks pass. No hardware
GPU work or downstream training changes were performed. The patch release is core/FFI 0.63.1, Burn bridge 0.47.1, capture 0.6.21 and
publication 0.9.9. Registry and deployment status belong to the release receipt.
