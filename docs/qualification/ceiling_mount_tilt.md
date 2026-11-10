# Ceiling camera mounting tilt

Ceiling cameras sample an optical-axis offset independently of their supporting
plane. The default `ceiling_mount.tilt_degrees = [0, 12]` uses an isotropic,
bounded Rayleigh radial prior: small mounting errors are more frequent than
large ones, without a preferred azimuth. This is a designed augmentation prior,
not an empirical fit to installed cameras. Ceiling slope is additional.

The ordered bounds support 0–45 degrees. `[0, 0]` gives exact normal alignment;
equal nonzero bounds give fixed magnitude and varying azimuth. Tilt is sampled
once per seed and slot, independently of lenses, image clocking and placement
retries. Rail motion keeps that orientation. Increasing the camera count does
not redraw an existing slot's tilt.

Studio exposes both bounds under **Cameras → Ceiling mix & mounting**. The same
typed settings work in `--indoor-camera` JSON and sparse query parameters such
as `camera.ceiling_mount.tilt_degrees.1=18`. Default values stay out of the URL.
`CeilingCameraMount::optical_axis()` and `tilt_degrees()` expose the realized
orientation; archived rotations remain authoritative for rendering, calibration,
annotations and gizmos.

## Physical and sampling contract

- Surface contact, standoff and rails follow the actual roof or soffit plane.
  Mounting tilt changes the optical axis, not the supporting plane.
- Near-field clearance, downward content and calibrated overlap use that actual
  optical axis. Tilted pinhole near-plane corners and rail sweeps must remain
  inside the envelope. Fisheye near spheres also check plane-normal standoff,
  rather than relying only on vertical clearance under a sloping roof.
- Mixed rigs retain requested lenses, motion roles, travel, baselines and overlap
  admission. When neighboring-room placements are allowed, coupled room-camera
  proposals mix primary-room and unrestricted sampling; the requested policy is
  restored after each proposal. This primary-room preference is disabled for
  independent/negative-pair policies.
- Bounded recovery can assemble previously qualified paths after its budget for
  new visibility tests is exhausted. It tries small complete candidate domains
  before expanding them. Search budgets and admission thresholds are unchanged.
- No new render passes, per-frame random sampling, model loads or dependencies
  are introduced. A zero ceiling-camera probability retains room-only behavior.

## Qualification

Local source identity: generator **116**, `capture-v141`, `ceiling_camera=6`.
Camera qualification is in `out/ceiling_tilt_20261109/`; release-source checks
and the refreshed matched measurement are in `out/release_0630/`. The complete
local core suite passed 1,221 tests plus 14 isolated voxel fixtures. All four
inherited CI failures are covered: explicit primary-room population fixtures,
512-room coverage for optional area rugs, functional seating relationships, and
recovery idempotence on a fixed furniture state. Recovery streams are independent
per sofa, so skipping a completed group cannot perturb later candidates.
Registry and CI status are recorded separately in the release receipt.

The 120 ceiling-camera tests, Studio sharing test and explicit matched timing
test passed. This includes:

- 384 all-ceiling rig configurations across 32 seeds and 1–12 cameras: 2,496
  mounted cameras, including sloping roofs and flat soffits.
- 96 mixed rig configurations across eight seeds and 1–12 cameras: 624 cameras
  covering every room/ceiling × pinhole/fisheye × static/moving combination.
- Sixteen additional placement configurations exercising exact alignment and
  stronger `[10, 35]` tilt ranges with both lens and motion families.
- Seed/slot repeatability, manifest round trips, projection/ray round trips
  within 0.002 pixels at five times along mounted paths, and world-yaw transforms.
- Mid-rail obstruction, tilted near-plane/roof intersection, sloping-roof
  near-sphere clearance, invalid settings and transactional failure tests.
- Existing rejected-training-seed regressions and sixteen full mesh checks,
  with zero degenerate triangles in that mesh subset. Seed 65573956 retains its
  full eight-camera mixed rig and its explicitly unrestricted room policy.

Across 16,384 independent prior samples, tilt p05/median/p95 is
**1.25° / 4.62° / 9.50°**. Eight equal azimuth bins contain
`[2092, 1988, 2079, 2068, 2107, 1972, 2076, 2002]` samples. In the all-ceiling
population, realized tilt p05/median/p95 is **1.30° / 4.86° / 9.96°**.
These are finite seed cohorts, not a claim of unrestricted population coverage.

Strict workspace Clippy with CUDA IPC/WGPU fusion features, the WASM library
check with `web,human_motion`, formatting and diff checks passed. Cargo still
reports an upstream `burn-cubecl-fusion 0.22.1` future-incompatibility notice.

## CPU placement cost

Matched current-code controls reuse identical generated geometry for eight seeds
at each camera count, alternating aligned/tilted order. Room and ceiling motion
fractions are 0.65, ceiling probability is 0.5, and both lens families are sampled.
Measurements use the optimized test profile on a shared host, with two worker
threads. They time camera resampling, excluding mesh/material preparation,
rendering and readback. No samples were discarded.

| Cameras | Tilt range | Mean | Median | Maximum | Failures |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3 | 0° | 53.1 ms | 23.6 ms | 155.5 ms | 0/8 |
| 3 | 0–12° | 62.7 ms | 20.8 ms | 254.0 ms | 0/8 |
| 8 | 0° | 811.3 ms | 172.0 ms | 3,468.2 ms | 0/8 |
| 8 | 0–12° | 930.9 ms | 190.8 ms | 5,016.5 ms | 0/8 |
| 12 | 0° | 1,209.5 ms | 449.0 ms | 5,128.8 ms | 0/8 |
| 12 | 0–12° | 1,162.8 ms | 411.7 ms | 4,911.1 ms | 0/8 |

The separate eight-camera seed 65573956 passed the aligned/tilted controls in
3.82/3.42 seconds of camera resampling. Three-view medians are 21–24 ms and means
53–63 ms in this small cohort. Larger rigs retain multi-second search tails and
the eight-view tilted mean is higher. This does not establish a general
throughput improvement or a worst-case latency guarantee.

Qualification is CPU geometry/calibration and compile coverage. It does not
include fresh rendered RGB/co-visibility comparisons, browser-device execution,
CUDA replay, GPU throughput or concurrent-training memory qualification.

Reproduce the focused checks with:

```sh
cargo test --locked --lib scene::procedural_indoor::cameras::ceiling -- --test-threads=2
cargo test --locked --lib app::editor::tests::ceiling_mount_controls_preserve_walkthrough_policy_and_share_compactly
cargo test --locked --lib ceiling_tilt_placement_qualification -- --ignored --nocapture --test-threads=1
```
