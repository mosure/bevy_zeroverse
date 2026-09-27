# Optical-flow annotation validation — 2026-09-27 UTC

This qualification tests numeric forward surface correspondence at dataset
capture times. It uses an RTX PRO 6000 Blackwell 96 GiB, NVIDIA 610.43.02 and
native Vulkan. See the [coordinate/storage contract](optical_flow.md) and
[machine-readable evidence](optical_flow_validation.json).

## Controlled GPU reference

The analytic reference independently projects source world positions into the
target camera. It checks 6,097 pixels in each of five cases: stationary geometry,
signed rigid translation, camera translation/rotation and FOV changes, vertex
shear, and skeletal deformation. Maximum endpoint error is below 0.00002 pixels.
The test also checks occlusion masks, out-of-frame correspondences, behind-camera
targets, changed triangle indices, removed objects, and sequence resets.

Every capture includes paused-camera intervals and unrelated renderer updates.
This reproduces the production optimization that disables cameras while GPU
readback is pending. An initial implementation incorrectly discarded history
when the temporary `ExtractedView` disappeared. The corrected implementation
retains history while the capture camera still exists; the regression fails if
any expected source correspondence disappears. A separate sequence-reset test
covers changing the sequence during an idle interval before the next request.

## Animated procedural interiors

Seeds 1, 2 and 13 are captured with real ARDY motion, two cameras, five times
`0, 0.25, 0.5, 0.75, 1`, and resolution 384×288. These are 30 views containing
24 temporal pairs and six terminal views. The selected scenes include admitted
walking, entering, exercise and reaching motions, alongside static people.

All 2,654,208 nonterminal pixels retain valid surface correspondence. The audit
independently checks expected validity on static surfaces before applying the
flow mask, so missing or all-zero intermediate fields cannot pass unnoticed.
No expected static correspondence is missing. Maximum per-view static
camera-reprojection p99 error is 0.002447 pixels. Both vector representations
agree, masks are binary, visible implies valid, and every terminal field is zero.

After subtracting camera-only projection, 17,933 person-pixel observations exceed
0.05 pixels of residual motion. This establishes that flow includes actor
deformation; it is not a semantic action-quality score. RGB, depth, normal,
semantic and human-pose checks also pass. One model pair is reused across the
three rooms. Indirect lighting is disabled in this diagnostic room sweep.

[RGB source frame](evidence/optical_flow/walking_rgb.png) and
[forward-flow preview](evidence/optical_flow/walking_flow.png) show one walking
capture. Preview hue encodes direction; saturation uses a fixed 40-pixel scale;
hidden endpoints are dimmed. These PNGs visualize the separate float32 data.

## Dataset interoperability

The production CLI additionally exports a three-timestep, two-camera indoor
sample with RGB, depth, normals, semantics and both temporal modes. Its archive
is decoded independently with safetensors and with the Python dataloader.
All temporal slots, signed components, float32 values, uint8 masks and normalized
displacements are checked. A separate one-timestep Cornell capture exports only
the two flow modes: its intentionally all-zero fields must complete successfully
without RGB or human-motion model loading.

Rust codec tests cover raw/compressed safetensors and NPZ, signed subpixel values,
occluded/terminal masks, flow-only datasets and malformed values. Python tests
cover live-style tensor conversion, chunk/folder roundtrips, global convention
metadata, legacy-color rejection and indoor configuration.

Local artifacts are under `out/flow_motion`, `out/flow_cli_final` and
`out/flow_terminal`; their hashes and measured counts are in the evidence JSON.
`out/flow_cli` predates the paused-camera fix and is diagnostic output, not a
qualified dataset.

## Limits

This is a numerical and integration qualification, not a photographic-realism,
throughput or long-run memory benchmark. Visibility uses the documented tangent
plane/raster tolerance and is resolution-dependent at silhouettes. Glass follows
the same first-surface geometry policy as depth and semantics; reflection,
refraction, shadows and texture-only changes are not optical correspondences.
Unsupported morph targets fail capture; custom shader vertex displacement must
be baked into geometry.

Numeric sequence export uses the native capture renderer. The WebGPU viewer
build passes, but its velocity colors remain a frame-based preview; no browser
numeric export or fresh browser runtime qualification is claimed here. Strict
workspace/all-target Clippy passes; Cargo still reports the upstream
`burn-cubecl 0.21` future-compatibility notice.

## Reproduction

```sh
cargo run --features human_motion --example validate_optical_flow
cargo test -p bevy_zeroverse_burn --features human_motion \
  --test flow_roundtrip --test indoor_roundtrip
cargo run --features human_motion --bin motion_validate -- \
  --seeds 0 --render-seed-list 1,2,13 --flow --output out/flow_motion
cargo clippy --workspace --all-targets --features human_motion -- -D warnings
CARGO_PROFILE_DEV_DEBUG=0 cargo check --target wasm32-unknown-unknown \
  --bin viewer --no-default-features --features web,human_motion
python -m unittest discover -s crates/ffi/python -p 'test_flow.py'
python -m unittest discover -s crates/ffi/python -p 'test_indoor.py'
cargo run -p bevy_zeroverse_burn --features human_motion --bin zeroverse_gen -- \
  --output out/flow_cli_final --samples 1 --workers 1 --chunk-size 1 --seed 1 \
  --scene-type procedural-indoor --indoor-density 0.35 --indoor-human-density 0.7 \
  --indoor-gi-rays 64 --width 128 --height 96 --cameras 2 \
  --playback-steps 3 --playback-step 0.5 \
  --render-modes color depth normal semantic optical-flow motion-vectors \
  --ov-mode disabled --timeout-secs 900 --no-ui \
  --human-motion '{"fraction":0.7,"max_actors":4,"frames":120,"batch_size":2}'
target/debug/zeroverse_gen --output out/flow_terminal --samples 1 --workers 1 \
  --chunk-size 1 --scene-type cornell-cube --width 97 --height 73 --cameras 2 \
  --playback-steps 1 --render-modes optical-flow motion-vectors \
  --ov-mode disabled --timeout-secs 300 --no-ui
```

The Python tests require the dataloader dependencies and an importable FFI module.
They test conversion and codecs; the live render/export qualification is the
native CLI, not a rebuilt Python extension.
