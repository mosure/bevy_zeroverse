# Motion vectors and optical-flow annotations

Native dataset capture supports `optical-flow` and `motion-vectors` for both
static and animated geometry. These are numeric surface correspondences, not
the viewer's color visualization. Human motion is optional; camera trajectories
alone also produce flow.

```sh
cargo run -p bevy_zeroverse_burn --features human_motion --bin zeroverse_gen -- \
  --output out/flow --samples 1 --workers 1 --seed 1 \
  --scene-type procedural-indoor --indoor-density 0.35 --indoor-human-density 0.7 \
  --width 384 --height 288 --cameras 2 --playback-steps 5 --playback-step 0.25 \
  --render-modes color depth normal semantic optical-flow motion-vectors \
  --human-motion '{"fraction":0.7,"max_actors":4,"frames":120,"batch_size":2}' \
  --timeout-secs 900 --no-ui
```

Omit `--features human_motion` and `--human-motion` when model-generated actors
are not needed. Enable flow in the configuration before creating capture
cameras. Adding it later to a camera without temporal attachments fails with a
configuration error instead of returning a colored image as ground truth.

## Coordinate and time contract

For pixel center `p=(x+0.5,y+0.5)` in image `I[t,camera]`, the corresponding
surface point projects to `p + optical_flow[t,camera,y,x]` in `I[t+1,camera]`.
Positive x is right; positive y is down. Values are signed **pixels per captured
interval**, not pixels per second. `motion_vectors` contains exactly the same
displacement divided by `(width,height)`. Nothing clamps, normalizes by observed
magnitude, tone-maps, or compresses these values with an image codec.

The source and target cameras use their own captured projections and world
transforms, including FOV changes. Mesh correspondences include entity motion,
scene/parent transforms, same-topology vertex deformation, and Bevy four-weight
skeletal skinning. Indoor actors' clothing, hair and accessories participate.
Both endpoints use the same controlled `Playback` samples as RGB and poses.
Warm-up, shader compilation, modality switching, paused capture cameras and
asynchronous readback do not advance or discard correspondence history.

Each vector field has two masks:

- `valid`: the source pixel hits geometry with a surviving vertex correspondence,
  and its target is in front of the target camera between its near/far planes.
  A target outside the image can still have a valid, unbounded vector.
- `visible`: the valid target is inside the image and passes the target-surface
  visibility test. Occluded and out-of-frame correspondences have `visible=0`.

Use `valid` for all surface correspondences or `visible` for visible matching
supervision. Background without geometry has both masks zero. Despawned or
replaced meshes and changed index topology have no asserted correspondence.
The final timestep has no successor: its vectors and masks are all zero. A
one-timestep capture is therefore a valid all-zero flow export. Static scenes
with static cameras have zero displacement and valid foreground masks.

## Storage

Safetensors stores `optical_flow` and/or `motion_vectors` as float32 arrays with
shape `[batch,time,camera,height,width,2]`. Each present field has uint8
`<field>_valid` and `<field>_visible` arrays with the same leading dimensions and
one final channel. The `flow_metadata` uint8 JSON tensor records the convention
and schema version. The existing `time` tensor specifies the sampled endpoints.

The filesystem format stores lossless `<field>_TTT_CC.npz` arrays shaped
`[height,width,4]`, ordered `(dx,dy,valid,visible)`, and `flow_metadata.json`.
Rust `View` and Python `View.optical_flow` / `View.motion_vectors` expose this same
four-channel float32 layout as bytes. Legacy optical-flow RGB/JPEG exports are
visualizations; loaders reject them as numeric flow rather than silently
interpreting their colors as vectors.

## Rendering and limitations

Flow uses a separate RGBA32Float raster pass on the source geometry, with target
positions interpolated using source barycentric coordinates. Target visibility
uses the current float32 geometric attachment. The pass compares the target
texel's world position with the transported surface's tangent plane, using
`max(1 mm, 0.0001 * target_view_depth)` tolerance. This avoids false occlusion from
comparing depths along different rays on sloped planes, but silhouette and thin
surface boundaries remain subject to raster resolution and this tolerance.

The geometric surface policy matches depth/normal/semantic capture: glass is
treated as a first surface. These annotations do not represent reflected,
refracted, shadow, texture-animation, or other purely photometric motion. Custom
vertex-shader displacement and unbaked morph targets are not silently inferred;
unsupported morph geometry fails capture. Motion blur and transparency-layer
flow are outside this contract.

All synchronized cameras share one geometry pair. History is bounded to one
previous snapshot, resets at sequence boundaries, and releases when flow
cameras disappear. GPU pipelines and temporal buffers are created lazily only
for requested flow. Readback uses the existing bounded asynchronous capture
transport. The native renderer performs numeric sequence export. Native and
WebGPU viewer modes remain engine-frame velocity color previews; their colors
are not dataset annotations.

## Validation commands

[Measured GPU, indoor capture and codec results](optical_flow_validation.md).

```sh
cargo run --features human_motion --example validate_optical_flow
cargo test -p bevy_zeroverse_burn --features human_motion --test flow_roundtrip
cargo run --features human_motion --bin motion_validate -- \
  --seeds 0 --render-seed-list 1,2,13 --flow --output out/flow_motion
```

The example runs the same reference as the ignored `optical_flow_render`
integration test, without building every application binary. The GPU reference
tests stationary flow, signed rigid displacement, camera pose
and FOV changes, vertex deformation, skeletal deformation, occlusion, topology
changes, despawn, inactive-camera intervals and sequence reset. Codec tests preserve signed subpixel values
and both masks through raw/compressed safetensors and NPZ, including captures
without RGB. The indoor audit checks camera-only reprojection on static surfaces
and requires every eligible static surface to retain its correspondence. It also
reports person motion remaining after removing camera motion.

## Interactive preview

The native/WebGPU viewer scales Bevy's per-frame motion vectors by the actual
frame duration and a configurable reference interval (default 0.05 seconds).
The **Flow preview interval** and **Flow color scale** controls are in Rendering
and annotations. A constant image velocity therefore keeps the same color at
30, 60 or 120 FPS. Pause produces zero flow; regeneration and playback wrap
suppress invalid temporal history for two frames.

This preview is a backward-difference velocity estimate displayed on the current
image, not an exact finite-interval correspondence field. Fast acceleration,
visibility changes and low frame rates can still affect it. Dataset optical flow
and motion vectors continue to use exact captured endpoints and their validity
and visibility masks; preview settings do not alter those exports.
