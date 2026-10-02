# bevy_zeroverse 0.27

This release adds explicit camera calibration and time contracts, optional
handheld motion and overlap mixtures, independent appearance controls, and
recorded RGB sensor transforms. The project page and paper share a Rust
qualification pipeline and a single interactive co-visibility gallery.

Registry versions: `bevy_zeroverse` and `bevy_zeroverse_ffi` **0.27.0**,
`bevy_zeroverse_burn` **0.10.0**, and the first releases of
`bevy_zeroverse_capture` and `bevy_zeroverse_publication` **0.1.0**.
`burn_siglip2` remains **0.1.1**.

## Compatibility

- Capture identity **v34**, architectural generator **22**. Start new shards
  rather than resuming captures from an older engine identity.
- View/config struct literals need the new fields. Archived JSON without the
  optional controls retains existing defaults. Legacy uncalibrated datasets
  remain readable; mixed or malformed calibration is rejected.
- Calibration includes a full pixel-space K, image dimensions, versioned
  centered pinhole model and half-pixel-center convention. Lens distortion and
  off-center rendering are not introduced. Calibrated Python image loading
  rejects silent crops that would invalidate K.
- Physical seconds are available only with an explicit trajectory duration;
  raw playback time and eased trajectory progress remain separate.
- Pair strata are seeded before placement. Proxy overlap, rendered directed
  overlap, metric baseline and shared-surface triangulation estimates remain
  distinct. No post-inference filtering is performed.
- RGB-only sensor operations preserve geometric annotations and calibration.
  JPEG sensor simulation requires raw tensor storage to avoid a second lossy
  encoding stage.

## Qualification

`publication.toml` specifies the 512-room layout audit and 32-room rendered
gallery. The additional matched qualification suite covers 11 camera, aspect,
material and illumination factors over two geometry families (88 images).
Receipts record actual capture provenance, artifact hashes, self-reprojection,
directed overlap and exact duplicate counts. The generated page and paper
report measurements from those receipts.

Validation includes native camera/placement tests, annotation/calibration
round trips, individual sensor-factor checks, FFI and Wasm compilation, a
headless multi-timestep CLI capture, browser gallery checks and publication
contract tests. Registry packages are validated without checkout WGPU patches.
The small factor suite is a correctness qualification, not evidence of broad
transfer, photographic realism or unlimited-process memory stability.
