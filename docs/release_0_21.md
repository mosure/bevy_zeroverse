# bevy_zeroverse 0.21.0

This release brings generator 13 and capture-v20: optional generated human motion,
signed optical flow and motion-vector annotations, stricter headless capture
readiness, primary-room camera paths, broader procedural furniture, architecture,
clothing, glass and lighting. Human motion uses the published batched-inference
crates, deterministic per-actor seeds and compositional prompts. Static scenes do
not initialize motion models. The WebGPU deployment enables motion support.

Companion releases are `bevy_zeroverse_ffi` 0.21.0 and `bevy_zeroverse_burn` 0.4.0.
The independently versioned `burn_siglip2` 0.1.1 is now maintained and published
from this workspace; its optional embedding audit remains separate from scene
generation. Bevy remains 0.19.1 and Burn remains 0.21.0.

## Compatibility

- Indoor manifests, camera/object programs and capture metadata gained fields.
  Update Rust struct literals and exhaustive matches when upgrading.
- Start a new generation shard when changing capture identities; resume checks
  prevent mixing contracts. Existing datasets remain readable.
- Optical flow is signed pixel displacement to the next captured timestep;
  motion vectors divide that displacement by image width/height. Exports retain
  correspondence masks, and terminal frames have zero vectors and invalid masks.
- Capture waits for current-scene motion, scene construction, lighting, GPU asset
  preparation and shaders. A completed report from an older seed cannot release
  the next scene. Geometry annotations default to float32; the optional legacy
  float16 mode still has measurable quantization error.
- Motion is opt-in through `human_motion` plus a requested policy. Loaders cache
  model artifacts, and workers reuse loaded models across scenes. Browser motion
  requires WebGPU and compatible model hosting/CORS.
- Registry packages exclude the asset directory. Deploy `assets/burn_human` from
  the matching checkout under the application asset root, or set indoor human
  density to zero. No imported furniture or texture catalog is required.
- Registry consumers use published wgpu 29.0.4-compatible dependencies, without
  inheriting the checkout's root patches. Keep bounded dataset-worker lifetimes;
  patched-checkout memory/performance measurements do not qualify registry builds.

## Evidence and limits

The [v13 review](indoor_review_v13.md) records 120 passing library tests, an actual
GPU capture regression, native multiframe CLI motion, a real Wasm/WebGPU motion
run, 128 valid procedural rooms and a small rendered diagnostic cohort. Earlier
[flow qualification](optical_flow_validation.md), [motion seed qualification](human_motion_randomness.md),
[prompt review](motion_prompt_grammar.md) and [SigLIP audit](embedding_audit.md)
retain their own data and scope.

Photographic realism, 10M-sample learning utility, unlimited-process memory
stability and a measured reduction in glass noise remain unproven. People and
some procedural materials remain visibly synthetic. See the
[wgpu comparison](wgpu_dependency_review.md) for the measured reasons local
patches remain, and do not treat this version bump as a new realism benchmark.
