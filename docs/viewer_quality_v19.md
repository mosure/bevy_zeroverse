# Clothing, motion controls and viewer annotation review

> **Archived evaluation.** This report describes its recorded build. Use the
> [documentation index](README.md) for current capabilities and defaults.

Generator 19 replaces the squared torso offset with smooth sections fitted to
each Anny body. Front and back profiles hang below their supports instead of
following every anatomical depression. Shoulder and hem offsets fade smoothly;
compression folds concentrate near garment edges and joints. Collars, lapels
and button bands use the actual garment triangles and a small surface clearance.
Clothing still uses the Anny geometry and skinning, including moving actors.

[Eight matched clothing comparisons](evidence/viewer_quality_v19/clothing_before_after.png)
use identical people, poses, cameras, materials and lighting before and after the
change. All three outfit types are included. This is a visual geometry review,
not a photographic similarity benchmark or a cloth simulation. Fine fabric,
facial appearance and source-model motion still have visible limitations.

The **Human motion → ARDY generation parameters** panel exposes diffusion steps,
clip length, history length, guidance strengths, dense path conditioning, batch
size and retry limits. Changes apply with **Regenerate / R**. The existing default
already used the model's maximum of ten diffusion steps. Prompt program 2 keeps
the action/route grammar but adds at most one optional style modifier; its default
probability is 0.35. **Play once at model speed (20 Hz)** restarts the active clip
forward at its original rate. Default Sin playback at speed 0.2 traverses an eight-second clip
forward in 2.5 seconds and then reverses it, so it is a poor setting for judging
natural movement. Sin remains available. See [the motion policy](human_motion.md).

Pose gizmos use the correct 21-joint hierarchy for indoor people and the model
hierarchy for other Burn humans. An optional editor depth pass reuses environment
mesh handles and excludes people, including clothing. Joints therefore show
through people while walls and furniture occlude them. This adds one depth pass
only while the editor pose overlay is active; toggling it off removes its camera
and mesh instances. It does not add geometry to dataset annotations. The camera
grid continues to show capture cameras without editor overlays.

Editor regeneration preserves projection settings, including field of view,
near/far planes and aspect ratio. Repositioning in the primary room remains.
Advanced indoor settings now expose the indirect-lighting budget, orientation
augmentation and a summary of the active room, furnishings and lighting.
[WebGPU screenshots](evidence/viewer_quality_v19/web_controls.png) show the motion
controls; [advanced settings](evidence/viewer_quality_v19/web_advanced.png) and
[pose overlays](evidence/viewer_quality_v19/web_pose_on.png) were also checked in
the browser. Editing generation controls still requires explicit regeneration.

The optical-flow preview now shows velocity over a fixed reference interval,
defaulting to 0.05 seconds, with adjustable color scale. This removes frame-rate
scaling for constant velocity. The preview remains an estimate from successive
frames; [numeric dataset flow](optical_flow.md) still uses exact captured endpoints.
Regeneration, playback-mode changes and timeline jumps invalidate preview history
while the renderer establishes fresh previous-frame transforms.
Motion interpolation also interpolates local bone translations and scale, avoiding
translation discontinuities at frame boundaries when offsets vary.

## Recorded checks

[Machine-readable results](evidence/viewer_quality_v19/validation.json) record the
small-sample qualification and its limits.

| Check | Result |
| --- | --- |
| Clothing | Eight identical adults before/after, 18 RGB/semantic views each; additional upright wardrobe review |
| Backless stools | 201 / 1,198 primary-room seats across seeds 0–127; present in 89 / 128 rooms; no geometry above the seating datum |
| Pose occlusion | 1,115 changed pixels through an opaque person; zero through the foreground wall; disabling restores the original image exactly |
| Viewer flow cadence | Identical center RGB `[255,241,241]` at 30, 60 and 120 Hz for constant translation |
| Motion planning | 58 feasible requests, 34 action programs and 51 texts across 32 rooms; 11–37 tokenizer tokens including header |
| Native motion | Three accepted motions across seeds 13 and 2; one person remained static because no collision-free behavior fit |
| WebGPU motion | Two accepted motions before and after regeneration; one GPU device, no extra model-manifest requests on regeneration; static control loads no motion models |
| Rust | 166 library tests passed, three ignored; explicit GPU pose, optical-flow and co-visibility regressions passed |
| Builds | Native and WebGPU (`web,human_motion`); strict workspace/all-target Clippy with `human_motion,embedding_audit` |

The model runs are smoke tests of generation, navigation admission, playback and
cache reuse. They do not prove natural movement for every prompt, nor establish
photographic realism. Burn's dependencies still emit upstream Rust
future-compatibility notices; the strict Clippy check passes.

## Reproduce

```sh
cargo test --features human_motion --lib
cargo test --features human_motion --test viewer_annotations -- --ignored --nocapture
cargo test --features human_motion --test optical_flow_render --test co_visibility_render -- --ignored
cargo run --features human_motion --example review_humans -- out/clothing_review
cargo run --features human_motion --example review_humans -- out/clothing_standing --standing
cargo clippy --workspace --all-targets --features human_motion,embedding_audit -- -D warnings
cargo build --no-default-features --target wasm32-unknown-unknown --features web,human_motion --bin viewer
```

After serving the current Wasm bindings and assets, run
`scripts/smoke_viewer_annotations_browser.py` and
`scripts/validate_human_motion_web.py` against the local server. The latter also
requires the motion CDN or a local model mirror.
