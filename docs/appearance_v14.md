# Indoor appearance and optical qualification, v14

> **Archived evaluation.** This report describes its recorded build. Use the
> [documentation index](README.md) for current capabilities and defaults.

This update improves fitted hair, material microstructure and glass sampling, and adds a reproducible physical-rendering benchmark. **It does not establish photographic realism.** The measurements and visual failures are retained in [the evidence bundle](evidence/appearance_v14/summary.json), including the difficult dark scene. No exposure fitting, image registration, or denoised Cycles reference is used.

Hair now has a tessellated Anny scalp, a shaped and sealed hairline, fibre-aligned UVs, continuous part/curl relief, and swept, tapered bun/ponytail volumes. Tied hair attaches to the actual phenotype surface. The entire groom follows the head bone during motion, avoiding nearest-shoulder weight transfer. Geometry tests cover all eight style controls at length/curl/part extremes and bound groom vertex counts. [Before/after portraits](evidence/appearance_v14/hair_before_after.png) retain the same people, poses, cameras and studio lighting. Close-ups still reveal a stylized surface groom; this is not individual-fibre transport or a subsurface skin model. [PBRT's hair model](https://pbr-book.org/4ed/Reflection_Models/Scattering_from_Hair) describes the richer scattering model that this approximation does not implement.

Timber uses multiple directional frequency bands and independently cut grain per floor plank. Staggered planks preserve their identity across the texture wrap. Mineral veins use warped noise level sets instead of equally spaced sine bands. Skin pores use irregular relief rather than an embossed grid. Texture resolution remains 256² with linear-light colour mips and normalized normal mips; the existing neutral human maps and role-specific materials remain shared. The old fallback texture program is no longer evaluated when a sampled recipe supplies the pixels.

A small integration in `src/scene/procedural_indoor/shading.rs` adapts the pinned Bevy 0.19.1 shader libraries. StandardMaterial remains the asset type used by annotations and exports; there is no Bevy/wgpu fork or additional shader/material asset family. It corrects two anisotropy initialization guards (normal prepass and absent optional anisotropy textures), Beer-Lambert extinction, and transmission coverage normalization. Captures wait until these integrations are installed. The shader-source assertions deliberately require requalification when upgrading Bevy.

Glass uses deterministic symmetric quadrature instead of per-pixel random rotations/checkerboard blur radii. The radial distribution matches the previous native Ultra kernel. Clear panes with subpixel blur use one fetch; resolved rough panes use 64 taps on native Auto and 32 on WebGPU Auto. IOR, tint, thickness and the clear/frosted distribution remain independently sampled. Bright radiance is not clipped to hide fireflies. Rejected samples return coverage and unassociated colour, avoiding a second coverage attenuation when blending with the environment. Screen-space refraction, missing offscreen geometry and approximate indirect lighting remain limitations.

The Cycles bridge now maps volume absorption using `sigma_a = -log(T) / distance`. Its volume node uses `(1-Color)*Density`, following [Cycles' implementation](https://raw.githubusercontent.com/blender/blender/main/intern/cycles/kernel/osl/shaders/node_absorption_volume.osl). A new analytic slab control verifies this mapping; all six radiometry controls pass, including absorption with relative error below 0.000001. The native volume thickness approximation and Cycles' traced path length can still differ.

| Measurement | Observed result | Scope |
| --- | --- | --- |
| Glass quadrature error | **72.97% lower luminance RMSE** | 16 consecutive rooms, 32 views; 24 views have sufficient glass coverage; 407,489 eroded window pixels. Each filter is compared to its own 2,048-sample integral. |
| Error against one shared legacy integral | **61.85% lower aggregate RMSE** | Includes changed filter/coverage bias. Spatial block bootstrap after/before interval is **0.310–1.058**, so a uniformly improved shared-reference result is not established. |
| Sampling-error uncertainty | After/before RMSE interval **0.255–0.402** | 1,000 resamples of 32×32 spatial blocks within these scenes; not confidence over the full procedural distribution. |
| Cycles radiance comparison | Raw RGB relative MAE **20.3–103.3%**; 32×32 block MAE **9.0–98.7%** | Four rooms, eight views, two independent raw 8,192-sample OptiX references per view. No human content in this optical test. |
| Reference uncertainty | Independent Cycles block disagreement **0.05–1.27%** | All eight views pass the existing coarse convergence gate. Per-pixel convergence is not claimed. |
| Capture wall time | Median **0.80 s**, range **0.53–1.85 s** per two-view room | Includes generation, readback and reference export. Checks ran concurrently; these are not isolated GPU timings or a throughput improvement claim. |

[Glass comparisons](evidence/appearance_v14/glass.png), [native/Cycles comparisons](evidence/appearance_v14/native_cycles.png), [per-view optical reports](evidence/appearance_v14/cycles), and [source/artifact provenance](evidence/appearance_v14/provenance.json) are included. Seed 47 is intentionally very dark: its high relative error is retained, not hidden by increasing exposure. Native/Cycles error includes light-proxy, probe, BSDF, reflection and refraction differences. Matched generated-scene comparisons also cannot establish likeness to real photographs or downstream training utility.

Validation: 135 Rust library tests passed, three ignored; 56 Python tests passed; workspace/all-target strict Clippy and formatting passed. Native captures include aligned depth, normals, position and semantic outputs. The Wasm viewer builds with `web,human_motion`; occupied Auto and Portable scenes were rendered in headed Chrome/WebGPU with nonblank scene pixels and no WebGPU validation errors. [Browser evidence](evidence/appearance_v14/browser.json) records the scene-ready logs and screenshots. The browser test server must serve the existing Anny assets; GPU submissions while waiting for assets do not count as success. Model inference was not initialized for these static appearance tests. Motion attachment is covered by an actual Anny groom/head-turn deformation test, not a fresh ARDY inference run. Burn CubeCL still emits its upstream Rust future-incompatibility notice.

To reproduce the glass test in a fresh output directory:

```sh
set -e
cargo build --features human_motion --bin indoor_validate --example review_humans
for mode in legacy legacy-reference default reference; do
  target/debug/indoor_validate --seed 44 --audit-seeds 16 --renders 16 \
    --cameras 2 --width 384 --height 288 --human-density 0 \
    --labels --linear-rgb --export-reference --glass-filter "$mode" \
    --output "out/appearance_check/glass_$mode"
done
python scripts/compare_indoor_glass.py --root out/appearance_check \
  --output out/appearance_check/report
target/debug/examples/review_humans out/appearance_check/people
```

`--glass-filter` is an offline validator diagnostic. Dataset generation/viewers use the default filter; the 2,048-sample modes are intentionally expensive. The comparator verifies exact resolved optical inputs and identical semantic images, and rejects mismatched runs. Reference export refuses to overwrite an existing reference directory.

For a fresh physical reference, run the controls and two independent sampling seeds:

```sh
set -e
blender -b --factory-startup --python scripts/validate_cycles_radiometry.py -- \
  --device OPTIX --output out/appearance_check/controls
for offset in 0 10000; do
  blender -b --factory-startup --python scripts/render_indoor_cycles.py -- \
    --scene out/appearance_check/glass_default/seed_000044/reference/scene.json \
    --output "out/appearance_check/cycles_$offset" --device OPTIX \
    --samples 8192 --bounces 12 --camera-indices 0 1 --seed-offset "$offset" \
    --radiometry-report out/appearance_check/controls/report.json
done
python scripts/compare_indoor_cycles.py \
  --native out/appearance_check/glass_default/seed_000044 \
  --cycles out/appearance_check/cycles_0 \
  --reference-repeat out/appearance_check/cycles_10000 \
  --output out/appearance_check/physical_comparison
```

`python scripts/smoke_indoor_browser.py --url http://127.0.0.1:8766 --output out/browser_check` checks a locally served viewer plus assets. Use the matching wasm-bindgen CLI (0.2.128 for this lockfile). Headless Chrome compositing returned black on this machine; the successful qualification uses headed Chrome and verifies scene pixels outside the inspector panel.

Photographic realism, full-distribution quality, temporal shimmer, 10M-scale learning utility and unlimited-process memory stability remain open qualifications. This evidence establishes the bounded improvements above, not a state-of-the-art claim. The scene grammar is version 14 and the capture identity is `capture-v23`; previous capture resumptions must not silently mix these pixels/meshes with older versions.
