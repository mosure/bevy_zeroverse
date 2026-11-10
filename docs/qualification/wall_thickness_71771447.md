# Wall section qualification: seed 71771447

Core/FFI 0.63.3, Burn bridge 0.47.3, capture 0.6.23 and publication 0.9.11
contain this repair. Generator identity is 119; capture identity is
`capture-v144`, with exterior 6 and envelope 11. Burn remains 0.22.1.

## Failure and repair

The downstream trainer rejected this seed with `invalid exterior wall section
or window return`. An unpatched registry 0.63.2 consumer reproduced the failure.
The logarithmic wall-thickness sample converted back to 0.14999999 m: one
float32 step below the strict 0.15 m minimum. All 874,506 constructed triangles
passed mesh validation, with no degenerate triangles.

The sampler now clamps its converted metric result to [0.15, 0.72] m. Validation
remains strict, including rejection of NaN and out-of-range archived values.
The RNG stream and all in-range samples are unchanged. The correction adds
scalar bounds checks during scene preparation, with no allocation, retries,
GPU work or per-frame cost.

## Validation

All 19 facade tests passed, including full layout and constructed mesh checks
for 71771446, 71771447 and 71771448. Each scene also passes eight-camera
trajectory clearance, lens/mount assignment, motion-policy and deterministic
replay checks using the recorded downstream mixed rig configuration.

Existing distribution, aperture, glass-ray, legacy-manifest and malformed-data
tests pass. Strict core/capture all-target Clippy, WASM `web,human_motion`
compilation, formatting and diff checks pass. Hardware GPU capture, browser
runtime and concurrent-training throughput were not rerun for this scalar
sampling correction. Production training was not modified.

Detailed local evidence is under `out/seed_71771447` and `out/release_0633`.
Registry, CI and deployment status belong to the release receipt separately
from these source-level checks.
