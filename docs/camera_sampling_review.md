# Camera baseline and count qualification

The current sampler is measured with bevy_zeroverse 0.39.0, generator 37,
source `760448699a3a7edb6ef25b2cd8623614dcd2bf53d3871ed9225a28e9c8ec8130`.
The formal Rust publication pipeline consumes the [portable report](evidence/camera_counts/study.json),
checks its identity and receipt, and generates the project-page and paper figures.

## Sampling contract

Baseline coordinates spacing, shared-surface overlap, path deformation and visible-region
target variation. Zero keeps close stereo rigs with small aim jitter and first-fit placement.
Positive variation samples reference-visible surface hits throughout the primary room;
its target-proposal probability increases with camera count. The aim interpolates toward
the sampled hit by the variation value. Every candidate passes swept clearance, semantic
content, synchronized group geometry and bidirectional reference overlap before selection.

Candidate preferences combine nearest-peer distance, nearest optical-axis angle and
additional quantized primary-room surface cells. Preferred distance scales with room
diagonal divided by sqrt(camera count). A small bounded candidate pool grows from two
to four feasible views with count. These are soft preferences; impossible requests do
not receive weaker thresholds. One bounded overcomplete recovery may retain a validated
subset. Temporal paths can remain related; varied gaze does not imply independent route
topology. No 90-degree cap or guaranteed 360-degree arrangement is imposed.

The default overlap floor is approximately 0.342 at baseline 0.5 and 0.15 at baseline 1.
Custom/archived policies omitting target_variation keep shared-target behavior.
Single-camera requests are invariant to baseline. Different counts resample complete
sets; they are not nested prefixes. Objects, people, materials and lighting remain
unchanged when cameras are resampled. See [configuration and API](multiview_cameras.md).

## Complete placement audit

All 2,304 requested sets passed: 64 consecutive room seeds, 1–12 cameras and baselines
0, 0.5 and 1. No set is omitted and no accepted set contains an exact repeated pose
sequence at the 33 synchronized samples. Furnishing density is 0.65, static human density
0.25 and aspect ratio 1.6. At baseline 1:

| Cameras | Median initial nearest neighbor | Median largest optical-axis angle | Median look-target diameter |
| ---: | ---: | ---: | ---: |
| 2 | 3.58 m | 66.1° | 1.34 m |
| 3 | 2.81 m | 104.2° | 2.46 m |
| 4 | 2.42 m | 112.9° | 3.44 m |
| 8 | 1.28 m | 132.1° | 4.28 m |
| 12 | 0.99 m | 157.3° | 4.82 m |

Spacing observations are camera-level; angle and target diameter are room-level.
Pose/trajectory statistics measure geometric sampling, not perceptual uniqueness or
model-learning utility. The full report retains intrinsics, path lengths, absolute-pose
similarity, relative-motion similarity, heatmaps and denominators.

## Native rendered qualification

Two prespecified room families, twelve configurations and two endpoints provide 312
calibrated views and 2,424 directed pair/time observations. Native shadows, SSAO, GPU
materials and approximate indirect lighting are retained at 320×200. Every channel
and the executable are checksum-bound. Rendered poses match CPU traces. At baseline 1:

| Cameras | Mean directed overlap | Mean visible in any peer | Minimum reference overlap |
| ---: | ---: | ---: | ---: |
| 2 | 50.8% | 50.8% | 35.5% |
| 4 | 42.1% | 74.8% | 23.6% |
| 8 | 41.3% | 87.9% | 19.4% |
| 12 | 34.7% | 94.3% | 17.1% |

Room families are the independent units; camera pairs and endpoints are correlated.
These two families are a rendered spot check, not a population estimate. Pixel overlap
uses valid first-surface source observations and the solid-glass annotation policy;
it does not count unique 3D points. Proxy surface-cell novelty is not measured mesh area,
photographic realism or a reconstruction-accuracy result.

## Reproduction

The Rust CPU audit initializes no GPU or motion models:

```sh
cargo run --locked --example audit_cameras --no-default-features --features multi_threaded -- \
  --rooms 64 --workers 4 --output out/camera-counts/audit
```

The native sweep uses `bevy_zeroverse_publication qualify`; its versioned recipe is
embedded in `study.json → rendered_qualification.recipe`. Set cameras per case,
human_density 0.25, two rooms and two playback steps. The tool preflights the actual
validator identity and uses an explicit asset root.

```sh
cargo run --locked -p bevy_zeroverse_publication -- camera-study \
  --input out/camera-counts/audit --rendered out/camera-counts/native \
  --output docs/evidence/camera_counts
cargo run --locked -p bevy_zeroverse_publication -- refresh
cargo run --locked -p bevy_zeroverse_publication -- verify
```

Study folders are immutable; archive old reports locally before replacing the configured
snapshot. publication.toml declares the current input. The project page and paper contain
current-generator evidence only. Matched cross-version diagnostics remain local.
