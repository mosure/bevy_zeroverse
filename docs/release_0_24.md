# bevy_zeroverse 0.24

This release replaces translated copies of multi-view camera trajectories with
varied headings, travel and curvature, and enforces group separation and spread
throughout sampled playback. It includes a 2,048-layout / 512-rendered-room
evaluation, all-camera co-visibility diagnostics, refreshed matched annotation
renders and video, and an updated technical whitepaper.

Versions: `bevy_zeroverse` **0.24.0**, `bevy_zeroverse_ffi` **0.24.0**, and
`bevy_zeroverse_burn` **0.7.0**. `burn_siglip2` remains **0.1.1**; Bevy, Burn and
human-model dependencies retain their published versions.

## Compatibility

- Generator **20**, capture identity **v31**, multi-view policy **3**. Start new
  capture shards; resume rejects a changed engine identity.
- `MultiViewSettings` adds `min_spread` and `trajectory_variation`. Rust struct
  literals need the new fields or `..Default::default()`. The minimum baseline
  now constrains every camera pair; the maximum remains reference-relative.
- New configuration requests use spread 0.25 and variation 1.0. Archives missing
  those fields deserialize with them disabled. Explicit rigid/collinear rigs
  remain possible. Impossible constraints fail without a relaxed fallback.
- Metrics schema **9** adds per-room group statistics and 33-time camera-path
  CSVs. The inspector exposes spread and variation; edits apply on regeneration.
- Existing scene modes, render channels, human-model loading and static-scene
  behavior remain supported.

## Evidence and limitations

[Expanded results](camera_evaluation_v20.md): 626 of 2,048 old starting groups were
near-collinear; none of the new groups are. The 512-room render run contains
6,144 views. Depth-derived co-visibility shows 84.27% of valid source pixels shared
with another camera. Reference overlap averages 53.36%; 245/4,608 pairs miss the
35% proxy target, and the minimum is 20.50%. Those misses remain public.

The depth diagnostic is independent of the production GPU annotation's
tangent-plane test. The matched gallery exports exact production masks. Camera
checks are finite-time constraints with separate swept collision validation;
the audit does not establish photographic realism, downstream learning gains,
or unlimited-process stability. Four bounded capture processes are not a
continuous-process endurance qualification.

Local validation includes 169 library tests, the Burn generator suite, FFI
headless capture integration, Python analytic/report checks, strict workspace
Clippy, Wasm motion-enabled compilation, package verification, and the updated
paper/gallery checks. Existing Burn dependency future-incompatibility notices
remain. Final CI, live-site and registry provenance are verified during release.

Packages exclude research media, the website, assets and local graphics-library
patches. Published dependencies resolve through crates.io. Occupied scenes need
the application's AnnyBody assets; static generation does not initialize ARDY.
