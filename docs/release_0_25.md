# bevy_zeroverse 0.25

The camera baseline slider continuously coordinates view spacing, shared-surface
overlap, horizontal spread and independent path variation. Separate trajectory
controls specify travel within the room. The current project page and technical
whitepaper report absolute measurements at five baseline settings, with refreshed
renders, annotation previews and video.

Versions: `bevy_zeroverse` **0.25.0**, `bevy_zeroverse_ffi` **0.25.0**, and
`bevy_zeroverse_burn` **0.8.0**. `burn_siglip2` remains **0.1.1**; model and engine
dependencies retain their published versions.

## Compatibility

- Generator **21**, capture identity **v32**, multi-view policy **4**. Start new
  capture shards; resume rejects a changed engine identity.
- `MultiViewSettings` adds `min_reference_baseline`. Rust struct literals need
  the new field or `..Default::default()`. Missing JSON fields use zero for this
  reference minimum, preserving explicit custom policies from older archives.
- New requests default to baseline **0.5**: every pair at least **0.27 m** apart;
  reference distances **1.21–5.175 m**; proxy overlap approximately **28.7%**;
  horizontal spread **0.25**; independent path variation **0.5**.
- `indoor_camera` accepts `{"baseline":0.75}`. It expands to concrete settings on
  serialization. Combining the shorthand with `multiview` is rejected. Advanced
  edits enter Custom mode; regeneration never silently reapplies a preset.
- Reference proposals account for reachable room extent and usable height, with
  bounded retries. All constraints retain strict rejection and full-path swept
  collision checks; impossible requests report an error.
- Metrics schema **10** adds minimum reference distance to group exports.
  Existing scene modes and render channels remain supported.

## Validation and evidence

[Current protocol and results](camera_baseline_v21.md): 2,048 audited layouts,
512 distinct rendered rooms, and a matched 128-room sweep at five baseline values.
All measurements retain low-overlap samples and report their denominators. The
independent rendered-depth diagnostic differs from exact production GPU masks;
the matched gallery exports the latter separately.

Local release checks cover 171 library tests, 10 generator unit tests, 66 Python analytic/report tests, strict workspace
Clippy and motion-enabled Wasm compilation. Native GPU co-visibility, FFI
headless capture, 64 browser annotation combinations, video playback and a
standalone build of the whitepaper source ZIP pass. Clean package verification
and exact-commit CI are required before publication. Registry archives are checked
against the pushed commit. Dependency future-incompatibility notices from Burn
remain; no project Clippy warnings are suppressed.

Packages exclude research media, assets and local graphics-library patches.
Published dependencies resolve through crates.io. Static scenes do not initialize
ARDY; occupied scenes require the application's AnnyBody assets. The evaluation
does not qualify photographic realism, unlimited-process memory stability or
measurable downstream training gains.
