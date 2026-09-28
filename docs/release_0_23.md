# bevy_zeroverse 0.23

This release adds exact co-visibility annotations, generator-19 clothing and viewer improvements, and a project page with rendered multi-view examples, videos, downloadable annotations and a compiled technical whitepaper. Companion versions are `bevy_zeroverse_ffi` **0.23.0** and `bevy_zeroverse_burn` **0.6.0**. The unchanged `burn_siglip2` crate remains **0.1.1**; Bevy, Burn and human-model dependencies retain their existing published versions.

## Behavior and compatibility

- Generator **19**, capture identity **v30**, ARDY motion policy **6**: start new shards when upgrading. Resume rejects mixed capture identities. Public Rust sample/view structs and the render-mode enum gain co-visibility fields/variants, so callers with exhaustive matches or struct literals need updating. Existing supported safetensor/folder captures without co-visibility remain readable.
- Optional `co-visibility` records exact per-pixel `u16` membership for up to 16 capture cameras, with separate validity and deterministic additive colors. The editor camera is excluded. Native CLI, Rust/Python loaders, lossless PNG/NPZ previews and WebGPU viewer previews are supported. Glass is treated as the first opaque geometric surface, as in the other geometric annotations. See the [contract and validation](co_visibility.md).
- Fitted clothing follows sampled body sections and rounded torso contours; collar/lapel/button details use local surface projection. Pose joints remain visible through people, while walls and furniture occlude them. Backless stools occur in the procedural seating distribution.
- Motion controls expose diffusion steps, frame count, history, guidance and generation limits. Default prompts use fewer modifiers; playback can run once at the model's 20 Hz. Policy changes apply on explicit regeneration. Editor intrinsics persist across regeneration; advanced scene controls and room statistics are visible.
- Viewer flow previews use a fixed reference interval instead of frame-dependent magnitude and reset history at discontinuities. Dataset flow continues to use exact capture timesteps. See the [viewer/clothing validation](viewer_quality_v19.md).
- The [project page](https://mosure.github.io/bevy_zeroverse/project/) presents eight matched annotation modes, four-view traversal and human-motion videos, exact co-visibility/calibration downloads, and the compiled [whitepaper](https://mosure.github.io/bevy_zeroverse/project/static/papers/bevy_zeroverse.pdf). Its selected v19 examples are explicitly separated from the retained 1,024-layout / 64-rendered-room v18 population audit. See [asset provenance and reproduction](project_page.md).

## Validation and package scope

Recorded validation covers native unit tests, GPU co-visibility and temporal-flow regressions, pose occlusion and 30/60/120 Hz preview consistency, exact serialization round trips, native/WebGPU motion smoke tests, strict workspace Clippy, and native/wasm viewer builds. The page passes 64 view/mode browser cases, responsive layouts, media playback and download checks; the paper and source archive compile independently. Detailed denominators, settings and limits are retained in the linked reports.

During release validation, one isolated dataset-resume test process exited unsuccessfully after its assertions passed. A focused rerun and the complete Burn suite passed. The intermittent process exit was not reproduced or diagnosed; passing bounded tests does not establish long-run process stability.

Crate packages exclude the website media, research evidence, asset catalog and checkout-only graphics patches. They resolve published registry dependencies. Occupied indoor scenes still require the bundled AnnyBody assets under the application's asset root; static scenes do not load motion models. The published site contains its viewer assets separately.

These checks do not establish photographic realism, physical cloth simulation, arbitrary-prompt motion fidelity, unlimited-process memory stability, or downstream ten-million-sample learning gains. Production dataset generation should retain its process lifetime limit.
