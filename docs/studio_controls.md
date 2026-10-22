# Scene Studio configuration

Scene Studio uses the same validated policies as native generation and the browser viewer. Geometry, population, camera sampling, and quality changes are drafts until **Apply** (same seed) or **Next** (new seed). **Discard** restores the active configuration. Annotation selection, overlays, editor navigation, and timeline controls update live.

| Page | Purpose |
| --- | --- |
| Scene | Room program, furnishing density, table belongings, seed, regeneration |
| Cameras | Multi-view placement, baseline, lens distributions, ceiling rigs, trajectories |
| Materials & light | Surface and lighting seeds, exposure, PBR detail, render quality |
| People & motion | People population, human trajectories, articulated furniture and portable-item motion |
| View & playback | Editor/grid/schematic, annotations, overlays, timeline |
| Capture & tools | Image dimensions and timesteps, O-voxel, complete policy import, qualification evidence |

Controls are declared in `src/app/editor/fields/`, separate from retained Feathers widgets, distribution previews, validated editor state, and URL encoding. Updating a continuous control preserves its widget identity, focus and scroll position. Human motion controls do not initialize models until an enabled policy is applied.

## Navigating groups

Each page starts with its primary controls open. Click a group header to expand or collapse it, use **Jump to group…** to open and scroll directly to a topic, or use **Expand all / Collapse all** to browse the whole page. Headers show the number of controls inside. Switching pages remembers each page's scroll position. Group visibility is retained in shared URLs; opening, closing or navigating groups does not apply pending settings or regenerate the scene.

- **Cameras:** placement and baseline; perspective lenses; fisheye FOV; fisheye sensor shape and distortion; optical-center variation; trajectories and timing; advanced visibility, spacing and overlap sampling. Ceiling mounting exposes its own clearance controls and hides inapplicable travel controls.
- **Materials & light:** surfaces, lighting and exposure, and rendering quality are separate groups.
- **People & motion:** population, human trajectories, model quality, behavior sampling and custom trajectories are separate from furnishing selection, travel limits and fan speed. Model controls appear only when supported and enabled.
- **View & playback:** viewport and annotation selection, depth and glass, overlays, playback, editor navigation, optical flow and the co-visibility legend. Schematic mode hides controls that cannot affect its drawing.
- **Capture & tools:** capture resolution and timesteps come first; voxel export, full configuration, detailed policy JSON and qualification evidence are collapsed until needed.

The navigation layer in `src/app/editor/navigation.rs` retains controls when groups fold, so slider drags, text drafts and keyboard focus do not require rebuilding the UI. Stable group keys are independent of visible titles. Older links using `ADVANCED RIG`, `PROMPT DISTRIBUTION` or `OPTICAL FLOW` remain readable. The serialized page key for **Capture & tools** remains `advanced`.

For a bounded native visual review (one room, including a compact window), run:

```sh
cargo run --example review_studio -- out/studio_review --groups
```

## Table belongings

`indoor_clutter` accepts a typed `items::distribution::ClutterSettings` policy as an object in configuration files or JSON in the native CLI. For example:

```sh
cargo run --bin viewer -- --scene-type procedural_indoor \
  --indoor-clutter '{"probability":0.6,"minimum":0.05,"maximum":0.85,"bias":-0.3,"spread":0.8}'
```

- `probability`: fraction of table surfaces selected to carry loose belongings. The default is **0.75**, leaving a 25% proposal probability of a clear table.
- `minimum`, `maximum`: normalized amount bounds for selected tables, default **0.05–1**.
- `bias`: continuous preference from **−1** (light) to **+1** (busy), default **−0.15**.
- `spread`: **0** fixes the amount at its median; **1** spans the configured interval, the default.

This is a mixture of empty tables and a continuous logistic-quantile distribution. Each table has a stable seeded draw shared by seated belongings and loose dressing. Area, room furnishing density, physical support, room activity and collision checks affect the realized item count. The amount is a proposal, not a promise to fit a fixed number of objects. Counters and storage retain their functional programs; clear desks may retain installed monitors, laptops and their input devices. A zero probability or zero maximum suppresses loose table belongings without disabling furniture.

The native/Burn generator configuration and `zeroverse_gen --indoor-clutter` expose the same policy. Manifests and metric reports record it. The preparation cache includes it in the exact request identity, preventing reuse of a room prepared with different clutter settings.

## Readable browser URLs

Generated links contain only settings differing from browser defaults. Nested policies use dotted keys rather than encoded JSON documents:

```text
?scene_type=ProceduralIndoor&indoor_seed=400
&clutter.probability=0.6&clutter.bias=-0.3
&camera.lenses.probability=0.8
&camera.lenses.fov_degrees.0=140&camera.lenses.fov_degrees.1=210
```

(Line breaks above are for readability.) Policy prefixes are `camera`, `clutter`, `appearance`, `human_motion`, `object_motion`, and `view`. Other top-level settings retain their CLI names with underscores. Arrays use numeric components such as `.0` and `.1`. Explicit `null` disables an optional policy. Empty-object `{}` and empty-array `[]` markers enable/reset groups when necessary; values are percent-encoded normally. Unapplied edits travel under `view.pending` and do not silently become active on reload.

Old `indoor_camera={...}`, `human_motion={...}`, and `viewer_state={...}` links remain readable. Dotted fields override legacy whole-policy values independently of URL order. Duplicate keys, malformed paths, invalid typed values and excessively large inputs are rejected. The active seed and view state are synchronized with `history.replaceState`, preserving browser history and the page fragment without reloading.
