# Scene Studio

Native and WebGPU viewers use the same Bevy Feathers controller. The scene remains
visible beside a fixed, scrollable control panel. Hide expands the scene viewport;
View → Capture grid reserves the same viewport for dataset cameras.

- **Scene:** mode, full-width room seed, room program and population.
- **Cameras:** view count, spatial baseline, proxy overlap constraints, continuous
  lens bias/spread with a live PDF preview, moving-camera percentage, handheld /
  room-exploration presets, metric increments and duration.
- **Materials & light:** independent material / lighting seeds, relief, dimming,
  exposure and platform quality. Auto retains all platform-supported effects.
- **People & motion:** population, ARDY sampling, diffusion, guidance, batching,
  prompt-family weights and exact custom actor trajectories.
- **View & playback:** annotations, camera grid, overlays, editor lens, timeline,
  playback curves (including Sin) and fixed-interval optical-flow visualization.
- **Advanced:** complete policy JSON, capture resolution and timesteps, O-voxel,
  depth units, navigation preferences and optional scene-resource diagnostics.

**Apply changes** regenerates with the selected seed. **Next / R** applies the
configuration and advances from the displayed seed. Moving sliders or typing
never regenerates a scene. Pending changes are visibly marked; Discard returns
them to the last applied configuration. Preview controls update immediately.
Space controls playback when the scene has keyboard focus; focused buttons retain
their normal keyboard activation.
Invalid fields remain editable and block generation with a readable error.

**Mixed · all room types** samples every concrete room program uniformly: the ten
office/lounge profiles, Kitchenette, Kitchen, ElevatorLanding, Cafe, Cafeteria,
Classroom, LectureHall, Showroom, Gallery and PublicSuite (5% each before any
construction failure). `indoor_layout=all` and `indoor_layout=mixed` select the
same sampler. Selecting a specific program restricts generation to that program.
The selected concrete program is stored in each manifest. Public suites can
contain several functional zones; they are one top-level draw, not extra samples.

On WebGPU, the address bar is updated with `history.replaceState` after valid edits
and scene generation. It records the current seed, complete nested policies,
render/grid/overlay state, panel, timeline and editor camera in versioned
`viewer_state` JSON. Copying the URL replays those settings without carrying a
sequence offset from previous rooms. Pending edits are restored as pending,
preserving the rendered scene until Apply. Startup options such as headless mode or
readback require relaunching with the desired CLI/query setting. Continuous motion/camera updates are
throttled to twice per second. Random legacy object scenes do not have the indoor
seed replay contract. Low-level ECS inspector edits are not serialized in URLs.

Camera baseline remains `indoor_camera={"baseline":0.8}`; the editor expands it into
explicit constraints in shared links so custom overrides remain unambiguous.
Baseline separates cameras; travel bounds move selected cameras. Moving cameras
at 0% fixes every capture pose while other scene motion continues; default 100%
keeps all camera trajectories enabled. Lens preference and spread control the
intrinsic proposal PDF continuously, independently of baseline. Neither guarantees a
particular measured pixel overlap. Collision checks and placement constraints
remain active for all presets.

**Materials & light → Capture supersampling (SSAA)** controls the dataset camera
grid. Apply rebuilds its targets; the editor camera is unaffected. Factor 2 uses
four samples per final RGB pixel, factor 3 uses nine, and factor 4 uses sixteen.
The default is **Auto** (`ssaa_factor=0`): 2× below 1280×720 pixels, otherwise 1×.
An explicit 1 disables SSAA; 2/3/4 force that factor at any resolution. Shared
WebGPU URLs preserve the selection. The threshold compares pixel counts, so
portrait and landscape orientations receive the same treatment.
Continuous depth/normal/position/flow previews are area-filtered, with normals
renormalized. Semantic and co-visibility previews keep categorical samples.
Numeric dataset annotations always retain exact final-grid pixel-center rays,
even when their viewer preview is filtered; preview colors are not metric labels.

Motion models initialize only on generation with an enabled motion policy. Native
builds need `--features human_motion`. WebGPU keeps native-only GI options in shared
settings and identifies their platform limitation. O-voxel rejects motion and
multiple capture timesteps rather than producing inconsistent geometry labels.

## Validation

Pure Rust tests exercise draft/application separation, 64-bit seed handling,
policy validation, all enumerated choices, URL round trips and shortcut isolation.
`cargo run --example review_camera_controls` captures the fixed/mixed camera controls
and lens PDF states under `out/camera_controls/native/`, and checks that fixed
capture transforms stay unchanged during playback.
`cargo run --example review_editor --features human_motion` captures the actual
native controller states to `out/ui_revamp/native/` for visual review.

## Native visual review

Actual Bevy screenshots of seed **46839170**, four 512 × 512 capture views, in a
1360 × 900 editor window. Motion settings shown as pending do not start model loading.

<details><summary>Scene and active seed</summary>

![Scene and active seed](viewer/00_Scene.png)

</details>

<details><summary>Camera baseline and travel</summary>

![Camera baseline and travel](viewer/01_Cameras.png)

</details>

<details><summary>Materials and lighting</summary>

![Materials and lighting](viewer/02_Appearance.png)

</details>

<details><summary>Population and ARDY motion</summary>

![Population and ARDY motion](viewer/03_People.png)

</details>

<details><summary>Annotations and playback</summary>

![Annotations and playback](viewer/04_View.png)

</details>

<details><summary>Complete policy editors</summary>

![Complete policy editors](viewer/05_Advanced.png)

</details>

<details><summary>Pending changes</summary>

![Pending changes](viewer/06_Scene.png)

</details>

<details><summary>Semantic preview</summary>

![Semantic preview](viewer/07_View.png)

</details>

<details><summary>Co-visibility preview</summary>

![Co-visibility preview](viewer/08_View.png)

</details>

<details><summary>Annotation dropdown</summary>

![Annotation dropdown](viewer/09_View.png)

</details>

The native review also exercises text entry of `u64::MAX`, slider events, Discard,
text hit-target bounds, menu activation and focused-button keyboard activation.
Core validation passed 383 tests
(22 ignored); the reported seed passed 27 camera/population/furnishing configurations.
The shared WebGPU controller compiles with motion enabled. Browser rendering could
not be qualified in the connected in-app browser because it exposes no WebGPU
adapter; its preflight error state was checked. The native images above are not
presented as browser captures.

The View & playback viewport selector also offers a [room schematic](schematic.md):
a metric top-down diagram of primary-room geometry, capture cameras, trajectories
and human joints. Its native/Wasm renderer is shared with optional dataset exports
and supports prediction overlays.

**View & playback → Capture cameras & paths** shows compact, calibrated camera
glyphs in the editor viewport. Each capture ID has a stable color shared with the
schematic, a `C0`, `C1`, … label and a subdued matching trajectory. Pinhole glyphs
retain their image-plane proportions; fisheye glyphs show their curved angular
coverage, including rearward rays above 180°. Glyph radius is capped at 0.35 m
and 24 display pixels, adapting to editor zoom without changing capture optics.

Insert `app::editor::schematic::Predictions` for matching dashed camera overlays
in both views. Label predictions with their capture ID (for example `Pred C7`)
and supply their predicted calibration and rigid `world_from_view` matrix.
Invalid poses or lenses are skipped in the 3D view. Editor overlays are
depth-tested against room geometry and excluded from capture render layers.

Custom inference viewers can reuse `camera::gizmo::{color, rgb, color_index,
wireframe, prediction_wireframe, display_radius, config}` for the same palette, calibrated outline and
solid/dashed styles. Colors depend on camera ID rather than draw order or camera
count; IDs 0–15 use distinct palette entries instead of repeating every five.
