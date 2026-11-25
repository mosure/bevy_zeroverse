# Reproducible indoor datasets

Multi-view reconstruction uses four cameras and the default
[shared-surface camera policy](multiview_cameras.md), equivalent to
`--indoor-camera '{"baseline":0.5}'`. This requests at least 0.27 m pair
separation, 1.21–5.175 m reference distance and approximately 28.7% estimated
bidirectional overlap; accepted spacing depends on room geometry and visibility.
The CLI defaults to one worker, 16 rooms per chunk, one timestep, static people
and 256 GI rays/probe. These are practical starting points, not a trained optimum.
Use `--indoor-camera '{"multiview":null}'` for independent views.

[Indoor O-voxel export](ovoxel_indoor.md) captures primary-room surface geometry
and semantics for single-timestep datasets with human motion disabled.

The supported native generator is `zeroverse_gen`. Indoor captures default to
lossless float32 sRGB color tensors, with lossless geometric and semantic labels.
Legacy scene modes retain their JPEG default. An example bounded dataset is:

```sh
cargo run -p bevy_zeroverse_burn --bin zeroverse_gen -- \
  --scene-type procedural-indoor --output out/office_dataset \
  --samples 100 --workers 1 --chunk-size 16 --seed 777 \
  --indoor-layout mixed --indoor-density 0.65 --indoor-human-density 0.25 \
  --indoor-quality auto --indoor-gi-rays 256 \
  --width 640 --height 480 --cameras 4 \
  --playback-steps 1 --rotation-augmentation \
  --render-modes color depth normal semantic position \
  --color-codec raw --compression zstd --ov-mode disabled --no-ui
```

Each process owns one Bevy engine. `--workers` controls separate processes by
default. `--per-process=false --workers=1` runs one engine directly. Concurrent
capture threads sharing one engine are rejected for indoor generation because
they cannot preserve index ordering. The seed for global sample index `i` is
`base_seed.wrapping_add(i)`, independent of process count or chunk size.

Wall outlets, desktop power strips and floor service boxes share the room’s socket
convention. A portable strip fits one or two rows of polarized, round-well or three-pin
inserts to its chassis. The openings and their backings are part of the mesh,
so RGB, depth and normals resolve the same recesses. A mains lead seats at the
contact face and reserves clearance for its plug body. The cable records the
destination object ID and zero-based `cable_target_socket`. Dedicated USB
and data ports omit the index. Archives predating socket selection use the
original first port. Metrics include insert material, width, socket count,
round-well depth and the socket indices used by cables. A floor box exposes
one or two sockets under a hinged hatch when in use. Unused boxes close flush
when the surrounding wiring permits. The distribution export records housing,
hatch dimensions and closure state for every fitted box, with lid angle and
socket pattern recorded for open boxes.

Wall switches use one to three modules per plate. Plates in a room often share
a finish and control family. The object audit's `switch_bank.modules` records
each paddle or lever angle, rotary setting, slider position or button latch.
These settings describe the visible hardware. Room metrics include
`switch_actuator`, `switch_paddle_deg`, and `switch_recess_depth_m`.

Laptops fit one to six sockets into the available chassis thickness and edge
length. USB-C, USB-A, HDMI and audio sockets have distinct apertures and
recessed interiors. The first entry in the laptop object audit's `io.ports`
is the charging and display-dock receptacle. Its `center_m` is in laptop
coordinates, and either side can carry it. Power and display leads use this
same anchor. Room metrics include `laptop_charge_side` and
`laptop_fitted_socket_count`. Older archived laptops retain their original
port markings and cable endpoints on replay.

Workstation mice include scroll wheels, finger-operated trackballs and side-mounted thumb balls.
An asymmetric grip follows the neighboring keyboard's operating side when available.
Object metrics report the control family, grip, control diameter, exposed fraction and trackball finish.
All parts share the mouse semantic label and object bounding box.

Potted succulents can form a single off-centre rosette or a parent and a smaller
offset in a wide planter. The pot opening bounds both roots, and its rim height
sets the clearance for the lowest fleshy leaves. The plant object audit records
leaf count, spiral direction, profile, thickness and cup. Tapered stems retain
continuous bark texture coordinates through bends. Split-leaf plants grow young,
intact blades alongside mature leaves with asymmetric edge clefts and enclosed
openings. RGB, shadows and geometric annotations use the same perforated mesh.
The object audit records realized counts in `split_leaf_habit`. Room metrics
report blades, intact blades, clefts and openings per split-leaf plant.

Planters sample loam, perlite mix, bark chips, expanded clay or mineral gravel.
The opening and the sampled grain size determine how many coarse pieces appear
above the fine bed. Each plant keeps a single semantic label. The object audit
records the mix, nominal coarse diameter in metres, realized grain count and the
fraction of the usable opening covered by the grains’ bounding ellipses in
`potting_medium`. The room distribution includes `plant_potting_medium` and
`plant_coarse_grains` for the plants in the room manifest. Exterior pergola
planters use the same materials. Trees planted directly in the ground generate
only their foliage and stems. Archived manifests predating generator 222 replay
their original soil geometry.

Wall pictures and exhibition-panel artworks share a physical mount. Front
images use the full normalized image domain. Framed prints may have a cut mat,
and floated canvases sit above a recessed channel. For wall pictures, the dataset
metrics record moulding shape and relief, mat thickness and bevel, float depth,
and whether the canvas edges carry the image.

Ceiling pendants hang from a flex, a downrod, or three suspension wires.
Their open bulb, recessed lens, or opal bowl supplies the emitting area for
both the visible material and the lighting model. The metrics export records
the installed optic, suspension, emitting area, canopy diameter and drop.
Diffuser dimensions contribute to the distributions only for fixtures with a physical diffuser.

Linear ceiling lights can share a product across a room or vary between its zones.
Exposed strips and wing reflectors fit one to four tubes to the available width.
Wraparound fixtures have one continuous opal lens. The visible mesh and reflection
probes use the same emitting faces. In the metrics export,
`linear_luminaire_construction` counts installed fixtures by type.
Strip width and illuminated length are measured in metres. Pitch is recorded
only for fixtures with multiple strips.

Ceiling fans sample paddle, carved, swept and commercial multi-blade rotors.
Blades on one hub share a balanced cross-section and pitch. Wood blades have
separate grain cuts along their span. Two fans can share a stocked model while
retaining independent rotor phases and motion. Their downrods fit the available
headroom and their mounting pads meet the finished ceiling. The exported fan
metrics include blade family, count, chord ratios, pitch, sweep and camber.

Laminate, ceramic and cast-mineral desks and tables share the room's PBR stock.
Each top takes a seeded cut in metres. Tables from the same order can share a
finish and cutting direction without repeating the same fleck or vein at the
centre. At a joint the material can continue across leaves or use a separate
cut. Exposed slab and formed foil carry the face pattern over the bevel.
Bonded and wood edges have their own strip, and laminate has a separate backing.
The table audit exposes `sheet_cut`. The metrics export records edge treatment,
cut origin in metres and heading in degrees. Manifest replay uses the archived
generator version to select the sheet mapping.

Desk cups use the same writing-tool geometry as loose tabletop pencils and pens.
A held tool rests on the inner floor and leans on the lip. The cup and its contents
share the holder's `stationery` label and object ID. The metrics export records
requested and fitted counts, inclination in degrees, mechanism and cup wall
thickness in metres. `held_writing_upper_end` distinguishes pencil points,
ballpoints, marker caps, erasers, push buttons and plain barrel ends. Loose tools
also record whether a barrel face or the barrel and pocket clip support them on
the worktop.

Wall loudspeakers can sit flush or tilt down on a pivoting bracket. Speakers on
furniture rest on their cabinets. Exposed drivers, woven acoustic cloth and
perforated steel all share the `speaker` label and the cabinet's instance ID.
The grilles use filtered color, roughness and surface relief over an opaque
backing. Metrics export `speaker_mount`, `speaker_face` and
`speaker_downward_pitch_deg`.

Exposed ducts carry directional, opposed-blade or crossed-return registers.
Outlets on a duct share blade stock and finish. Their dimensions and positions
fit the space between hangers. Metal louvers, the flange and the recessed
backing all use the `ceiling` label. The metrics export records register pattern,
finish, installed face dimensions in metres and station along the duct.

Sprinkler heads on one exposed pipe run share a frame finish, bulb tint and
slotted deflector. Branches can follow one side, alternate sides or change
sides in pairs. Every part carries the `ceiling` label. The
`ceiling_sprinkler_*` metrics count layouts and finishes per run, signed
lateral offsets per head, and spacing in metres between consecutive heads.

Ceiling detectors mount on a continuous ceiling or tile face. A shallow sleeve seats the housing
on pitched plaster. Paired devices can share a manufactured model and are spaced according to the room area.
Optical housings have open radial inlets backed by a recessed chamber. Heat
probes have an open guard, and status lenses have their own material. All
parts retain the `fire_detector` class and instance ID. Metrics include the
face design, inlet count, angular opening fraction, diameter and chamber setback.

Mugs can sit directly on a table or off-centre in a dished saucer. The cup foot
rests on the saucer well, and liquid follows the inside of the cup, including
faceted rims. Takeaway cups can have a paperboard or printed sleeve and a fitted
lid with an open sip port. These parts share the cup's `drink_container` label
and instance ID. Metrics record saucer support height and offset, lid rise and
port offset, and the beverage and fill fraction for cups that contain a drink.

Beverage cans have scored lids or physical pour openings with an attached folded
panel. The oval score follows the available lid diameter and leaves room for
the rivet and seam. An open can shows its interior floor or a liquid surface below the rim.
Bottle water levels follow the moulded walls. Product labels wrap once around
the container and share the container’s semantic class and instance ID. Metrics
include `can_score_area_mm2`, `can_panel_fold_deg` for open cans, `can_visible_fill_fraction` when
drink is present, `bottle_cap_flutes` and `bottle_visible_fill_height_m` for
filled transparent bottles.

Table settings, cupboards, basins and drying racks share the same plate
geometry. Plates have a recessed underside, a foot ring, a rolled rim and
round, oval or softly squared outlines. A stack can be centred or slightly
staggered. Each foot rests in the well below it. Dataset metrics for loose
and served plates include stack pitch, lateral offset, outline, foot dimensions
and contrasting ceramic rim finishes.

Sequential CLI generation enables `--indoor-prefetch=true` and
`--indoor-prefetch-depth=3`: up to three future rooms are constructed on the CPU
while the GPU captures the current room. Depth accepts 1–4; lower values reduce
CPU asset residency. Geometry,
material maps and camera planning use the same seeded functions as serial
generation. The future room is installed only after its seed and every
construction input match the next request. GPU uploads, lighting readiness,
motion inference and annotation capture keep their existing barriers. There is
no lookahead after a worker's final sample. Set `--indoor-prefetch=false` to
disable this additional CPU asset residency on memory-constrained machines.
This setting can change on resume because it does not change the dataset.

Direct `LiveDataset::get`, Python indexed capture and the interactive viewer do
not speculate about the next request. Rust callers driving the headless channel
can set `AppFrameRequest::prefetch_indoor` to the number of future consecutive
rooms needed (0 disables; values are capped at 4). Requests must reduce this
count near the end of a finite run. CPU file encoding remains bounded and overlaps rendering.
The [three-view 512×512 throughput report](generation_throughput.md) includes
capture and complete-export timings, process-count comparisons and pixel checks.

Finite indoor jobs using `--per-process=true` replace each child after at most
256 captured scenes by default. Set `--max-scenes-per-process 16` for a shorter
lifetime, or `--max-scenes-per-process 0` to disable replacement; the alias
`--scenes-per-child` is also accepted. At most `--workers` children are alive at
once. The default does not affect other scene types or direct
`--per-process=false` generation. An explicitly positive limit with direct mode
is rejected. Python's persistent in-process engine is not restarted by this CLI
option.

Child replacement limits process lifetime; it does not establish that the engine's
unbounded live heap reaches a plateau. Peak memory still depends on scene and image
sizes, chunk size, and concurrent workers, and every replacement pays engine/GPU
startup costs. Final distribution metrics currently retain numeric samples for
quantiles, so that parent-side phase grows with dataset size; use
`--indoor-metrics=false` for large jobs when those post-generation statistics are
unnecessary. Short child jobs can produce partial chunks even before the final
job. Chunk indices reserve each job's complete span, and global sample indices and
seeds remain unchanged. The limit and worker count are scheduling choices and may
change on a compatible resume. `worker_lifecycle.jsonl` records each parent run,
child PID, reused worker slot, job ID, assigned sample/chunk spans, and terminal
status without retaining an ever-growing in-memory job list.

On Linux with glibc and x86-64, spawned indoor workers default to streaming
memory copies for larger GPU uploads. The child receives
`GLIBC_TUNABLES=glibc.cpu.x86_non_temporal_threshold=32768:glibc.cpu.x86_rep_movsb_threshold=1073741824`
only when that environment variable is absent. Any user-supplied value is
preserved; `GLIBC_TUNABLES= zeroverse_gen ...` selects the system defaults.
This is a measured optimization on the qualified NVIDIA/Vulkan workstation,
not a portable speedup guarantee. It does not change tensor values, the capture
contract, the parent process, direct generation, or Python's in-process engine.
Those direct paths can opt in by setting the variable before process startup.

A failed child or process-launch error stops new launches and kills/reaps remaining
children. Completed files remain on disk. If parallel work leaves a gap after a
failure, resume refuses the incomplete index sequence; it does not silently skip
failed scenes or overwrite surviving outputs.

`--playback-step` is **normalized trajectory progress**, not seconds. Three
steps with increment `0.5` capture the start, midpoint, and endpoint. Each sample
contains `[time, camera, height, width, channels]`; the Rust in-memory view list
is time-major. Exported `time` records trajectory progress. Optical flow and
motion-vector capture export numeric forward correspondence to the next captured
timestep, including animated meshes, with separate validity and visibility masks.
The final timestep has zero vectors and masks. See the [temporal annotation
contract](optical_flow.md). [Co-visibility](co_visibility.md) instead measures
shared surface visibility across cameras at the same timestep.

`generation_config.json` records the base seed, generator version, capture-engine
identity, image size,
grammar, density, quality, augmentation, modalities, trajectory schedule,
storage encoding, and geometric conventions. Each sample carries its full
scene manifest, camera calibration, and per-sample `indoor_render_metadata`
with effective lighting/GI settings, transport backend/statistics, annotation
policy, and capture transport. GI settings and the renderer/dependency protocol identity are part of the resume contract.
Native capture uses CPU light clustering for reproducible accumulation; each
sample records the actual `light_clustering` policy. Native capture compiles
pipelines synchronously on the render thread to finish compiler tasks before
worker shutdown. GPU submissions/readback and the CPU writer remain asynchronous;
`pipeline_compilation` records this distinction.
A dataset made by an older capture engine remains readable, but appending with an
incompatible engine requires a new output directory. `--indoor-gi-rays` selects
64–16384 rays per probe: 256 is the efficient default; 1024 reduces measured
probe integration noise at higher GPU cost. Portable mode omits GI.
Successful finite generation also
writes `metrics/metrics.json`, scene/object/camera CSVs, `windows.csv`, and SVG distributions
and heatmaps for the **complete exported seed population**. Disable these
post-generation statistics with `--indoor-metrics=false` when unnecessary.
Planned object counts are distinct from projected visible counts. Heatmaps
use unaugmented room coordinates, with neighbor-room instances counted
separately; see the policies embedded in the metrics JSON. Exterior metrics count
primary-room facade walls/openings separately from the internal glass partition.
`windows.csv` records wall-local rough openings, mullion columns, frame/inset
dimensions, shade coverage and near-full-height classification. Architecture
metrics and `architecture.jsonl` record polygon outlines, roof slopes, floor
patches, pillars, arches and mezzanines; see the [architectural evaluation](architecture_v22.md).

Single and paired swing doors record their installation in `program.swing_hanging`.
The leaf closes against stops inside the frame. The stationary hinge pin and the
moving leaf share the rotation axis used for motion, navigation and schematics.
The metrics export records face setback, stop projection, hinge length and hinges
per leaf. Older door records with this field absent replay their original geometry.

For doors with vision panels, `door_vision_shape` counts the openings in individual
leaves. Width, height, center height, corner radius and cutout area describe the
installed opening before the bead and gasket are fitted. The optional
`program.vision_light` records shaped glazing stock. An absent field on a vision
panel gives the original square-cornered opening.

Exposed masonry replaces the room-side wall face up to its finish line. Fired
brick and concrete block have metric course pitches, recessed mortar, and
separate mineral finishes. `wall_lining_stock` and `wall_lining_arrangement`
count installed walls. The latter distinguishes stacked, alternating, and
left- or right-advancing third-lap courses. Lining dimensions and joint widths
and depths are in metres. Both masonry and mortar carry the wall semantic.

The `rug_cut` and `rug_binding` distributions describe installed products.
`rug_binding_width_m` measures the physical edge band. For sink mats,
`sink_mat_toe_gap_m` measures the distance from the nearest rotated corner
to the cabinet front. Lateral offsets and yaw are relative to the sink.

Chair seat construction distinguishes crowned upholstery (0), upholstered waterfall
edges (1), separate front bolsters (2), pressed plywood (3), carved timber (4)
and moulded polymer (5). Formed seats report centre thickness, lateral and rear
rise, front drop and signed width taper. A positive taper narrows the rear.
Seats from the same table order can share a shell design, with individual
grain cuts and occasional replacement chairs. The centre of a formed seat is
0.47 m above the floor. For stools, `stool_seat_height_m` is the installed seat's
centre height, and `stool_seat_to_rail_drop_m` measures the drop to the top of
its footrest. Low tables, dining tables and counter-height worktops can have
different stool heights. Cafés and cafeterias can mix high tables with stools
and ordinary dining sets in one room. A seated person's pose records the stool
support in local metres, including whether the front footrest is curved.

For sled and cantilever chairs, `chair_runner_section` distinguishes tubular
metal from laminated bent plywood. Width, thickness and nominal bend radius
are in metres. Tight frames limit the bend at the seat to keep both bearings
supported. A positive `chair_runner_front_rake_depth_fraction` means the
front upright leans toward the rear as it rises to the seat. Floor glides
fit the straight portion of each runner. These parts share the chair instance
and semantic label in every view.

Face-up phones report their camera mount (bezel, display punch or sensor pill),
its horizontal position and width. Face-down phones report the optical cluster
and flash location. `phone_individual_lens_radius_m` counts each installed rear
lens once. `phone_display_state` counts visible sleeping, clock-only
and active displays. `screen_luminance` records zero for a sleeping phone.
Sleeping and clock-only phones are excluded from handset application-layout counts.

Wall clocks share a seeded local time consistent with the room's sampled daytime
or evening lighting. Individual quartz movements can drift up to eight seconds.
The distribution report records `clock_time_seconds`, `clock_dial_style`, and the applicable
`clock_minute_rail` or `clock_24_hour` setting. These are snapshot times. Analog
hands and bezels contribute real depth. Printed indices, reflective LCD digits,
and lit LED segments share the dial surface and its clock semantic label.

Projection screens record their stock format, mounting mechanism and deployment.
Image width and height measure exposed cloth. Both are zero when the screen is
stowed. Full image height and stock format describe the fully deployed image.
Placement checks sightlines from the primary viewing area independently of the
capture cameras. Large teaching and conference rooms can have paired screens
on a shared mounting line. Standby tabletop projectors require a fully exposed
screen and a clear optical path to its center.

Open notebooks report their fitted cover width and depth in metres, the fraction
of pages on the left, and the ruling shared by the two facing leaves. Both
leaves use the `paper` semantic class. Writing is part of the paper color map
and follows the same curved surface used for depth and normals.

Tabletop tissue dispensers report their casing, draw state and mouth dimensions
in metres. Folding cartons also report the installed print. Open dispensers
report the remaining pack height as a fraction of the casing and the exposed
draw height in metres. The case and cut edge use the box semantic label. The
refill and protruding sheet use the paper label and share the dispenser instance.

Banquette backrests report their installed construction. Sewn backs also report
foam thickness, tuck depth and channel or diamond repeat in metres. Their fabric
chart follows the padded surface across the seams. Seats, backs and brackets
share the bench semantic class and the furnishing instance.

Perimeter heaters report their installed construction in `heater_construction`.
Rooms can have a coordinated heating series or independently sourced units.
Pressed panels record channel pitch, relief and spacing between paired banks.
Tubular radiators record pitch and clear space between tubes. These distances
are in metres. Casings, floor supports and supply/return fittings belong to
the same appliance instance.

Rolling carts and mobile drawer pedestals report their fitted wheel diameter,
tread width and swivel trail in metres. The wheel and fixed-caster histograms
count matched sets per furnishing. Solid hubs, pierced spoke wheels and twin
casters use the same physical surfaces for RGB, depth and semantics.

Classroom tablet chairs report the writing surface's side, outline, yaw in degrees,
face and edge materials. `teaching_back_carry_opening` counts solid and pierced
backs among these chairs. Grip width and height in metres describe the clear
opening on the front skin. The RGB and geometry annotations use the same mesh
for the shell, its opening and the handed writing surface.

Sofas and upholstered armchairs report their installed frame type, stock,
number of discrete floor glides, and frame section widths and depths in metres.
A chaise depth change starts a separate frame. Cushion seams on a straight run
can share one frame. Plinths and pedestal discs bear directly on the floor and
contribute zero discrete glides. Recliner frames fit the stationary seat below the articulated back and footrest.

Freestanding room dividers report the installed arrangement, bay count and insert
thickness. `divider_hinge_turn_deg` records one signed turn per interior hinge,
including zero for a straight joint. A single-bay screen contributes no hinge samples.

`architecture.jsonl` also records the adjoining room's footprint and
`adjoining_elevation`. Its optional `windows.edge` indexes that footprint,
with aperture coordinates measured in metres from the edge midpoint along
the wall and up from the floor. The distribution report includes adjoining
exterior window count, rough opening area and sill height. Through-glass
annotations can see scenery beyond both the interior partition and these windows.

On the entry wall, clear glass, full frost, privacy bands, gradients, etched
stripes and ceramic markings share a finish across installed bays. End bays
can be cut to fit the jamb or corner. `shared_glazing_treatment` counts one
installation or solid wall per room. `shared_glazing_bay_width_m` measures each
rough opening and `shared_glazing_thickness_m` measures the installed glass
once per wall. Frost changes RGB scattering. Opaque ceramic markings also
appear in through-glass geometry annotations. A material-seed override
resamples the glass optics and frost strength with the installed layout,
pane thickness and marking positions held fixed.

To append another 100 samples, repeat the capture settings with `--resume
--samples 100`. The seed can be omitted on resume: the saved base seed is
restored. The sample count means additional samples, not a target total.
Resume verifies the capture contract, counts actual records in every chunk
(including partial chunks), and rejects missing chunk/sample indices or
incomplete records. New generation refuses existing indexed output. Writes
publish completed chunks or sample folders atomically. Failed or missing
modalities, non-finite buffers, unexpected seed/index ordering, and geometric
alignment failures stop generation; they do not silently skip scene seeds.

`--output-mode fs` produces one sample folder with `meta.safetensors`, the
manifest/color metadata, float32 NPZ image planes, and previews. Under the
default indoor `--color-codec raw`, color NPZ is authoritative and JPEG is only
a preview. `--color-codec jpeg` explicitly selects lossy JPEG quality 75 for
RGB storage. Semantic labels are always lossless float32 planes; palette PNGs
are previews. Existing legacy JPEG semantic files remain readable but their
lossy colors must not be treated as exact labels.

The renderer applies its tone map before capture. Both raw and JPEG exports
apply the sRGB transfer exactly once, without per-image contrast fitting.
Depth is linear camera-space Z in metres, normal is view-space `(n + 1) / 2`,
position is world position affine-normalized by the exported AABB, and semantic is
the linear RGB class palette. For indoor scenes, `aabb` encloses the primary room
and its structural shell and equals the default `ovoxel_aabb`, excluding exterior
context. Positions are not clamped: context outside the reconstruction box may
produce values outside `[0, 1]`. Decode with `min + position * (max - min)`.
Matrices are column-major `world_from_view`,
right-handed with camera forward `-Z`; vertical FOV is in radians. Annotation
precision is explicitly recorded per sample: `annotation_precision=1`
(`float32_geometry`) selects the native indoor MRT geometry pass, while `0`
(`float16_hdr`) identifies the legacy/fallback HDR intermediate. This field
records the render path, independently of float32 storage.

In the Auto indoor quality setting, rough architectural glass filters the opaque
HDR background before tone mapping. Each camera builds its own view of that
background. Near objects and window frames use the depth prepass where
available. Clear glass retains its sharp refraction. The surface/through glass
annotation policy still determines which physical intersection supplies depth,
normals, semantics and co-visibility.

Procedural people use an independent seeded stream. `--indoor-human-density`
defaults to `0.25`; zero disables people without changing room furnishings.
The manifest records pose, chair support, stature/build, clothing, skin/hair
palettes, and local joints. `human_instance_ids` align the human axis with
manifest IDs; `human_count` excludes padded rows in a chunk. Pose positions
are world coordinates. A static person's joints stay fixed as the camera
moves. For a moving person, the joints follow the sampled motion timestep.
The 21-joint bone names and parent indices are exported with poses.
`object_obb_instance_ids` associate furniture and person oriented boxes with
manifest IDs; `-1` indicates padding or an unlabelled legacy instance.
People use the bundled AnnyBody reference. Static poses sample forearm and shin
length relative to total reach. The same segment lengths constrain supported
hands. Clothing fits the shaped torso, with a curved span between chest supports,
projected button bands, and separate dye and grain for trousers and inner shirts.
Pigmentation, lip sheen and brow shape vary independently of outfit. Shoes use
mirrored lasts with toe rise, an instep and laced or slip-on uppers. The manifest
records the appearance programs and the metrics export includes segment lengths,
skin undertone, lip moisture and shoe shape. Each fitted haircut samples temple
recession, peak and nape height. Combed cuts can expose a curved part in the
actual scalp geometry. Clipper length and fibre density determine the visible
scalp fraction of a buzz cut. Loose cuts vary how much of each ear they cover.
Their locks and underlayers follow the circumference of the actor's head.
The metrics export includes the applied part
width in rest metres and coverage for clipped hair. Spectacles fit to both
eye centres, the nasal bridge and the ears. Full rims, upper rims and
drill-mounted lenses vary in silhouette, wrap, tilt and finish. The manifest
stores the sampled eyewear program. Distribution metrics count spectacles worn
by people in the room and report their frame dimensions, roughness and lens
opacity. The frame and lens follow the head during playback. Garments follow
the rig during playback without cloth simulation.

The native CLI uses a rendezvous CPU writer: one batch may be encoding while
the next is captured. There is no waiting batch queue. Maximum batch residency
is therefore two uncompressed batches, plus encoder workspace and GPU/render
assets. Choose `--chunk-size` accordingly: five RGBA32 capture planes at
640×480 with four cameras and three steps require about 281 MiB per scene
before compression (and before metadata). A chunk size of 16 can therefore
consume several GiB per batch; smaller chunks reduce peak memory. The bounded CLI
writer/RSS check is reproducible with `scripts/qualify_indoor_writer.py`.

## Python capture and storage

`BevyZeroverseDataset(scene_type="procedural_indoor", indoor_seed=777, ...)`
uses dataset index `i` to request exactly `777 + i`. Random-access requests,
shuffling, repeated indices, and DataLoader workers therefore refer to the
same scene. Set `indoor_layout`, `indoor_density`, `indoor_human_density`, `indoor_quality`, and `indoor_gi_rays` directly;
the indoor depth default is linear. One engine/configuration is supported per
process. Use DataLoader's `multiprocessing_context="spawn"` when creating
workers after initializing a GPU engine in the parent.

`chunk_and_save(..., color_codec="raw")` preserves float32 sRGB color. It keeps
partial final chunks by default. Existing outputs require explicit
`append=True` and a dataset containing **only additional samples**; this is
append, not automatic dataset-index resume. Chunk iteration assigns every
chunk exactly once across workers, retains unequal final shards, and fails on
corrupt data unless `skip_corrupt_chunks=True` is explicitly requested.
Distributed training must handle unequal shard lengths explicitly rather
than relying on silent truncation. Cache fingerprints detect changed files.

`save_to_folders` and `FolderDataset` preserve all rows/channels, manifest,
calibration, semantic labels, and stored color encoding. NPZ labels are
authoritative. The old FFI `generate` example only implements the legacy
color/depth workflow and rejects indoor requests with a pointer to
`zeroverse_gen`.

## Qualification

CPU regressions are in `crates/burn/tests/indoor_roundtrip.rs` and
`crates/ffi/python/test_indoor.py`. The GPU qualification script below retains
its CLI logs, chunks, metadata, metrics, and final `report.json` in a new output
directory:

```sh
PYTHONPATH=/path/to/built/extension:crates/ffi/python \
  python crates/ffi/python/validate_indoor_dataset.py \
  --binary target/debug/zeroverse_gen --output out/dataset_qualification
```

It checks one versus two CLI workers, partial-chunk resume, rejected overwrite
and incompatible resume, native folder/Python interchange, out-of-order and
repeated indexed Python captures, and two spawned DataLoader workers. It uses
odd image dimensions (161 × 119), two cameras, three trajectory steps, and all
five RGB/geometric/semantic modalities. Temporal and co-visibility exports have
separate [flow](optical_flow.md) and [membership](co_visibility.md) regressions.
Pixel equality across GPU drivers or rendering backends is not promised.

The architecture distributions include the base treatment of each nearby building.
`exterior_plinth_stock` encodes continuous cladding (0), render (1), precast
concrete (2), fired clay (3) and sheet-metal flashing (4). Installed bases
report their top elevation and projection in metres. Unitized courses also
report nominal unit length. These measurements describe exterior context visible
through windows.

Recorded qualifications retain their configuration, build and adapter identity.
Re-run the commands above to qualify a deployment; archived results do not establish
current throughput, a memory plateau or cross-device determinism. The
[architectural evaluation](architecture_v22.md) reports the latest bounded scene
and annotation audit.
