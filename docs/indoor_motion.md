# Indoor scene dynamics

`BevyZeroverseConfig.indoor_motion` is an optional JSON policy. It is `None` by default. Native CLI, Rust/Burn consumers and the WebGPU viewer share this policy; no model or additional Cargo feature is required. The studio exposes it under **People & motion → Furnishing motion** and preserves it in shared URLs.

```json
{
  "chair_fraction": 0.35,
  "door_fraction": 0.65,
  "cabinet_fraction": 0.5,
  "covering_fraction": 0.5,
  "projection_screen_fraction": 0.4,
  "teaching_board_fraction": 0.5,
  "operation_return_fraction": 0.4,
  "fan_fraction": 0.8,
  "desk_fan_fraction": 0.5,
  "appliance_fraction": 0.5,
  "prop_fraction": 0.35,
  "cart_fraction": 0.5,
  "prop_pickup_fraction": 0.4,
  "prop_lift_m": 0.16,
  "screen_change_fraction": 0.35,
  "light_change_fraction": 0.2,
  "prop_travel_m": 0.25,
  "prop_rotation_degrees": 60.0,
  "cart_travel_m": 0.8,
  "fan_turns": [0.75, 12.0],
  "max_mechanisms": 12,
  "chair_travel_m": 0.65,
  "chair_swivel_degrees": 70.0
}
```

To request dense motion, set the desired category fractions to `1` and raise `max_mechanisms` (up to `128`). Travel and rotation are upper bounds, not minimum required displacements. A crowded room can contain many short motions, while genuinely obstructed, cable-connected or occupied objects remain stationary. Exported rejection reasons distinguish these cases from the budget limit. Static sofas, tables, plants, fixed installations are deliberately excluded.

For a multi-frame dataset, supply this as `--indoor-motion '<json>'` alongside the usual playback/capture options. `--indoor-motion '{}'` enables the defaults. Configuration edits take effect on Apply/regeneration, not while dragging a slider. Human motion remains a separately enabled policy.

## What can change in a clip

Every `ProceduralIndoor` layout, including kitchens, galleries, classrooms and offices, uses the same policy. What can move in a particular room depends on its actual fixtures, contents and free space. Cornell, Object, Human, Room and SemanticRoom support camera trajectories. Their optional Studio turntable rotates the scene root, and the dataset sampler suppresses this preview rotation. Their furnishings do not use the indoor motion planner.

| Subject | Temporal behavior | Admission or control |
| --- | --- | --- |
| Room and ceiling cameras | Fixed poses or calibrated trajectories | Indoor camera policy, independent of scene motion |
| People | Static poses or ARDY body and navigation trajectories | Separate human-motion policy and collision checks |
| Seating, carts and supported cargo | Roll, slide or swivel, with rigid cargo following the cart | Chair or cart probability, seat occupancy, cables and clearance |
| Doors, cabinets, appliances, laptop lids | Hinge, slide, fold or revolve around the constructed mechanism | Individual category probabilities, hardware travel and swept clearance |
| Curtains and blinds | Slats translate, fabric gathers or rolls | Covering probability |
| Projection screens | A spring or motorized screen winds into its cassette or lowers its weighted rail | `projection_screen_fraction` per retractable installation |
| Folding teaching boards | Outer writing wings pivot on their installed pins, including tilted stands | `teaching_board_fraction` per hinged wing, fitted stops and swept clearance |
| Ceiling and desk fans | The hub and blades spin around their installed shaft | Independent `fan_fraction` and `desk_fan_fraction`, shared `fan_turns` |
| Loose items | Translate or turn on their support. Selected hand-held items also lift and settle back onto it | `prop_fraction`, then `prop_pickup_fraction`, with metric travel and lift limits |
| Electronic displays | An active desktop, laptop, phone or TV advances to a second seeded page. An active ambient clock advances its displayed minute | `screen_change_fraction` per powered display with an image face |
| Powered lights | Task lamps and wired ceiling circuits brighten or dim smoothly. The emitting surface and analytic light share the same curve | `light_change_fraction` per individual lamp or installed circuit |

Pickup, display changes, light changes, desk-fan rotation, projection-screen travel, teaching-board travel and return cycles each default to zero. In a viewer URL use dotted keys such as `object_motion.screen_change_fraction=0.35`. Appearance changes have their own probabilities. Each installed source is sampled independently of camera visibility. Only physical mechanisms use the clearance-planning budget.

## Construction and playback

- Ordinary chairs, including visitor/dining/shell/tablet-arm chairs, backless stools and lounge armchairs, can slide or turn. Task/conference swivel chairs preserve their separate upper-seat mechanism. Occupied seats, recliners and fixed auditorium seats remain static.
- Hinged, paired, sliding, telescoping and revolving door leaves preserve their mechanical pivots, tracks, attached handles and semantic identities.
- Kitchen and office cabinet leaves retain separate hardware. Hinged, sliding and upper lift fronts can move when space permits. Fitted-counter drawers, island drawer/door bays and office drawer units include real drawer boxes and moving fronts; open shelving stays fixed. Bypass leaves use flush pulls that fit between their tracks.
- Refrigerator (including paired doors and attached notes), microwave, oven/range and dishwasher doors operate about their actual hinges. Laptop screens tilt/close while the chassis, ports and connected cables stay fixed.
- Portable items slide and turn on their existing support: phones, pencils, drink containers, dishware, stationery, desk accessories and countertop utensils/appliances. Bags, storage boxes and bins can move on the floor as well. Whole-object movement is disabled for connected wires or unsupported dependants. A wall/ceiling-mounted version does not become a loose prop.
- Wheeled carts carry their load as one clearance-tested unit. Cargo keeps its instance labels and boxes. When a cart is obstructed or tethered, eligible loose items on its shelves can move independently within the remaining mechanism budget. Each item must clear the actual shelf, its rim, other contents and any connected leads.
- Window coverings gather or deploy using stable topology and metric UVs. Rigid blind slats retain their orientation and dimensions while translating; rail/support geometry remains attached. Incompatible or blocked deformation proposals remain static.
- Teaching-board wings carry both writing surfaces, trimmed edges and the moving hinge straps. The center panel, stand, fixed hinge knuckles and pen tray stay in place. Each folding wing can open or close from its installed angle.
- Projection screens start at their sampled installation height. A stowed screen can deploy, a presented screen can retract, and a partially lowered screen can travel either way. The border and image pass through the cassette slot at their physical scale. The weighted rail and pull loop translate with the cloth. Clearances can shorten the stroke or leave the screen parked. Fixed frames stay mounted.
- A running desk fan turns its blades and hub on the sampled head axis. The base, motor, guard and bracket stay in place. Clearance covers every blade phase, including high speeds and reversed playback. The small shaft that meets the hub is the sole contact exemption. A running rotor takes priority over sliding the whole desk fan. An admitted moving cart carries a fan rigidly with its other cargo.
- Ceiling fans have stationary canopies/downrods and a rotating hub/blade assembly. Seeded geometry varies blade count (2–9), pitch, sweep, taper, span, housing, height, finishes and placement. The `ceiling_fan` semantic class and object boxes include the complete fixture; fan geometry also exists when motion is disabled.

`operation_return_fraction` controls how often admitted doors, cabinet fronts and drawers, appliance doors, laptop lids and teaching-board wings reach their stop, pause and return to their initial pose during **forward** playback. The hold and the speeds in each direction vary by installation. Linked telescoping leaves share their timing. Returning uses the same cleared stroke and spatial reservation in reverse. Revolving doors, fans, window coverings, projection screens and loose objects keep their own motion programs.

A pickup uses a seeded time window with a smooth start and stop. A phone, pencil, notebook, cup or other selected hand-held item may translate and turn during the lift. Its height returns to the support at both endpoints. The continuous sweep includes the arc between tested poses. The planner can shorten the lift below an overhead shelf, try a different direction, or keep a tabletop gesture planar. Lower shelf contents, support boundaries, other motion, occupied space and connected leads still constrain admission.

Chairs, portable items, carts, appliance doors, cabinets and coverings sample absolute normalized timeline progress with seeded start/end intervals and smooth acceleration. Fans rotate linearly across that shared timeline, with seeded direction and a uniformly sampled number of revolutions from `fan_turns`. Set its maximum to zero to disable spinning. When camera `duration_seconds` is assigned, the reference fan rate is `60 × turns / duration_seconds` RPM. Viewer playback speed/easing retimes the shared timeline, including fans; it does not change captured trajectories. Scrubbing, reversed playback and synchronized multi-camera capture use the same state, independent of FPS. Physical RPM requires a timeline duration. With evenly spaced identical blades, two frames separated by one blade pitch can show the same silhouette. A half-pitch step is useful for inspecting motion and flow.

With motion disabled, the normal batched static builders are used. No temporal BVH, swept volumes, mesh splitting, model initialization or animation mesh work runs. With motion enabled, rigid parts update transforms only; deforming coverings update their bounded mesh subset. Stationary timeline samples are skipped.

## Clearance contract

After static construction, admission checks complete conservative swept volumes against the constructed triangle geometry, continuous camera path segments, static people and previously accepted moving mechanisms. Cubic camera paths include a curvature error bound between samples. Thin cabinet/appliance/laptop parts use triangle endpoint convex hulls expanded by a rotation error bound, avoiding false collisions from empty space in swept boxes. Doorway hinge/frame contacts and explicitly tagged laptop hinge barrels are intentional mechanical attachments; the rest of each carcass remains a collision obstacle. Floor-supported motion must remain on the same floor level. Sliding props must remain fully on their original surface throughout the entire path; the check includes rotated footprints and shaped table/counter outlines. Only coplanar contact triangles of that specific support are exempted from intersection tests, not its sides or other geometry.

Travel/swivel settings are upper bounds. A blocked proposal tries progressively smaller motions; chairs, carts and supported props additionally try alternate headings, translation only, and swivel only. Door/cabinet motion stays within the requested mechanical interval, and shortened cloth motion must preserve triangle orientation throughout interpolation. Fully blocked or occupied mechanisms remain static. The bounded search tests at most 24 candidates per movable body and six per hinged mechanism. Fan clearance uses the complete rotor disk at any speed: slowing a colliding rotor is not a safe fallback.

The mechanism budget is shared across categories with seeded priorities, so construction order does not always give doors priority over chairs. Fractions select eligible mechanisms; a four-leg chair moves as a complete body rather than swivelling its back independently. The shared budget accepts 1–128 independent mechanisms. Carried cargo does not consume additional mechanism slots. Disabled fractions and zero amplitudes do not acquire motion through fallbacks.

Accepted travel regions are reserved for the **entire timeline**. Human navigation uses these reservations; dressed pose admission additionally bounds the continuous skinning interpolation between motion frames. Consequently, changing playback speed, reversing time or jumping directly to a capture timestep cannot make a previously valid motion unsafe. This is intentionally conservative: the planner does not schedule a person to pass through a door only during its open interval. Requested fractions are proposals, not guaranteed acceptance rates.

This is kinematic procedural animation, not a force/cloth/contact physics simulation. It does not animate a seated person together with their chair or infer who operates a door.

## Annotations and capture

`IndoorManifest.indoor_motion` records policy, accepted/rejected mechanisms, seeded intervals, scene-local rigid parameters and swept reservations. Report version 8 records requested and admitted translation, rotation and peak pickup height, candidate attempts, `carried_object_ids` for cart contents, actual moving-part `semantic_labels`, and the accepted adaptation (`shortened`, `rerouted`, `translation_only`, `swivel_only`, `pickup_only`, or `surface_only`). `MechanismReport::rigid_transform_at` samples the admitted rigid scene-local delta, including the full pickup arc and constant-rate fan rotation. An admitted cart lists the instances that follow its scene-local delta in `carried_object_ids`. A parked cart lists none. Each independently moving item on it has its own report. For a retractable screen, `projection_screen_travel` reports the initial, requested and admitted exposed cloth lengths in metres. `projection_screen_drop_at(progress)` samples the admitted stroke, including a parked installation. The captured geometry contains the individual fabric correspondences.

For a returning mechanism, `operation_cycle` records `starts_at`, `reaches_stop_at`, `leaves_stop_at` and `returns_at`, all in normalized scene time. Angle and translation describe the full admitted excursion. `stroke_fraction_at(progress)` gives the fraction actually traversed, including the hold and the return. Sample the interior of the interval to observe movement. Capturing only progress 0 and 1 gives two identical rest poses for a returning mechanism. Older reports without the field retain their one-way timing.

The report's `appearance.screens` entries contain instance ID, page seeds and switch progress. Its `appearance.lights` entries identify the object or indexed ceiling mount and expose `curve.multiplier_at(progress)`. Rewinding the clock restores the original page and emitter power. Page images are prepared once for the room. No texture is regenerated or read back for a timeline step. Screen brightness and cover optics belong to the device and persist across its page change.

Screen content and illumination affect RGB. A stationary monitor showing a new page retains the same depth, normal, semantic class and geometric correspondence. Shadows of physically moving objects do change. The static indirect-light and environment probes use the minimum flux over each animated source's interval. Direct lights and their visible apertures follow the instantaneous flux. The room does not rebake indirect lighting per timestep.

Part entities, semantic labels, mesh topology and annotation order remain stable through playback. Window coverings recover their authored normals and tangent frames at both complete stroke endpoints, including after a rewind. Projection cloth retains its original weave scale as it passes through the cassette. Object bounding boxes are recomputed from the current parts. RGB, depth, position, normals, semantic masks and co-visibility therefore use the same current geometry. Dataset optical flow retains surface correspondence across the captured timesteps, including deforming coverings. The existing interactive prepass flow preview is a velocity visualization; exact dataset flow remains the capture API's contract.

Preparation completes temporal clearance before publishing the scene. When human motion is also enabled, capture additionally waits for its validated clips. Animated parts are excluded from static indirect-light occluder baking; realtime shadows still follow them. Portable lamp emitters follow the same rigid delta as their geometry, including when carried on carts. Indirect illumination remains approximate. O-voxel export rejects any enabled room-motion policy, including a single-timestep request, just as it rejects human motion.

## Current limits

Desk-fan heads retain their installed pitch and yaw. Analog clocks retain their initial hand positions. Sun and sky stay constant over the room clip. Pickups preserve clearance, but no hand is assigned to operate them. A dataset requiring observed hand–object contact needs a separate interaction constraint.
