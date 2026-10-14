# Doors, hardware and cabinetry

Architectural doors occupy real openings. Their serialized construction program is shared by mesh generation, static collision, camera visibility proposals, human path planning, schematics and annotation instances. Opening states are static by default. The opt-in [articulated room motion policy](indoor_motion.md) adds clearance-checked opening/closing on the capture timeline; it is kinematic animation rather than physical hinge simulation.

| Category | Construction and continuous variation |
|---|---|
| Hinged entrance/interior door | Left/right hand, opening angle, height/width/thickness, slab/recessed/glazed/vision-lite/louvered panels |
| Paired door | Opposed leaves and hinge-side hardware, independent finish from jambs and handles |
| Sliding door | Surface track, translated leaf, pull or recessed handle; lateral travel requires adjoining wall space |
| Telescoping door | Two leaves on offset tracks; differential travel and actual opening state |
| Revolving entrance | Wide lobby/exhibition entrance, curved side glazing, four radial leaves, centre shaft, continuous rotor phase |
| Office storage | Hinged ajar fronts, bounded bypass sliding fronts, drawer/mixed/open/slatted configurations |
| Kitchen base units | Drawer banks or hinged/sliding doors, slab/inset/raised/fluted fronts, visible shelves/contents |
| Overhead cupboards | Variable mounting height, width, depth and lateral offset; hinged, bypass or lift-up fronts |
| Pantry and sink bases | Matching cabinet construction with separately admitted leaf travel and supported contents |

Lever, knob, pull, panic-bar and recessed architectural hardware have real thickness and attachment geometry. Cabinet hardware includes bar, knob, compact pull, edge pull and touch-latch fronts. Timber, coated polymer, metal and chrome roles use the procedural material system; leaf, frame and hardware finishes are independently sampled. Revolving doors use a rotor phase instead of an opening fraction.

## Placement and annotation contract

The primary entrance moves along the shared rear wall; internal doors inherit the position and orientation of accepted partition openings. Archways remain possible. Doors are sampled before furnishing, and approach/travel regions reject incompatible furniture, people and architectural floor features. Sliding mechanisms require lateral space; unsupported interior proposals fall back to hinged construction. Revolving drums are conservative navigation obstacles: people and cameras cannot navigate between a static rotor's blades.

Cabinet placement envelopes include the full realized opening travel before admission. Worktop support bounds refer to the actual slab, not the additional space reserved for a projecting door. Lower and upper cupboards share a program, while upper dimensions and mounting height vary continuously. Office cabinets use a smaller bounded ajar angle within their existing compact footprint.

Procedural assembly roots preserve explicitly labeled mesh parts; their object-level class no longer overwrites child labels. Legacy scene hierarchies retain inherited labeling.

Semantic taxonomy 7 appends `door_frame`, `door_hardware` and `cabinet_hardware`, retaining existing class IDs and colors. Leaves remain `door`; cabinet fronts and cabinet glazing remain `cabinet`. Appliance fronts retain their appliance category. Architectural leaves, jambs and hardware are grouped into one `door#portalN` bounding-box instance. An empty arch frame is `door_frame`, not a phantom door. The configured glass annotation policy applies to glazed leaves and glazed cabinet fronts as it does to other glazing. Schematic leaf footprints show the actual opening state.

Serialized room manifests carry `entry_door` and each interior partition's optional `door` program. Topology exports include door state: the existence of a portal does not guarantee traversability. Metrics record door counts, mechanisms, panels, handle families, finishes, opening fractions, rotor phase, dimensions and positions. Legacy manifests without these fields still deserialize.

## Qualification and publication

`review_doors` uses production builders and the native capture pipeline. Its fixed protocol audits 64 consecutive samples in each of ten architectural/cabinet categories and captures six illustrative examples in aligned RGB, normal and semantic channels. These are construction/state examples, not a frequency estimate or a claim of independently multiplied combinations. Cabinet dimensions include opening travel; the protocol is not a fixed-envelope perceptual-volume comparison.

The Rust publisher's `door-study` command verifies the source identity, full inventory, selected sample metadata, mesh receipts, image sizes and asset hashes before constructing the gallery. The atlas also includes the paper section as a standalone `paper.tex` source. The full publication refresh includes this atlas and generates the corresponding paper figures and category section. Missing or stale evidence blocks publication. Existing local room studies are not relabeled as new captures.

```sh
cargo run --no-default-features --features multi_threaded --example review_doors -- --output out/doors/capture
cargo run -p bevy_zeroverse_publication -- door-study --input out/doors/capture --output out/doors/report
```

The local qualification at `out/doors/qualification.json` records:

- **640/640 construction samples** passed mesh, winding, normal, semantic and placement-envelope checks; **60 assemblies** were captured in three aligned channels (180 images).
- **640/640 dense public rooms** passed layout, three-camera and actual-geometry validation: 64 consecutive seeds for each of ten public programs, with no failed scenes discarded.
- **Eight room captures / 24 views at 512×512** passed native RGB, depth, position, normal, semantic and co-visibility checks. The two kitchen scenes were captured under both glass policies with identical geometry and cameras. Worst self-reprojection error was 0.000526 pixels; worst depth/position disagreement was 0.00000573 metres.
- Category-median CPU assembly construction ranged from **0.046 to 0.664 ms**; the largest category p95 was **0.769 ms**. These are diagnostic timings on a host concurrently running training; they exclude procedural texture synthesis, uploads and rendering and are not a full-generator speedup claim.
- **542 root tests passed**, with 28 explicitly ignored; **63 publication tests passed**. Native Clippy (including motion-enabled code), publication Clippy and the motion-enabled WASM viewer compile passed. WASM rendering was not runtime-qualified in this run.

The local interactive atlas is `out/doors/report/index.html`; its compiled paper section is `out/doors/report/paper.pdf`. These are focused current-source artifacts. The Rust publication templates now include the category viewer and paper section; a complete project-page/paper refresh still requires regenerating the other source-bound cohorts before publication. These checks do not establish photographic equivalence or architectural code compliance.
