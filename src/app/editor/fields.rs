//! A single control catalog feeds Feathers widgets and coverage tests.
use super::model::{EditorState, Page};
use serde_json::{json, Value};

#[derive(Clone)]
pub enum Kind {
    Number(f32, f32, f32),
    Toggle,
    Choice(Vec<(String, Value)>),
    Text,
    Json,
}
#[derive(Clone)]
pub struct Field {
    pub path: String,
    pub label: String,
    pub help: String,
    pub kind: Kind,
}
pub enum Item {
    Section(&'static str, &'static str),
    Field(Field),
    Note(String),
    LensDistribution,
    CapacityEvidence,
}
fn field(path: &str, label: &str, help: &str, kind: Kind) -> Item {
    Item::Field(Field {
        path: path.into(),
        label: label.into(),
        help: help.into(),
        kind,
    })
}
fn num(path: &str, label: &str, min: f32, max: f32, step: f32, help: &str) -> Item {
    field(path, label, help, Kind::Number(min, max, step))
}
fn toggle(path: &str, label: &str, help: &str) -> Item {
    field(path, label, help, Kind::Toggle)
}
fn choices(path: &str, label: &str, options: &[(&str, &str)], help: &str) -> Item {
    field(
        path,
        label,
        help,
        Kind::Choice(
            options
                .iter()
                .map(|(label, value)| (label.to_string(), json!(value)))
                .collect(),
        ),
    )
}
fn text(path: &str, label: &str, help: &str) -> Item {
    field(path, label, help, Kind::Text)
}
fn json_field(path: &str, label: &str, help: &str) -> Item {
    field(path, label, help, Kind::Json)
}
fn section(title: &'static str, help: &'static str) -> Item {
    Item::Section(title, help)
}
pub fn items(state: &EditorState) -> Vec<Item> {
    let d = &state.draft;
    let indoor = d["scene_type"] == "ProceduralIndoor";
    let mut v=match state.view.page {
        Page::Scene=>vec![
            section("GENERATION", "Changes apply when you press Apply or Next."),
            choices("/scene_type","Scene type",&[("Procedural indoor","ProceduralIndoor"),("Zeroverse objects","Object"),("Semantic room","SemanticRoom"),("Primitive room","Room"),("Cornell cube","CornellCube"),("Human","Human"),("Custom","Custom")],"Every existing scene mode remains available."),
            text("/indoor_seed","Room seed","Exact 64-bit integer. Empty chooses a random seed on generation."),
            choices("/indoor_layout","Room program",&[("Mixed · all room types","Mixed"),("Conference","Conference"),("Open office","OpenOffice"),("Coworking","Coworking"),("Training","Training"),("Breakroom","Breakroom"),("Kitchenette","Kitchenette"),("Kitchen","Kitchen"),("Elevator landing","ElevatorLanding"),("Cafe","Cafe"),("Cafeteria","Cafeteria"),("Classroom","Classroom"),("Lecture hall","LectureHall"),("Showroom","Showroom"),("Gallery","Gallery"),("Public suite","PublicSuite"),("Reception","Reception"),("Workshop","Workshop"),("Lounge","Lounge"),("Library","Library"),("Studio","Studio")],"Mixed samples all 20 room programs uniformly, including kitchens, cafes, classrooms, galleries and public suites. Choose a specific program to restrict generation."),
            num("/indoor_density","Furnishing density",0.,1.,0.01,"Secondary furniture, plants and tabletop clutter."),
            num("/indoor_human_density","People density",0.,1.,0.01,"0 removes people; 1 requests the largest supported population."),
            toggle("/rotation_augmentation","Randomize world orientation","Dataset augmentation applied during generation."),
            section("AUTOMATION", "Manual generation is recommended while editing."),
            num("/regenerate_ms","Next scene interval (ms)",0.,60000.,1000.,"0 disables automatic generation. Waits for construction to finish."),
        ],
        Page::Cameras=>vec![
            section("MULTI-VIEW RIG","Spacing and overlap are constrained together."),
            num("/num_cameras","Capture views",0.,16.,1.,"0 uses only the editor view. Co-visibility supports up to 16 views."),
            toggle("@multiview","Coordinate views","Require shared geometry with a reference camera or a connected camera graph."),
            num("@camera_motion_percent","Moving cameras (%)",0.,100.,1.,"0 fixes every capture pose; 100 gives every camera a trajectory. Intermediate values seed a per-view probability; small rigs may differ from the percentage. Travel controls below apply only to moving views."),

        ],
        Page::Appearance=>vec![
            section("SURFACE & LIGHT VARIATION","Independent seeds preserve layout, geometry and cameras."),
            text("/indoor_appearance/material_seed","Material seed","Empty follows the room seed; set an integer to lock surfaces."),
            text("/indoor_appearance/lighting_seed","Lighting seed","Empty follows the room seed; changes sun, sky and electric lighting."),
            num("/indoor_appearance/material_detail","Material relief & contrast",0.,1.,0.01,"1 retains full procedural PBR detail."),
            num("/indoor_appearance/illumination_scale","Illumination multiplier",0.05,1.,0.01,"Uniformly dims sun, sky, electric and ambient lighting."),
            num("/indoor_appearance/exposure_ev100_offset","Exposure offset (EV100)",-2.,2.,0.05,"Positive values darken the image. Negative values brighten it."),
            section("RENDER QUALITY","Full supported effects are enabled by Auto."),
            field("/ssaa_factor","Capture supersampling (SSAA)","Auto uses 2× below 1280×720 pixels, otherwise native resolution. Geometry labels retain exact final pixel-center rays. Apply to rebuild targets.",Kind::Choice(vec![("Auto · 2× below 720p".into(),json!(0)),("Off · native resolution".into(),json!(1)),("2× · 4 RGB samples / pixel".into(),json!(2)),("3× · 9 RGB samples / pixel".into(),json!(3)),("4× · 16 RGB samples / pixel".into(),json!(4))])),
            toggle("/indoor_gpu_materials","Bake materials on GPU","Caches the same PBR recipes on the rendering device. Unsupported programs and devices use CPU synthesis."),
            choices("/indoor_quality","Quality",&[("Auto · platform supported","Auto"),("Portable · reduced effects","Portable")],"Portable omits shadows, GI, SSAO, bloom and refraction."),
            num("/indoor_gi_rays","Indirect-light rays / probe",64.,16384.,64.,"Native Auto only. More rays reduce bake noise and take longer."),
            Item::Note(if cfg!(target_arch="wasm32") {"WebGPU Auto retains shadows, bloom and glass refraction. GI and SSAO are native-only."} else {"Lighting is baked during generation. Apply to see changes."}.into()),
        ],
        Page::People=>vec![section("POPULATION","Body shape, skin, hair and clothing are seeded with the room."),num("/indoor_human_density","People density",0.,1.,0.01,"Static people do not load motion models."),toggle("@motion","Generate human motion","Only loads cached ARDY models after applying an enabled policy." )],
        Page::View=>vec![
            section("PREVIEW","Changes here update the current scene immediately."),
            choices("@viewport","Viewport",&[("Editor camera","editor"),("Capture grid","grid"),("Room schematic","schematic")],"Schematic shows metric primary-room footprints, live capture cameras, paths and human poses."),
            choices("/render_mode","Annotation",&[("Color / PBR","Color"),("Depth","Depth"),("Surface normals","Normal"),("World position","Position"),("Semantic classes","Semantic"),("Optical flow","OpticalFlow"),("Motion vectors","MotionVectors"),("Co-visibility","CoVisibility")],"Co-visibility uses capture views, excluding the editor camera."),
            choices("/depth_format","Depth visualization",&[("Grayscale","Normalized"),("Turbo · warm near / cool far","Turbo"),("Original color ramp","Colorized"),("Linear distance","Linear")],"Turbo uses log distance for indoor detail, with a fixed scale per camera. Linear and normalized numeric depth encodings are unchanged."),
            choices("/annotation_glass","Glass in annotations",&[("Glass surface","surface"),("Geometry behind glass","through")],"Depth, normals, position, semantic, flow and co-visibility use the same unrefracted geometric hits. RGB is unchanged."),
            num("@editor_fov","Editor field of view (degrees)",15.,150.,1.,"Editor lens only; capture intrinsics remain the seeded dataset cameras."),
            toggle("/gizmos","Camera frusta & trajectories","Capture cameras only."),
            toggle("/draw_obb_gizmo","Object bounding boxes","Semantic object envelopes."),
            toggle("/draw_pose_gizmos","Human skeletons","Visible through clothing, occluded by room geometry."),
            num("/gizmos_alpha","Overlay opacity",0.,1.,0.01,"Opacity for camera gizmos."),
            section("TIMELINE","Progress spans the complete generated camera and human trajectories."),
            num("@progress","Scrub time",0.,1.,0.001,"Scrubbing pauses playback. 0 is the start; 1 is the endpoint."),
            choices("/playback_mode","Playback curve",&[("Paused","Still"),("Once","Once"),("Loop","Loop"),("Ping-pong","PingPong"),("Sine","Sin"),("Ease in","EaseIn"),("Ease out","EaseOut"),("Ease in / out","EaseInOut"),("Cubic in","EaseInCubic"),("Cubic out","EaseOutCubic"),("Cubic in / out","EaseInOutCubic")],"Sine and ping-pong reverse motion; Once keeps natural forward motion."),
            num("/playback_speed","Progress / second",0.,2.,0.01,"0.2 completes a forward trajectory in 5 seconds."),
            num("/yaw_speed","World rotation (rad/s)",-1.,1.,0.01,"Rotates the entire scene; separate from camera travel."),
            section("OPTICAL FLOW","Viewer visualization scale; export uses exact capture timestep displacement."),
            num("@flow_interval","Preview interval (seconds)",0.005,0.5,0.005,"Velocity multiplied by this fixed interval, independent of frame rate."),
            num("@flow_scale","Full color scale (pixels)",1.,256.,1.,"A visualization scale, not a change to motion annotations."),
        ],
        Page::Advanced=>vec![
            section("COVERAGE EVIDENCE","Read-only qualification. Imported evidence never changes the scene or starts generation."),
            Item::CapacityEvidence,
            json_field("@capacity_receipt","Import capacity receipt JSON","Paste a qualified receipt, or null to clear it. Relevant control edits mark it mismatched; receipts are kept out of shared URLs."),
            section("EXACT PARAMETERS","JSON editors expose the complete validated public policy, including arrays."),
            json_field("@config","Complete viewer configuration","Every public CLI/query setting. Edits are validated before Apply."),
            json_field("/indoor_camera","Camera policy JSON","Includes overlap bands, handheld increments and capture duration."),
            json_field("/indoor_appearance","Appearance policy JSON","Material and illumination variation."),
            json_field("/indoor_motion","Furnishing motion policy JSON","Null disables articulated motion. Swept clearance is checked before the scene is ready."),
            json_field("/human_motion","Motion policy JSON","Null disables motion. Custom prompts and frame-indexed waypoints are supported."),
            section("CAPTURE & EXPORT","Capture options are preserved in shared links; the viewer previews their effects."),
            num("/width","Capture width (px)",64.,2048.,16.,"Changes capture targets on Apply; does not resize the window."),
            num("/height","Capture height (px)",64.,2048.,16.,"Capture height is independent of the editor viewport."),
            num("/playback_steps","Capture timesteps",1.,64.,1.,"O-voxel requires one timestep."),
            num("/playback_step","Capture time increment",0.001,1.,0.001,"Normalized time between captures."),
            choices("/ovoxel_mode","O-voxel",&[("Disabled","Disabled"),("CPU async","CpuAsync"),("GPU compute","GpuCompute")],"Primary-room geometry only; requires one timestep and no human or furnishing motion."),
            num("/ovoxel_resolution","Voxel resolution",16.,512.,16.,"Higher resolution increases compute and storage."),
            text("/ovoxel_max_output_voxels","Maximum output voxels","Integer buffer limit."),
            toggle("/z_depth","Use camera Z depth","Otherwise measures ray distance."),
            section("EDITOR NAVIGATION","These affect the editor camera only."),
            num("/orbit_smoothness","Orbit smoothing",0.,1.,0.01,"0 is immediate; higher values ease motion."),
            num("/pan_smoothness","Pan smoothing",0.,1.,0.01,"Mouse navigation is disabled over controls."),
            num("/zoom_smoothness","Zoom smoothing",0.,1.,0.01,"Editor intrinsics persist across regeneration."),
            toggle("/keybinds","Enable shortcuts","R: next room; Space: play/pause. Text fields capture keyboard input."),
        ],
    };
    if state.view.page == Page::Cameras {
        if !d["indoor_camera"]["multiview"].is_null() {
            v.extend([
                num("@baseline","Camera baseline",0.,1.,0.01,"0 = close stereo; 1 = wider spacing and more varied visible targets. Accounts for camera count; overlap and clearance remain required. Replaces advanced values below."),
                section("ADVANCED RIG","Custom constraints override the baseline program. Impossible requests report an error."),
                choices("@overlap_graph","Visibility connections",&[("Reference camera","reference"),("Connected graph","connected")],"A connected graph allows overlap through intermediate cameras. The same spanning edges must pass along the paths. Explicit overlap strata require Reference camera."),
                num("/indoor_camera/multiview/min_overlap","Minimum proxy overlap",0.,1.,0.01,"Placement estimate; measured rendered overlap may differ."),
                num("/indoor_camera/multiview/min_baseline","Minimum pair spacing (m)",0.01,10.,0.01,"Between every pair of views."),
                num("/indoor_camera/multiview/min_reference_baseline","Minimum distance to view 0 (m)",0.,20.,0.05,"Metric reference baseline."),
                num("/indoor_camera/multiview/max_baseline","Maximum distance to view 0 (m)",0.01,30.,0.05,"Upper bound in room space."),
                num("/indoor_camera/multiview/min_spread","Group spread",0.,1.,0.01,"For 3+ views: discourages a line of cameras."),
                num("/indoor_camera/multiview/trajectory_variation","Independent path variation",0.,1.,0.01,"0 permits rigid shared camera motion; 1 requests varied paths."),
                num("/indoor_camera/multiview/target_variation","View coverage variation",0.,1.,0.01,"0 keeps shared look targets. Higher values sample visible surface regions and prefer distinct, useful viewpoints; selection accounts for camera count."),
                toggle("@mixture","Sample overlap strata","Mix high, low and no-overlap pairs instead of one overlap threshold."),
            ]);
            if !d["indoor_camera"]["overlap_mixture"].is_null() {
                for (i, label) in [
                    "High-overlap weight",
                    "Low-overlap weight",
                    "No-overlap weight",
                ]
                .iter()
                .enumerate()
                {
                    v.push(num(
                        &format!("/indoor_camera/overlap_mixture/weights/{i}"),
                        label,
                        0.,
                        1.,
                        0.01,
                        "Relative sampling weight; all three must not be zero.",
                    ));
                }
            }
        }
        v.extend([
            section("LENSES", "Vertical perspective FOV; wide lenses retain full calibration and annotation support."),
            num("/indoor_camera/intrinsics/fov_y_min_degrees","Minimum vertical FOV",15.,150.,1.,"Lower bound for capture lenses. Horizontal coverage also depends on image aspect."),
            num("/indoor_camera/intrinsics/fov_y_max_degrees","Maximum vertical FOV",15.,150.,1.,"145° includes ultra-wide perspective lenses. Near-plane clearance is checked across the complete camera path."),
            num("@lens_bias","Lens preference · narrow / wide",-1.,1.,0.01,"Continuously shifts the distribution toward narrow (−1) or wide (+1) lenses. Independent of spatial baseline. Editing uses the continuous distribution."),
            num("@lens_spread","Lens spread",0.,1.,0.01,"0 fixes every lens at the median; 1 spans the full FOV range. Continuous PDF in log focal space; placement and overlap can bias accepted lenses."),
            Item::LensDistribution,
            num("@fisheye_probability","Fisheye camera probability",0.,1.,0.01,"0 keeps all cameras pinhole. Assignment is seeded per camera before visibility planning; lens parameters stay fixed through time."),
            num("@lens_x","Principal-point variation X",0.,0.2,0.005,"Seeded optical-center displacement, as a fraction of image width. Zero keeps the centered pinhole."),
            num("@lens_y","Principal-point variation Y",0.,0.2,0.005,"Seeded optical-center displacement, as a fraction of image height. Full K is exported for each view."),
        ]);
        if !d["indoor_camera"]["lenses"].is_null() {
            v.extend([
                num("/indoor_camera/lenses/fov_degrees/0","Fisheye minimum diameter FOV",40.,220.,1.,"Angular domain; diagonal or circular sensor fit. Not a perspective vertical FOV."),
                num("/indoor_camera/lenses/fov_degrees/1","Fisheye maximum diameter FOV",40.,220.,1.,"Values above 180 degrees include rearward rays."),
                num("/indoor_camera/lenses/circular_probability","Circular image probability",0.,1.,0.01,"Contained circular image with equal focal axes. Circular + elliptical probability must be at most one."),
                num("/indoor_camera/lenses/elliptical_probability","Elliptical image probability",0.,1.,0.01,"Contained ellipse with sampled focal-axis scaling. Remaining probability uses full-frame / diagonal fisheye."),
                num("/indoor_camera/lenses/distortion_strength","Angular distortion variation",0.,1.,0.01,"0 is equidistant; 1 samples a bounded monotonic polynomial family."),
                num("/indoor_camera/lenses/anisotropy/0","Fisheye minimum aspect scaling",0.8,1.25,0.01,"Sensor focal-axis anisotropy, independent of image aspect."),
                num("/indoor_camera/lenses/anisotropy/1","Fisheye maximum aspect scaling",0.8,1.25,0.01,"Full pixel calibration is exported."),
            ]);
        }
        v.extend([
            section("CAMERA MOTION","Independent of furniture/human animation and camera spacing. Intrinsics remain fixed through time."),
            choices("@path_preset","Trajectory preset",&[("Custom","custom"),("Short handheld","handheld"),("Explore the room","explore")],"Presets set travel bounds; all fields remain editable."),
            num("/indoor_camera/path_length_min","Minimum travel (m)",0.,20.,0.05,"Minimum path length for moving cameras. Use Moving cameras = 0% to freeze poses without losing travel settings."),
            num("/indoor_camera/path_length_max","Maximum travel (m)",0.,30.,0.05,"Swept collision checks always apply."),
            num("/indoor_camera/long_path_fraction","Long-route probability",0.,1.,0.01,"Fraction of proposals that explore farther through the room."),
            toggle("/indoor_camera/primary_room","Stay in primary room","Bounds capture trajectories to the reconstruction target."),
            toggle("@handheld","Independent handheld increments","Translation and yaw/pitch/roll sampled independently from the shared look target."),
            text("/indoor_camera/duration_seconds","Physical duration (seconds)","Empty leaves physical time unspecified. Positive number maps normalized progress to seconds."),
        ]);
        if !d["indoor_camera"]["handheld"].is_null() {
            for (key, units, names, bound) in [
                ("translation_m", "m", ["Right", "Up", "Forward"], 10.),
                ("rotation_degrees", "deg", ["Yaw", "Pitch", "Roll"], 90.),
            ] {
                for (i, name) in names.iter().enumerate() {
                    for (j, endpoint) in ["min", "max"].iter().enumerate() {
                        v.push(num(
                            &format!("/indoor_camera/handheld/{key}/{i}/{j}"),
                            &format!("{name} {endpoint} ({units})"),
                            -bound,
                            bound,
                            0.01,
                            "Signed endpoint increment. Minimum must not exceed maximum.",
                        ));
                    }
                }
            }
            v.push(num(
                "/indoor_camera/handheld/reverse_probability",
                "Reverse path probability",
                0.,
                1.,
                0.01,
                "Reverses positions and orientations together.",
            ));
        }
    }
    if indoor && state.view.page == Page::Scene {
        v.push(section(
            "ROOM MOTION",
            "Optional deterministic chair, door, cabinet, covering and ceiling-fan motion.",
        ));
        v.push(toggle(
            "@indoor_motion",
            "Animate furnishings",
            "Apply to generate clearance-checked motion; no models are loaded.",
        ));
        if !d["indoor_motion"].is_null() {
            for (key, label) in [
                ("chair_fraction", "Moving chair fraction"),
                ("door_fraction", "Moving door fraction"),
                ("cabinet_fraction", "Moving cabinet fraction"),
                ("covering_fraction", "Moving covering fraction"),
                ("fan_fraction", "Spinning fan fraction"),
                ("appliance_fraction", "Opening appliance fraction"),
                ("prop_fraction", "Moving portable item fraction"),
                ("cart_fraction", "Moving cart fraction"),
            ] {
                v.push(num(
                    &format!("/indoor_motion/{key}"),
                    label,
                    0.,
                    1.,
                    0.01,
                    "Selects eligible mechanisms. Constrained motions shorten or swivel; occupied or fully blocked objects remain static.",
                ));
            }
            v.push(num(
                "/indoor_motion/chair_travel_m",
                "Chair travel (m)",
                0.,
                1.5,
                0.05,
                "Maximum chair rolling/sliding travel; shorter paths and other headings are tried when blocked.",
            ));
            v.push(num(
                "/indoor_motion/chair_swivel_degrees",
                "Chair swivel (degrees)",
                0.,
                120.,
                5.,
                "Maximum chair turn or upper-seat swivel; allows rotation-only motion in tight spaces.",
            ));
            v.push(num(
                "/indoor_motion/max_mechanisms",
                "Mechanism limit",
                1.,
                128.,
                1.,
                "Bounds preparation and animation cost.",
            ));
            for (key, label, maximum, step) in [
                ("prop_travel_m", "Portable item travel (m)", 0.75, 0.01),
                (
                    "prop_rotation_degrees",
                    "Portable item rotation (degrees)",
                    180.,
                    5.,
                ),
                ("cart_travel_m", "Cart travel (m)", 2., 0.05),
            ] {
                v.push(num(&format!("/indoor_motion/{key}"), label, 0., maximum, step,
                    "Upper bound; clearance and support checks shorten blocked motion. Connected cables stay attached."));
            }
            for (index, label) in [(0, "Fan turns minimum"), (1, "Fan turns maximum")] {
                v.push(num(&format!("/indoor_motion/fan_turns/{index}"),label,
                    0.,120.,0.25,"Revolutions per shared timeline. Minimum must not exceed maximum; zero maximum disables spinning."));
            }
        }
    }
    if state.view.page == Page::People && !cfg!(feature = "human_motion") {
        v.push(Item::Note("Motion is unavailable in this build. Start native with --features human_motion to enable ARDY controls.".into()));
    }
    if state.view.page == Page::People && !d["human_motion"].is_null() {
        v.push(Item::Note(if cfg!(feature="human_motion") {"Models download once through burn_human loaders; configuration edits do not initialize inference."} else {"This build lacks human_motion. Enable that Cargo feature to apply generated motion."}.into()));
        for (name, key, min, max, step) in [
            ("Moving fraction", "fraction", 0., 1., 0.01),
            ("Navigate the room", "locomotion_fraction", 0., 1., 0.01),
            ("Walk / action / resume", "sequence_fraction", 0., 1., 0.01),
            ("Energetic travel", "energetic_fraction", 0., 1., 0.01),
            ("Maximum moving people", "max_actors", 1., 16., 1.),
            ("Clip frames (20 Hz)", "frames", 40., 640., 4.),
            ("Diffusion steps", "diffusion_steps", 1., 10., 1.),
            ("History frames", "history_frames", 0., 160., 4.),
            ("Inference batch size", "batch_size", 1., 8., 1.),
            ("Attempts per person", "max_attempts", 1., 3., 1.),
            ("Text guidance", "text_guidance", 0., 10., 0.1),
            ("Trajectory guidance", "trajectory_guidance", 0., 10., 0.1),
        ] {
            v.push(num(
                &format!("/human_motion/{key}"),
                name,
                min,
                max,
                step,
                "Applies on generation; collision rejection can retain a person as static.",
            ));
        }
        v.extend([
            toggle(
                "/human_motion/dense_trajectory",
                "Constrain complete navigation path",
                "Waypoints are planned after furniture placement.",
            ),
            toggle(
                "/human_motion/strict",
                "Fail capture on rejected motion",
                "Otherwise the requested actor stays static.",
            ),
            section(
                "PROMPT DISTRIBUTION",
                "Relative weights; at least one family must be nonzero.",
            ),
        ]);
        for (name, key) in [
            ("Travel / chair transitions", "locomotion"),
            ("Hand gestures", "gesture"),
            ("Exercise", "exercise"),
            ("Dance", "dance"),
            ("Floor actions", "floor"),
            ("Idle / looking", "idle"),
        ] {
            v.push(num(
                &format!("/human_motion/prompt_sampling/{key}"),
                name,
                0.,
                10.,
                0.1,
                "Continuous procedural prompt sampling weight.",
            ));
        }
        v.push(num(
            "/human_motion/prompt_sampling/style_fraction",
            "Posture & gaze variation",
            0.,
            1.,
            0.01,
            "Fraction of prompts with an additional style cue.",
        ));
        v.push(num(
            "/human_motion/prompt_sampling/max_sequence_actions",
            "Maximum action stops",
            1.,
            2.,
            1.,
            "Stops during compound walking sequences.",
        ));
        v.push(json_field(
            "/human_motion/trajectories",
            "Custom actor trajectories",
            "Array of actor_id, prompt and frame-indexed waypoints; [] uses sampled prompts.",
        ));
    }
    if state.view.page == Page::View && state.view.camera.is_none() && d["room_schematic"] != true {
        v.push(Item::Note(
            "Select Editor camera to initialize it and adjust its lens.".into(),
        ));
    }
    if !indoor {
        match state.view.page {
            Page::Scene => {
                v.retain(|item| !matches!(item, Item::Field(f) if f.path.starts_with("/indoor_")));
                if d["scene_type"] == "SemanticRoom" {
                    v.push(toggle(
                        "/cuboid_only",
                        "Architecture only",
                        "Omit interior objects in semantic rooms.",
                    ));
                }
                v.push(Item::Note(
                    "Edit this scene’s geometry program with the Geometry controls button below."
                        .into(),
                ));
            }
            Page::Cameras => {
                v = vec![section("CAPTURE CAMERAS", "Camera sampling follows this scene's geometry program."),
                    num("/num_cameras", "Capture views", 0.,16.,1., "0 uses only the editor camera."),
                    num("/max_camera_radius", "Maximum camera radius (m)", 0.,30.,0.1, "0 uses the scene default."),
                    Item::Note("Coordinated baselines and collision-checked room travel are available in Procedural indoor.".into())];
            }
            Page::Appearance => {
                v = vec![section("MATERIAL SAMPLING", "This scene samples the loaded MatSynth material and mesh catalog."),
                    toggle("/material_grid", "Browse material textures", "Shows the loaded MatSynth material catalog."),
                    num("/regenerate_scene_material_shuffle_period", "Scenes per material shuffle",0.,100.,1., "0 disables periodic shuffling."),
                    num("/regenerate_scene_mesh_shuffle_period", "Scenes per mesh shuffle",0.,100.,1., "0 disables periodic shuffling."),
                    Item::Note("Independent procedural material and lighting policies are available in Procedural indoor.".into())];
            }
            Page::People => {
                v = vec![section("ANIMATION", "Animation controls for this scene."), toggle("/animated", "Animate scene elements", "Applies on regeneration in scene types supporting animation."),
                    Item::Note("ARDY text and waypoint motion controls are available in Procedural indoor.".into())];
            }
            _ => {}
        }
    }
    if state.view.page == Page::View && d["room_schematic"] == true {
        v.retain(|item| match item {
            Item::Field(f) => ![
                "/render_mode",
                "@editor_fov",
                "/gizmos",
                "/draw_obb_gizmo",
                "/draw_pose_gizmos",
                "/gizmos_alpha",
                "@flow_interval",
                "@flow_scale",
            ]
            .contains(&f.path.as_str()),
            Item::Section(title, _) => *title != "OPTICAL FLOW",
            _ => true,
        });
        v.push(Item::Note("All capture cameras and human joints are shown. Furnishings use oriented bounds; ceilings and neighboring rooms are omitted. Predictions appear in dashed magenta.".into()));
    }
    v
}
