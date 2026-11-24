//! Live viewport, RGB presentation and diagnostic controls.
use super::*;
use crate::render::anti_alias::PreviewAa;
use bevy_args::ValueEnum;

fn aa_choices() -> Kind {
    Kind::Choice(
        PreviewAa::value_variants()
            .iter()
            .map(|&v| {
                let label = match v {
                    PreviewAa::Auto => "Automatic",
                    PreviewAa::Off => "Off",
                    PreviewAa::Fxaa => "FXAA · fast spatial",
                    PreviewAa::Smaa => "SMAA · high",
                    PreviewAa::SmaaUltra => "SMAA · ultra",
                    PreviewAa::Taa => "TAA · temporal",
                    PreviewAa::TaaFxaa => "TAA + FXAA · temporal & spatial",
                    PreviewAa::Msaa2 => "MSAA · 2×",
                    PreviewAa::Msaa4 => "MSAA · 4×",
                    PreviewAa::Msaa8 => "MSAA · 8×",
                };
                (label.into(), json!(v))
            })
            .collect(),
    )
}

pub(super) fn controls() -> Vec<Item> {
    [
        section(
            "view.preview",
            "Viewport & annotation",
            "Changes here update the current scene immediately.",
        ),
        choices(
            "@viewport",
            "Viewport",
            &[
                ("Editor camera", "editor"),
                ("Capture grid", "grid"),
                ("Room schematic", "schematic"),
            ],
            "Schematic shows metric primary-room footprints, live capture cameras, paths and human poses.",
        ),
        choices(
            "/render_mode",
            "Annotation",
            &[
                ("Color / PBR", "Color"),
                ("Depth", "Depth"),
                ("Surface normals", "Normal"),
                ("World position", "Position"),
                ("Semantic classes", "Semantic"),
                ("Optical flow", "OpticalFlow"),
                ("Motion vectors", "MotionVectors"),
                ("Co-visibility", "CoVisibility"),
            ],
            "Co-visibility uses capture views, excluding the editor camera.",
        ),
        toggle(
            "/fullscreen",
            "Fullscreen · F11",
            "F11 toggles the current viewport. In a browser, click the viewer to honor a fullscreen startup link.",
        ),
        section(
            "view.rgb",
            "RGB display & anti-aliasing",
            "Live settings for the displayed color image.",
        ),
        #[cfg(all(feature = "solari", not(target_arch = "wasm32")))]
        choices(
            "/preview_lighting",
            "RGB preview lighting",
            &[("Raster PBR", "raster"), ("Solari · experimental", "solari"), ("Solari + ReSTIR", "solari-restir")],
            "Native indoor preview of traced opaque surfaces. Glass, skinned shadows and local lights differ from calibrated capture. Requires ray queries.",
        ),
        field(
            "/preview_aa",
            "Preview anti-aliasing",
            "TAA + FXAA smooths strong edges after tone mapping. For fine patterns in the grid, search for Capture supersampling and apply 2×. Temporal modes use spatial filtering on fisheye and inset views. MSAA disables native screen-space ambient occlusion.",
            aa_choices(),
        ),
        choices(
            "/tonemapping",
            "RGB tone mapping",
            &[
                ("Automatic", "auto"),
                ("Tony McMapface", "tony-mcmapface"),
                ("AgX", "agx"),
                ("ACES fitted", "aces-fitted"),
                ("Blender Filmic", "blender-filmic"),
                ("Khronos PBR Neutral", "khronos-pbr-neutral"),
                ("Reinhard", "reinhard"),
                ("Reinhard luminance", "reinhard-luminance"),
                ("Somewhat Boring Display Transform", "somewhat-boring-display-transform"),
                ("Linear · keep grading", "linear"),
                ("None · bypass grading", "none"),
            ],
            "Automatic uses Tony McMapface for SDR and linear grading for HDR.",
        ),
        toggle(
            "/hdr",
            "HDR RGB output",
            "Uses display headroom when available. Floating-point captures can retain highlights above reference white.",
        ),
        section(
            "view.geometry",
            "Depth & glass",
            "Shared geometric-hit rules for capture annotations.",
        ),
        choices(
            "/depth_format",
            "Depth visualization",
            &[
                ("Grayscale", "Normalized"),
                ("Turbo · warm near / cool far", "Turbo"),
                ("Original color ramp", "Colorized"),
                ("Linear distance", "Linear"),
            ],
            "Turbo uses log distance for indoor detail, with a fixed scale per camera. Linear and normalized numeric depth encodings are unchanged.",
        ),
        toggle(
            "/z_depth",
            "Use camera Z depth",
            "Otherwise measures ray distance.",
        ),
        choices(
            "/annotation_glass",
            "Glass in annotations",
            &[
                ("Glass surface", "surface"),
                ("Geometry behind glass", "through"),
            ],
            "Depth, normals, position, semantic, flow and co-visibility use the same unrefracted geometric hits. RGB is unchanged.",
        ),
        section(
            "view.overlays",
            "Scene overlays",
            "Diagnostic drawings in the editor view.",
        ),
        toggle(
            "/gizmos",
            "Capture cameras & paths",
            "Compact calibrated camera outlines and paths; one color per camera ID.",
        ),
        toggle(
            "/draw_obb_gizmo",
            "Object bounding boxes",
            "Semantic object envelopes.",
        ),
        toggle(
            "/draw_pose_gizmos",
            "Human skeletons",
            "Visible through clothing, occluded by room geometry.",
        ),
        num(
            "/gizmos_alpha",
            "Overlay opacity",
            0.,
            1.,
            0.01,
            "Opacity for camera gizmos.",
        ),
        num(
            "/camera_gizmo_scale",
            "Camera glyph size · editor",
            0.1,
            4.,
            0.05,
            "4× is the default. Scales capture and prediction glyphs immediately; does not change camera optics or paths.",
        ),
        section(
            "view.capture_overlays",
            "Capture-camera RGB overlays",
            "Draw in each calibrated color image, including the grid and exported RGB.",
        ),
        toggle(
            "/rgb_gizmos/object_obbs",
            "Object bounding boxes",
            "Oriented bounds of labeled objects.",
        ),
        toggle(
            "/rgb_gizmos/human_poses",
            "Human skeletons",
            "Joints and bones show through clothing; room geometry occludes them.",
        ),
        toggle(
            "/rgb_gizmos/camera_frustums",
            "Other cameras’ frustums",
            "Outlines follow each camera’s calibrated lens and ID color.",
        ),
        toggle(
            "/rgb_gizmos/camera_trajectories",
            "Camera trajectories",
            "Show each moving camera’s path through the room.",
        ),
        num(
            "/rgb_gizmos/camera_scale",
            "Capture frustum size",
            0.1,
            4.,
            0.05,
            "Metric glyph multiplier. Camera optics and path length stay fixed.",
        ),
        section(
            "view.timeline",
            "Timeline & playback",
            "Cameras, people, and moving objects share the same trajectory position.",
        ),
        num(
            "@progress",
            "Trajectory position",
            0.,
            1.,
            0.001,
            "Drag to pause at a position. Press Space to resume the selected curve.",
        ),
        choices(
            "/playback_mode",
            "Playback curve",
            &[
                ("Paused", "Still"),
                ("Once", "Once"),
                ("Loop (restart)", "Loop"),
                ("Ping-pong", "PingPong"),
                ("Sine", "Sin"),
                ("Ease in", "EaseIn"),
                ("Ease out", "EaseOut"),
                ("Ease in / out", "EaseInOut"),
                ("Cubic in", "EaseInCubic"),
                ("Cubic out", "EaseOutCubic"),
                ("Cubic in / out", "EaseInOutCubic"),
                ("Smoothstep", "Smoothstep"),
                ("Smootherstep", "Smootherstep"),
            ],
            "Once stops at the end. Loop restarts. The other moving curves retrace the path. Smootherstep gently comes to rest at both ends.",
        ),
        num(
            "/playback_speed",
            "Playback speed",
            0.,
            2.,
            0.01,
            "At 0.2, Sine takes 5 seconds for a round trip. Other moving curves take 5 seconds per leg.",
        ),
        num(
            "/yaw_speed",
            "World rotation (rad/s)",
            -1.,
            1.,
            0.01,
            "Rotates the entire scene; separate from camera travel.",
        ),
        toggle(
            "/keybinds",
            "Enable shortcuts",
            "R: next room; Space: play/pause. Text fields capture keyboard input.",
        ),
        section(
            "view.navigation",
            "Editor navigation",
            "These affect the editor camera only.",
        ),
        num(
            "@editor_fov",
            "Editor field of view (degrees)",
            15.,
            150.,
            1.,
            "Editor lens only; capture intrinsics remain the seeded dataset cameras.",
        ),
        num(
            "/orbit_smoothness",
            "Orbit smoothing",
            0.,
            1.,
            0.01,
            "0 is immediate; higher values ease motion.",
        ),
        num(
            "/pan_smoothness",
            "Pan smoothing",
            0.,
            1.,
            0.01,
            "Mouse navigation is disabled over controls.",
        ),
        num(
            "/zoom_smoothness",
            "Zoom smoothing",
            0.,
            1.,
            0.01,
            "Editor intrinsics persist across regeneration.",
        ),
        section(
            "view.flow",
            "Optical-flow preview",
            "Viewer visualization scale; export uses exact capture timestep displacement.",
        ),
        num(
            "@flow_interval",
            "Preview interval (seconds)",
            0.005,
            0.5,
            0.005,
            "Velocity multiplied by this fixed interval, independent of frame rate.",
        ),
        num(
            "@flow_scale",
            "Full color scale (pixels)",
            1.,
            256.,
            1.,
            "A visualization scale, not a change to motion annotations.",
        ),
    ].into()
}
