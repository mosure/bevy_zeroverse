//! A single control catalog feeds Feathers widgets and coverage tests.
use super::model::{EditorState, Page};
use serde_json::{json, Value};
mod cameras;
mod furnishing;
mod motion;
mod pages;

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
    Distribution(super::distribution_preview::DistributionKind),
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
    let mut v = pages::base(state.view.page);
    cameras::append(state, &mut v);
    motion::append(state, &mut v);
    if indoor && state.view.page == Page::Scene {
        furnishing::append(&mut v);
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
    if indoor && state.view.page == Page::Cameras && !d["indoor_camera"]["ceiling_mount"].is_null()
    {
        // Lens controls reuse the same widgets, bound to the mounting prior.
        // Free-camera travel controls cannot affect a fixed physical mount.
        v.retain(|item| match item {
            Item::Field(f) => {
                ![
                    "@camera_motion_percent",
                    "@path_preset",
                    "@handheld",
                    "/indoor_camera/primary_room",
                    "/indoor_camera/path_length_min",
                    "/indoor_camera/path_length_max",
                    "/indoor_camera/long_path_fraction",
                    "/indoor_camera/multiview/trajectory_variation",
                ]
                .contains(&f.path.as_str())
                    && !f.path.starts_with("/indoor_camera/handheld/")
            }
            Item::Section(title, _) => *title != "CAMERA MOTION",
            _ => true,
        });
        for item in &mut v {
            if let Item::Field(field) = item {
                if field.path.starts_with("/indoor_camera/intrinsics/")
                    || field.path.starts_with("/indoor_camera/lenses/")
                {
                    field.path =
                        field
                            .path
                            .replacen("/indoor_camera/", "/indoor_camera/ceiling_mount/", 1);
                }
                if field.path == "@fisheye_probability" {
                    field.help = "Ceiling rigs default to 100% fisheye. The remaining cameras use the wide perspective distribution above.".into();
                }
            }
        }
        v.extend([
            section("CEILING MOUNT", "Fixed poses in the primary room. The optical axis follows the inward ceiling normal, including slopes and soffits. Object and human animation remain available."),
            num("/indoor_camera/ceiling_mount/standoff_m/0", "Minimum ceiling standoff (m)", 0.08, 1., 0.01, "Optical-center distance from the supporting ceiling. Must exceed twice the body radius plus 5 mm."),
            num("/indoor_camera/ceiling_mount/standoff_m/1", "Maximum ceiling standoff (m)", 0.08, 1., 0.01, "Sampled per camera before placement retries."),
            num("/indoor_camera/ceiling_mount/body_radius_m", "Camera body radius (m)", 0.02, 0.20, 0.005, "Clearance reservation; no camera body is rendered into its own capture."),
            num("/indoor_camera/ceiling_mount/wall_clearance_m", "Wall clearance (m)", 0.20, 2., 0.01, "Mounts avoid walls, light fixtures, suspension, ducts and furniture."),
        ]);
    }
    v
}
