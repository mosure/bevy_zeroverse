//! A single control catalog feeds Feathers widgets and coverage tests.
use super::model::{EditorState, Page};
use serde_json::{json, Value};
mod cameras;
mod ceiling;
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
#[derive(Clone, Copy, Debug)]
pub struct Section {
    /// Stable navigation key, independent of the visible title.
    pub id: &'static str,
    pub title: &'static str,
    pub help: &'static str,
}
pub enum Item {
    Section(Section),
    Field(Field),
    Note(String),
    Distribution(super::distribution_preview::DistributionKind),
    CapacityEvidence,
    Action(&'static str, super::Action),
    CoVisibilityLegend,
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
fn section(id: &'static str, title: &'static str, help: &'static str) -> Item {
    Item::Section(Section { id, title, help })
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
        append_to_section(
            &mut v,
            "view.preview",
            Item::Note("Select Editor camera to initialize it and adjust its lens.".into()),
        );
    }
    if !indoor {
        match state.view.page {
            Page::Scene => {
                v.retain(|item| !matches!(item, Item::Field(f) if f.path.starts_with("/indoor_")));
            }
            Page::Cameras => {
                v = vec![section("cameras.rig", "Capture cameras", "Camera sampling follows this scene's geometry program."),
                    num("/num_cameras", "Capture views", 0.,16.,1., "0 uses only the editor camera."),
                    num("/max_camera_radius", "Maximum camera radius (m)", 0.,30.,0.1, "0 uses the scene default."),
                    Item::Note("Coordinated baselines and collision-checked room travel are available in Procedural indoor.".into())];
            }
            Page::Appearance => {
                v = vec![section("appearance.surfaces", "Material sampling", "This scene samples the loaded MatSynth material and mesh catalog."),
                    toggle("/material_grid", "Browse material textures", "Shows the loaded MatSynth material catalog."),
                    num("/regenerate_scene_material_shuffle_period", "Scenes per material shuffle",0.,100.,1., "0 disables periodic shuffling."),
                    num("/regenerate_scene_mesh_shuffle_period", "Scenes per mesh shuffle",0.,100.,1., "0 disables periodic shuffling."),
                    Item::Note("Independent procedural material and lighting policies are available in Procedural indoor.".into())];
            }
            Page::People => {
                v = vec![section("people.population", "Scene animation", "Animation controls for this scene."), toggle("/animated", "Animate scene elements", "Applies on regeneration in scene types supporting animation."),
                    Item::Note("ARDY text and waypoint motion controls are available in Procedural indoor.".into())];
            }
            _ => {}
        }
    }
    if state.view.page == Page::View && d["room_schematic"] == true {
        remove_sections(
            &mut v,
            &[
                "view.flow",
                "view.overlays",
                "view.geometry",
                "view.navigation",
            ],
        );
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
            _ => true,
        });
        append_to_section(&mut v, "view.preview", Item::Note("All capture cameras and human joints are shown. Furnishings use oriented bounds; ceilings and neighboring rooms are omitted. Camera predictions match the C-index color and use dashed lines; other predictions use magenta.".into()));
        append_to_section(&mut v, "view.preview", num("/schematic_camera_gizmo_scale", "Camera glyph size · schematic", 0.1, 4., 0.05, "1× is the default. Scales capture and prediction glyphs immediately; room dimensions, paths and calibration stay unchanged."));
    }
    if indoor && state.view.page == Page::Cameras && !d["indoor_camera"]["ceiling_mount"].is_null()
    {
        ceiling::append(&mut v);
    }

    if matches!(state.view.page, Page::Scene | Page::Advanced)
        && matches!(
            d["scene_type"].as_str(),
            Some("Object" | "Room" | "SemanticRoom")
        )
    {
        v.extend([
            section(
                "scene.geometry",
                "Scene geometry",
                "Geometry controls for the selected scene program.",
            ),
            Item::Action("Geometry controls", super::Action::Debug),
        ]);
        if state.view.page == Page::Scene && d["scene_type"] == "SemanticRoom" {
            v.push(toggle(
                "/cuboid_only",
                "Architecture only",
                "Omit interior objects in semantic rooms.",
            ));
        }
    }
    if state.view.page == Page::View && d["room_schematic"] != true {
        v.extend([
            section(
                "view.co-visibility",
                "Co-visibility legend",
                "Each bit identifies another capture camera; colors are consistent across views.",
            ),
            Item::CoVisibilityLegend,
        ]);
    }
    v
}

fn append_to_section(items: &mut Vec<Item>, id: &str, item: Item) {
    if let Some(start) = items
        .iter()
        .position(|i| matches!(i, Item::Section(s) if s.id == id))
    {
        let end = items
            .iter()
            .enumerate()
            .skip(start + 1)
            .find(|(_, i)| matches!(i, Item::Section(_)))
            .map_or(items.len(), |(i, _)| i);
        items.insert(end, item);
    }
}

fn remove_sections(items: &mut Vec<Item>, ids: &[&str]) {
    let mut remove = false;
    items.retain(|item| {
        if let Item::Section(section) = item {
            remove = ids.contains(&section.id);
        }
        !remove
    });
}
