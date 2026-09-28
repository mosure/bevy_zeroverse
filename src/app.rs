use bevy::{
    app::AppExit,
    prelude::*,
    render::{
        // render_resource::{
        //     Extent3d,
        //     TextureDescriptor,
        //     TextureDimension,
        //     TextureUsages,
        // },
        // renderer::RenderDevice,
        settings::{RenderCreation, WgpuFeatures, WgpuSettings},
        // texture::{
        //     ImageLoaderSettings,
        //     ImageSampler,
        // },
        RenderPlugin,
    },
    time::Stopwatch,
    winit::WinitPlugin,
};
#[cfg(not(target_arch = "wasm32"))]
use bevy_args::parse_args;
use bevy_args::{Deserialize, Parser, Serialize, ValueEnum};

#[cfg(feature = "viewer")]
use bevy_egui::EguiPlugin;
#[cfg(feature = "viewer")]
mod inspector;
mod settings;
#[cfg(feature = "viewer")]
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq, ValueEnum, Reflect)]
#[cfg_attr(feature = "python", pyclass(eq, eq_int))]
pub enum OvoxelMode {
    Disabled,
    CpuAsync,
    GpuCompute,
}

use crate::{
    camera::{DefaultZeroverseCamera, Playback, PlaybackMode},
    io,
    material::ShuffleMaterialsEvent,
    mesh::ShuffleMeshesEvent,
    // plucker::ZeroversePluckerSettings,
    ovoxel::GPU_DEFAULT_MAX_OUTPUT_VOXELS,
    render::{depth::DepthFormat, RenderMode},
    scene::{
        procedural_indoor::layout::IndoorLayout, semantic_room::ZeroverseSemanticRoomSettings,
        RegenerateSceneEvent, ZeroverseSceneRoot, ZeroverseSceneSettings, ZeroverseSceneType,
    },
    BevyZeroversePlugin,
};

#[cfg(feature = "viewer")]
use crate::{
    camera::{EditorCameraMarker, ZeroverseCamera},
    material::{MaterialsLoadedEvent, ZeroverseMaterials},
    primitive::ScaleSampler,
    scene::{room::ZeroverseRoomSettings, SceneLoadedEvent},
};
#[cfg(feature = "viewer")]
use bevy::camera::RenderTarget;

fn deserialize_human_motion<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<String>, D::Error> {
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    value
        .map(|value| {
            let json = match value {
                serde_json::Value::String(json) => json,
                value if value.is_object() => value.to_string(),
                _ => {
                    return Err(serde::de::Error::custom(
                        "human_motion must be a JSON object or JSON string",
                    ))
                }
            };
            crate::human_motion::HumanMotionConfig::parse(&json)
                .map_err(serde::de::Error::custom)?;
            Ok(json)
        })
        .transpose()
}

fn deserialize_indoor_camera<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<String>, D::Error> {
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    value
        .map(|value| {
            let json = if let serde_json::Value::String(s) = value {
                s
            } else if value.is_object() {
                value.to_string()
            } else {
                return Err(serde::de::Error::custom(
                    "indoor_camera must be a JSON object or JSON string",
                ));
            };
            crate::scene::procedural_indoor::cameras::CameraSettings::parse(&json)
                .map_err(serde::de::Error::custom)?;
            Ok(json)
        })
        .transpose()
}

fn default_indoor_human_density() -> f32 {
    0.25
}

fn default_indoor_gi_rays() -> u32 {
    256
}

fn deserialize_indoor_gi_rays<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<u32, D::Error> {
    let rays = <u32 as serde::Deserialize>::deserialize(deserializer)?;
    if (64..=16384).contains(&rays) {
        Ok(rays)
    } else {
        Err(serde::de::Error::custom(
            "indoor_gi_rays must be between 64 and 16384",
        ))
    }
}

fn default_indoor_density() -> f32 {
    0.65
}

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn indoor_gi_budget_has_bounded_cli_and_json_configuration() {
        assert_eq!(BevyZeroverseConfig::default().indoor_gi_rays, 256);
        let mut json = serde_json::to_value(BevyZeroverseConfig::default()).unwrap();
        json.as_object_mut().unwrap().remove("indoor_gi_rays");
        assert_eq!(
            serde_json::from_value::<BevyZeroverseConfig>(json.clone())
                .unwrap()
                .indoor_gi_rays,
            256
        );
        for rays in [64, 256, 1024, 16384] {
            json["indoor_gi_rays"] = serde_json::json!(rays);
            assert_eq!(
                serde_json::from_value::<BevyZeroverseConfig>(json.clone())
                    .unwrap()
                    .indoor_gi_rays,
                rays
            );
            assert_eq!(
                BevyZeroverseConfig::try_parse_from([
                    "bevy_zeroverse",
                    "--indoor-gi-rays",
                    &rays.to_string(),
                ])
                .unwrap()
                .indoor_gi_rays,
                rays
            );
        }
        for rays in [0, 63, 16385, u32::MAX] {
            json["indoor_gi_rays"] = serde_json::json!(rays);
            assert!(serde_json::from_value::<BevyZeroverseConfig>(json.clone()).is_err());
            assert!(BevyZeroverseConfig::try_parse_from([
                "bevy_zeroverse",
                "--indoor-gi-rays",
                &rays.to_string(),
            ])
            .is_err());
        }
    }
}

// TODO: add meta-derive macro to populate get/set methods
#[cfg(feature = "python")]
#[derive(Clone, Debug, Resource, Serialize, Deserialize, Parser, Reflect)]
#[pyclass]
#[command(about = "bevy_zeroverse viewer", version, long_about = None)]
#[reflect(Resource)]
pub struct BevyZeroverseConfig {
    /// enable the bevy inspector
    #[pyo3(get, set)]
    #[arg(long, action = clap::ArgAction::Set, default_value = "true")]
    pub editor: bool,

    /// draw capture-camera frusta and trajectories in the editor
    #[pyo3(get, set)]
    #[arg(long, action = clap::ArgAction::Set, default_value = "true")]
    pub gizmos: bool,

    /// alpha value for gizmos
    #[pyo3(get, set)]
    #[arg(long, default_value = "1.0")]
    pub gizmos_alpha: f32,

    /// draw oriented bounding box gizmos
    #[pyo3(get, set)]
    #[arg(long, action = clap::ArgAction::Set, default_value = "false")]
    pub draw_obb_gizmo: bool,

    /// draw burn_human pose gizmos
    #[pyo3(get, set)]
    #[arg(long, action = clap::ArgAction::Set, default_value = "false")]
    pub draw_pose_gizmos: bool,

    /// no window will be shown
    #[pyo3(get, set)]
    #[arg(long, default_value = "false")]
    pub headless: bool,

    /// whether or not zeroverse cameras receive image copiers
    #[pyo3(get, set)]
    #[arg(long, default_value = "false")]
    pub image_copiers: bool,

    /// view available material basecolor textures in a grid
    #[pyo3(get, set)]
    #[arg(long, default_value = "false")]
    pub material_grid: bool,

    /// view plücker embeddings
    #[pyo3(get, set)]
    #[arg(long, default_value = "false")]
    pub plucker_visualization: bool,

    /// enable closing the window with the escape key (doesn't work in web)
    #[pyo3(get, set)]
    #[arg(long, action = clap::ArgAction::Set, default_value = "true")]
    pub press_esc_close: bool,

    #[pyo3(get, set)]
    #[arg(long, default_value = "1920.0")]
    pub width: f32,

    #[pyo3(get, set)]
    #[arg(long, default_value = "1080.0")]
    pub height: f32,

    #[pyo3(get, set)]
    #[arg(long, default_value = "0")]
    pub num_cameras: usize,

    /// display a grid of Zeroverse cameras
    #[pyo3(get, set)]
    #[arg(long, default_value = "false")]
    pub camera_grid: bool,

    /// window title
    #[pyo3(get, set)]
    #[arg(long, default_value = "bevy_zeroverse")]
    pub name: String,

    /// move to the next scene after `regenerate_ms` milliseconds
    #[pyo3(get, set)]
    #[arg(long, default_value = "0")]
    pub regenerate_ms: u32,

    /// afer this many scene regenerations, shuffle the materials
    #[pyo3(get, set)]
    #[arg(long, default_value = "0")]
    pub regenerate_scene_material_shuffle_period: u32,

    /// afer this many scene regenerations, shuffle the meshes
    #[pyo3(get, set)]
    #[arg(long, default_value = "0")]
    pub regenerate_scene_mesh_shuffle_period: u32,

    /// automatically rotate the root scene object in the y axis
    #[pyo3(get, set)]
    #[arg(long, default_value = "0.0")]
    pub yaw_speed: f32,

    #[pyo3(get, set)]
    #[arg(long, value_enum, default_value_t = RenderMode::Color)]
    pub render_mode: RenderMode,

    #[pyo3(get, set)]
    pub render_modes: Vec<RenderMode>,

    #[pyo3(get, set)]
    #[arg(long, value_enum, default_value_t = ZeroverseSceneType::Object)]
    pub scene_type: ZeroverseSceneType,

    /// Base seed for procedural_indoor; successive scenes use seed + scene index.
    #[pyo3(get, set)]
    #[arg(long)]
    #[serde(default)]
    pub indoor_seed: Option<u64>,

    /// Indoor grammar family; mixed samples all four families.
    #[pyo3(get, set)]
    #[arg(long, value_enum, default_value_t = IndoorLayout::Mixed)]
    #[serde(default)]
    pub indoor_layout: IndoorLayout,

    /// Probability of secondary props and vegetation (0 through 1).
    #[pyo3(get, set)]
    #[arg(long, default_value = "0.65")]
    #[serde(default = "default_indoor_density")]
    pub indoor_density: f32,

    /// Fraction of chairs occupied, with sparse standing adults (0 disables people).
    #[pyo3(get, set)]
    #[arg(long, default_value = "0.25")]
    #[serde(default = "default_indoor_human_density")]
    pub indoor_human_density: f32,

    /// Capture camera JSON: primary_room, path_length_min/max, long_path_fraction, multiview {min_overlap, min_baseline, max_baseline}.
    #[pyo3(get, set)]
    #[arg(long)]
    #[serde(default, deserialize_with = "deserialize_indoor_camera")]
    pub indoor_camera: Option<String>,

    /// Opt-in motion policy as JSON; requires the human_motion Cargo feature.
    #[pyo3(get, set)]
    #[arg(long)]
    #[serde(default, deserialize_with = "deserialize_human_motion")]
    pub human_motion: Option<String>,

    /// Auto enables platform-supported effects; portable reduces GPU requirements.
    #[pyo3(get, set)]
    #[arg(long, value_enum, default_value_t = crate::scene::procedural_indoor::IndoorQuality::Auto)]
    #[serde(default)]
    pub indoor_quality: crate::scene::procedural_indoor::IndoorQuality,

    /// Native diffuse-GI rays per probe; 1024 reduces noise at higher generation cost.
    #[pyo3(get, set)]
    #[arg(long, default_value = "256", value_parser = clap::value_parser!(u32).range(64..=16384))]
    #[serde(
        default = "default_indoor_gi_rays",
        deserialize_with = "deserialize_indoor_gi_rays"
    )]
    pub indoor_gi_rays: u32,

    #[pyo3(get, set)]
    #[arg(long, default_value = "false")]
    pub rotation_augmentation: bool,

    #[pyo3(get, set)]
    #[arg(long, default_value = "0.0")]
    pub max_camera_radius: f32,

    #[pyo3(get, set)]
    #[arg(long, value_enum, default_value_t = PlaybackMode::Still)]
    pub playback_mode: PlaybackMode,

    #[pyo3(get, set)]
    #[arg(long, default_value = "0.2")]
    pub playback_speed: f32,

    #[pyo3(get, set)]
    #[arg(long, default_value = "0.05")]
    pub playback_step: f32,

    #[pyo3(get, set)]
    #[arg(long, default_value = "5")]
    pub playback_steps: u32,

    /// enable animation for animated scene elements (e.g. base room humans)
    #[pyo3(get, set)]
    #[arg(long, default_value = "false")]
    pub animated: bool,

    #[pyo3(get, set)]
    #[arg(long, action = clap::ArgAction::Set, default_value = "true")]
    pub keybinds: bool,

    #[pyo3(get, set)]
    #[arg(long, action = clap::ArgAction::Set, default_value = "true")]
    pub initialize_scene: bool,

    #[pyo3(get, set)]
    #[arg(long, default_value = "0.8")]
    pub orbit_smoothness: f32,

    #[pyo3(get, set)]
    #[arg(long, default_value = "0.6")]
    pub pan_smoothness: f32,

    #[pyo3(get, set)]
    #[arg(long, default_value = "0.8")]
    pub zoom_smoothness: f32,

    #[pyo3(get, set)]
    #[arg(long, value_enum, default_value_t = DepthFormat::Normalized)]
    pub depth_format: DepthFormat,

    /// use z-depth instead of ray depth
    #[pyo3(get, set)]
    #[arg(long, action = clap::ArgAction::Set, default_value = "true")]
    pub z_depth: bool,

    /// semantic_room no interior objects
    #[pyo3(get, set)]
    #[arg(long, default_value = "false")]
    pub cuboid_only: bool,

    /// override O-Voxel export resolution (0 = default)
    #[pyo3(get, set)]
    #[arg(long, default_value = "128")]
    pub ovoxel_resolution: u32,

    /// Maximum number of voxels retained from the GPU path (upper bound on buffer size)
    #[pyo3(get, set)]
    #[arg(long, default_value_t = GPU_DEFAULT_MAX_OUTPUT_VOXELS)]
    pub ovoxel_max_output_voxels: u32,

    /// O-Voxel generation mode
    #[pyo3(get, set)]
    #[arg(long, value_enum, default_value_t = OvoxelMode::Disabled)]
    pub ovoxel_mode: OvoxelMode,
}

#[cfg(feature = "python")]
#[pymethods]
impl BevyZeroverseConfig {
    #[new]
    pub fn new() -> Self {
        Default::default()
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{self:?}"))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{self:?}"))
    }
}

#[cfg(not(feature = "python"))]
#[derive(Clone, Debug, Resource, Serialize, Deserialize, Parser, Reflect)]
#[command(about = "bevy_zeroverse viewer", version, long_about = None)]
#[reflect(Resource)]
pub struct BevyZeroverseConfig {
    /// enable the bevy inspector
    #[arg(long, action = clap::ArgAction::Set, default_value = "true")]
    pub editor: bool,

    /// draw capture-camera frusta and trajectories in the editor
    #[arg(long, action = clap::ArgAction::Set, default_value = "true")]
    pub gizmos: bool,

    /// alpha value for gizmos
    #[arg(long, default_value = "1.0")]
    pub gizmos_alpha: f32,

    /// draw oriented bounding box gizmos
    #[arg(long, action = clap::ArgAction::Set, default_value = "false")]
    pub draw_obb_gizmo: bool,

    /// draw burn_human pose gizmos
    #[arg(long, action = clap::ArgAction::Set, default_value = "false")]
    pub draw_pose_gizmos: bool,

    /// no window will be shown
    #[arg(long, default_value = "false")]
    pub headless: bool,

    /// whether or not zeroverse cameras receive image copiers
    #[arg(long, default_value = "false")]
    pub image_copiers: bool,

    /// view available material basecolor textures in a grid
    #[arg(long, default_value = "false")]
    pub material_grid: bool,

    /// view plücker embeddings
    #[arg(long, default_value = "false")]
    pub plucker_visualization: bool,

    /// enable closing the window with the escape key (doesn't work in web)
    #[arg(long, action = clap::ArgAction::Set, default_value = "true")]
    pub press_esc_close: bool,

    #[arg(long, default_value = "1920.0")]
    pub width: f32,

    #[arg(long, default_value = "1080.0")]
    pub height: f32,

    #[arg(long, default_value = "0")]
    pub num_cameras: usize,

    /// display a grid of Zeroverse cameras
    #[arg(long, default_value = "false")]
    pub camera_grid: bool,

    /// window title
    #[arg(long, default_value = "bevy_zeroverse")]
    pub name: String,

    /// move to the next scene after `regenerate_ms` milliseconds
    #[arg(long, default_value = "0")]
    pub regenerate_ms: u32,

    /// afer this many scene regenerations, shuffle the materials
    #[arg(long, default_value = "0")]
    pub regenerate_scene_material_shuffle_period: u32,

    /// afer this many scene regenerations, shuffle the materials
    #[arg(long, default_value = "0")]
    pub regenerate_scene_mesh_shuffle_period: u32,

    /// automatically rotate the root scene object in the y axis
    #[arg(long, default_value = "0.0")]
    pub yaw_speed: f32,

    #[arg(long, value_enum, default_value_t = RenderMode::Color)]
    pub render_mode: RenderMode,

    pub render_modes: Vec<RenderMode>,

    #[arg(long, value_enum, default_value_t = ZeroverseSceneType::Object)]
    pub scene_type: ZeroverseSceneType,

    /// Base seed for procedural_indoor; successive scenes use seed + scene index.
    #[arg(long)]
    #[serde(default)]
    pub indoor_seed: Option<u64>,

    /// Indoor grammar family; mixed samples all four families.
    #[arg(long, value_enum, default_value_t = IndoorLayout::Mixed)]
    #[serde(default)]
    pub indoor_layout: IndoorLayout,

    /// Probability of secondary props and vegetation (0 through 1).
    #[arg(long, default_value = "0.65")]
    #[serde(default = "default_indoor_density")]
    pub indoor_density: f32,

    /// Fraction of chairs occupied, with sparse standing adults (0 disables people).
    #[arg(long, default_value = "0.25")]
    #[serde(default = "default_indoor_human_density")]
    pub indoor_human_density: f32,

    /// Capture camera JSON: primary_room, path_length_min/max, long_path_fraction, multiview {min_overlap, min_baseline, max_baseline}.
    #[arg(long)]
    #[serde(default, deserialize_with = "deserialize_indoor_camera")]
    pub indoor_camera: Option<String>,

    /// Opt-in motion policy as JSON; requires the human_motion Cargo feature.
    #[arg(long)]
    #[serde(default, deserialize_with = "deserialize_human_motion")]
    pub human_motion: Option<String>,

    /// Auto enables platform-supported effects; portable reduces GPU requirements.
    #[arg(long, value_enum, default_value_t = crate::scene::procedural_indoor::IndoorQuality::Auto)]
    #[serde(default)]
    pub indoor_quality: crate::scene::procedural_indoor::IndoorQuality,

    /// Native diffuse-GI rays per probe; 1024 reduces noise at higher generation cost.
    #[arg(long, default_value = "256", value_parser = clap::value_parser!(u32).range(64..=16384))]
    #[serde(
        default = "default_indoor_gi_rays",
        deserialize_with = "deserialize_indoor_gi_rays"
    )]
    pub indoor_gi_rays: u32,

    #[arg(long, default_value = "false")]
    pub rotation_augmentation: bool,

    #[arg(long, default_value = "0.0")]
    pub max_camera_radius: f32,

    #[arg(long, value_enum, default_value_t = PlaybackMode::Sin)]
    pub playback_mode: PlaybackMode,

    #[arg(long, default_value = "0.2")]
    pub playback_speed: f32,

    #[arg(long, default_value = "0.05")]
    pub playback_step: f32,

    #[arg(long, default_value = "5")]
    pub playback_steps: u32,

    /// enable animation for animated scene elements (e.g. base room humans)
    #[arg(long, default_value = "false")]
    pub animated: bool,

    #[arg(long, action = clap::ArgAction::Set, default_value = "true")]
    pub keybinds: bool,

    #[arg(long, action = clap::ArgAction::Set, default_value = "true")]
    pub initialize_scene: bool,

    #[arg(long, default_value = "0.8")]
    pub orbit_smoothness: f32,

    #[arg(long, default_value = "0.6")]
    pub pan_smoothness: f32,

    #[arg(long, default_value = "0.8")]
    pub zoom_smoothness: f32,

    #[arg(long, value_enum, default_value_t = DepthFormat::Normalized)]
    pub depth_format: DepthFormat,

    /// use z-depth instead of ray depth
    #[arg(long, action = clap::ArgAction::Set, default_value = "true")]
    pub z_depth: bool,

    /// semantic_room no interior objects
    #[arg(long, default_value = "false")]
    pub cuboid_only: bool,

    /// override O-Voxel export resolution (0 = default)
    #[arg(long, default_value = "128")]
    pub ovoxel_resolution: u32,

    /// Maximum number of voxels retained from the GPU path (upper bound on buffer size)
    #[arg(long, default_value_t = GPU_DEFAULT_MAX_OUTPUT_VOXELS)]
    pub ovoxel_max_output_voxels: u32,

    /// O-Voxel generation mode
    #[arg(long, value_enum, default_value_t = OvoxelMode::Disabled)]
    pub ovoxel_mode: OvoxelMode,
}

impl Default for BevyZeroverseConfig {
    fn default() -> BevyZeroverseConfig {
        BevyZeroverseConfig {
            editor: true,
            gizmos: true,
            gizmos_alpha: 1.0,
            draw_obb_gizmo: false,
            draw_pose_gizmos: false,
            headless: false,
            image_copiers: false,
            material_grid: false,
            plucker_visualization: false,
            press_esc_close: true,
            width: 1920.0,
            height: 1080.0,
            num_cameras: 0,
            camera_grid: false,
            name: "bevy_zeroverse".to_string(),
            regenerate_ms: 0,
            regenerate_scene_material_shuffle_period: 0,
            regenerate_scene_mesh_shuffle_period: 0,
            yaw_speed: 0.0,
            render_mode: Default::default(),
            render_modes: vec![],
            scene_type: Default::default(),
            indoor_seed: None,
            indoor_layout: IndoorLayout::Mixed,
            indoor_density: 0.65,
            indoor_human_density: 0.25,
            indoor_camera: None,
            human_motion: None,
            indoor_quality: crate::scene::procedural_indoor::IndoorQuality::Auto,
            indoor_gi_rays: 256,
            rotation_augmentation: false,
            max_camera_radius: 0.0,
            playback_mode: PlaybackMode::Sin,
            playback_speed: 0.2,
            playback_step: 0.05,
            playback_steps: 5,
            animated: false,
            keybinds: true,
            initialize_scene: true,
            orbit_smoothness: 0.8,
            pan_smoothness: 0.6,
            zoom_smoothness: 0.8,
            depth_format: DepthFormat::Normalized,
            z_depth: true,
            cuboid_only: false,
            ovoxel_resolution: 128,
            ovoxel_max_output_voxels: GPU_DEFAULT_MAX_OUTPUT_VOXELS,
            ovoxel_mode: OvoxelMode::Disabled,
        }
    }
}

// bevy_args 2.0 treats null Option fields as strings, which breaks ?indoor_seed=6.
// Decode URL values as JSON scalars first, retaining the established serde enum names
// and accepting the scene/layout spellings used by the native CLI.
#[cfg(any(target_arch = "wasm32", test))]
fn config_with_query(
    defaults: BevyZeroverseConfig,
    pairs: impl IntoIterator<Item = (String, String)>,
) -> Result<BevyZeroverseConfig, String> {
    let mut json = serde_json::to_value(defaults).map_err(|e| e.to_string())?;
    for (key, value) in pairs {
        let key = key.replace('-', "_");
        let field = json
            .get_mut(&key)
            .ok_or_else(|| format!("unknown viewer setting: {key}"))?;
        let parsed = serde_json::from_str(&value)
            .unwrap_or_else(|_| serde_json::Value::String(value.clone()));
        *field = match key.as_str() {
            "scene_type" => ZeroverseSceneType::from_str(&value, true)
                .ok()
                .map(serde_json::to_value)
                .transpose()
                .map_err(|e| e.to_string())?
                .unwrap_or(parsed),
            "indoor_layout" => IndoorLayout::from_str(&value, true)
                .ok()
                .map(serde_json::to_value)
                .transpose()
                .map_err(|e| e.to_string())?
                .unwrap_or(parsed),
            "indoor_quality" => {
                crate::scene::procedural_indoor::IndoorQuality::from_str(&value, true)
                    .ok()
                    .map(serde_json::to_value)
                    .transpose()
                    .map_err(|e| e.to_string())?
                    .unwrap_or(parsed)
            }
            "render_mode" => RenderMode::from_str(&value, true)
                .ok()
                .map(serde_json::to_value)
                .transpose()
                .map_err(|e| e.to_string())?
                .unwrap_or(parsed),
            _ if field.is_string() => serde_json::Value::String(value),
            _ => parsed,
        };
    }
    serde_json::from_value(json).map_err(|e| e.to_string())
}

#[cfg(target_arch = "wasm32")]
fn parse_web_config() -> Result<BevyZeroverseConfig, String> {
    let defaults = BevyZeroverseConfig::parse_from(["viewer"]);
    let search = web_sys::window()
        .ok_or("browser window is unavailable")?
        .location()
        .search()
        .map_err(|e| format!("URL search: {e:?}"))?;
    let params =
        web_sys::UrlSearchParams::new_with_str(&search).map_err(|e| format!("URL query: {e:?}"))?;
    let fields = serde_json::to_value(&defaults).map_err(|e| e.to_string())?;
    let pairs = fields
        .as_object()
        .unwrap()
        .keys()
        .filter_map(|key| {
            params
                .get(key)
                .or_else(|| params.get(&key.replace('_', "-")))
                .map(|value| (key.clone(), value))
        })
        .collect::<Vec<_>>();
    config_with_query(defaults, pairs)
}

#[cfg(test)]
mod web_config_tests {
    use super::*;

    #[test]
    fn typed_url_seed_and_cli_scene_names_are_supported() {
        let config = config_with_query(
            BevyZeroverseConfig::default(),
            [
                ("scene-type".into(), "procedural-indoor".into()),
                ("indoor_seed".into(), u64::MAX.to_string()),
                ("indoor_layout".into(), "open-office".into()),
                ("editor".into(), "false".into()),
                ("width".into(), "641".into()),
                (
                    "human_motion".into(),
                    r#"{"fraction":0.25,"frames":120}"#.into(),
                ),
            ],
        )
        .unwrap();
        assert_eq!(config.scene_type, ZeroverseSceneType::ProceduralIndoor);
        assert_eq!(config.indoor_seed, Some(u64::MAX));
        assert_eq!(config.indoor_layout, IndoorLayout::OpenOffice);
        assert!(!config.editor);
        assert_eq!(config.width, 641.0);
        assert_eq!(
            crate::human_motion::HumanMotionConfig::parse(config.human_motion.as_ref().unwrap())
                .unwrap()
                .fraction,
            0.25
        );
        assert!(config_with_query(config, [("indoor_seed".into(), "invalid".into())]).is_err());
    }
}

pub fn viewer_app(app: Option<App>, override_args: Option<BevyZeroverseConfig>) -> App {
    let args = match override_args {
        Some(args) => args,
        None => {
            #[cfg(not(target_arch = "wasm32"))]
            {
                parse_args::<BevyZeroverseConfig>()
            }
            #[cfg(target_arch = "wasm32")]
            {
                parse_web_config().expect("invalid viewer URL configuration")
            }
        }
    };
    args.validate_ovoxel()
        .expect("invalid O-voxel capture configuration");

    #[cfg(target_arch = "wasm32")]
    assert!(
        !args.image_copiers,
        "Browser rendering supports the viewer, not dataset readback. Set image_copiers=false; use the native indoor_validate or zeroverse_gen CLI for dataset capture."
    );

    let mut app = if let Some(original_app) = app {
        original_app
    } else {
        App::new()
    };

    info!("args: {:?}", args);
    app.insert_resource(args.clone());

    #[cfg(target_arch = "wasm32")]
    let primary_window = Some(Window {
        // fit_canvas_to_parent: true,
        canvas: Some("#bevy".to_string()),
        resolution: bevy::window::WindowResolution::new(
            args.width.round() as u32,
            args.height.round() as u32,
        ),
        mode: bevy::window::WindowMode::Windowed,
        prevent_default_event_handling: true,
        title: args.name.clone(),

        #[cfg(feature = "perftest")]
        present_mode: bevy::window::PresentMode::AutoNoVsync,
        #[cfg(not(feature = "perftest"))]
        present_mode: bevy::window::PresentMode::AutoVsync,

        ..default()
    });

    #[cfg(not(target_arch = "wasm32"))]
    let primary_window = (!args.headless).then(|| Window {
        mode: bevy::window::WindowMode::Windowed,
        prevent_default_event_handling: false,
        resolution: bevy::window::WindowResolution::new(
            args.width.round() as u32,
            args.height.round() as u32,
        ),
        title: args.name.clone(),
        #[cfg(feature = "perftest")]
        present_mode: bevy::window::PresentMode::AutoNoVsync,
        #[cfg(not(feature = "perftest"))]
        present_mode: bevy::window::PresentMode::AutoVsync,

        ..default()
    });

    app.insert_resource(ClearColor(Color::srgba(0.0, 0.0, 0.0, 0.0)));
    // Bound native interactive uploads. Capture workers wait for complete assets.
    // On Bevy 0.19.1/WebGPU, throttled uploads left entire material groups missing
    // even after pipeline compilation settled. Keep Bevy's default upload policy
    // on Wasm; paired RGB/semantic browser fixtures cover this regression.
    if !args.image_copiers && !cfg!(target_arch = "wasm32") {
        app.insert_resource(bevy::render::render_asset::RenderAssetBytesPerFrame::new(
            32 * 1024 * 1024,
        ));
    }

    let winit_plugin = WinitPlugin {
        run_on_any_thread: true,
    };

    let default_plugins = DefaultPlugins
        .set(AssetPlugin {
            meta_check: bevy::asset::AssetMetaCheck::Never,
            unapproved_path_mode: bevy::asset::UnapprovedPathMode::Allow,
            ..default()
        })
        .set(ImagePlugin::default_nearest())
        .set(RenderPlugin {
            // Dataset workers can exit immediately after their last capture.
            // Finish compilation on the render thread so asynchronous compiler
            // tasks cannot retain Vulkan devices beyond application teardown.
            synchronous_pipeline_compilation: !cfg!(target_arch = "wasm32") && args.image_copiers,
            render_creation: RenderCreation::Automatic(Box::new(WgpuSettings {
                memory_hints: if args.image_copiers {
                    wgpu::MemoryHints::MemoryUsage
                } else {
                    wgpu::MemoryHints::Performance
                },
                features: if cfg!(target_arch = "wasm32") {
                    WgpuFeatures::empty()
                } else {
                    WgpuFeatures::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                },
                ..Default::default()
            })),
            ..Default::default()
        })
        .set(winit_plugin);

    let default_plugins = default_plugins.set(viewer_window_plugin(primary_window));

    // Offscreen capture uses image targets and does not need a hidden window or
    // winit's process-global event loop. This also permits sequential capture
    // apps and workers without a display server.
    #[cfg(not(target_arch = "wasm32"))]
    let default_plugins = if args.headless {
        default_plugins.disable::<WinitPlugin>()
    } else {
        default_plugins
    };

    app.add_plugins(default_plugins);

    #[cfg(not(target_arch = "wasm32"))]
    if args.headless {
        // Preserve App::run() for the standalone headless viewer. Dataset
        // workers replace this runner with their request-driven runner.
        app.add_plugins(bevy::app::ScheduleRunnerPlugin::run_loop(
            std::time::Duration::from_millis(1),
        ));
    }

    if args.image_copiers {
        app.add_plugins(io::image_copy::ImageCopyPlugin);
        app.add_plugins(io::prepass_copy::PrepassCopyPlugin);
    }

    #[cfg(feature = "viewer")]
    app.add_plugins(PanOrbitCameraPlugin);

    #[cfg(feature = "viewer")]
    if args.editor && !args.headless {
        // Register config and the enum fields it exposes so the inspector keeps working
        // when the config definition changes.
        app.register_type::<BevyZeroverseConfig>();
        app.register_type::<DepthFormat>();
        app.register_type::<PlaybackMode>();
        app.register_type::<RenderMode>();
        app.register_type::<ZeroverseSceneType>();
        app.register_type::<IndoorLayout>();
        app.register_type::<crate::scene::procedural_indoor::IndoorQuality>();

        // setup_camera owns the sole inspector context. Automatic discovery can
        // choose the editor/capture camera before the UI camera is processed,
        // creating two contexts with the same multipass schedule at startup.
        app.insert_resource(bevy_egui::EguiGlobalSettings {
            auto_create_primary_context: false,
            ..default()
        });
        app.add_plugins(EguiPlugin::default());
        app.add_plugins(bevy_inspector_egui::DefaultInspectorConfigPlugin);
        app.add_systems(bevy_egui::EguiPrimaryContextPass, inspector::panel);
    }

    if args.press_esc_close {
        app.add_systems(Update, press_esc_close);
    }

    app.add_plugins(BevyZeroversePlugin);

    app.insert_resource(args.render_mode);

    app.insert_resource(DefaultZeroverseCamera {
        resolution: UVec2::new(args.width as u32, args.height as u32).into(),
    });

    app.add_systems(PostStartup, (settings::synchronize, setup_scene).chain());
    app.add_systems(
        First,
        settings::synchronize.before(crate::asset::AssetDemandSet),
    );

    if args.keybinds {
        app.add_systems(PreUpdate, press_m_shuffle_materials_and_meshes);
    }

    #[cfg(feature = "viewer")]
    {
        app.add_systems(PreUpdate, setup_material_grid);
        app.add_systems(
            PostUpdate,
            (setup_camera.in_set(EditorCameraSetup), setup_camera_grid),
        );
    }

    app.add_systems(Update, rotate_scene);

    app.add_systems(PostUpdate, regenerate_scene_system);

    app
}

fn viewer_window_plugin(primary_window: Option<Window>) -> WindowPlugin {
    WindowPlugin {
        // Capture workers have no window; interactive viewers should exit when
        // their primary window closes, including through the window manager.
        exit_condition: if primary_window.is_some() {
            bevy::window::ExitCondition::OnPrimaryClosed
        } else {
            bevy::window::ExitCondition::DontExit
        },
        primary_window,
        primary_cursor_options: None,
        close_when_requested: true,
    }
}

#[cfg(test)]
mod window_lifecycle_tests {
    use super::*;
    use bevy::window::{PrimaryWindow, WindowCloseRequested};

    #[test]
    fn primary_window_close_exits_even_with_an_auxiliary_window() {
        let mut app = App::new();
        app.add_plugins(viewer_window_plugin(Some(Window::default())));
        let primary = app
            .world_mut()
            .query_filtered::<Entity, With<PrimaryWindow>>()
            .single(app.world())
            .unwrap();
        let auxiliary = app.world_mut().spawn(Window::default()).id();
        app.update();
        assert!(app.should_exit().is_none());

        app.world_mut()
            .write_message(WindowCloseRequested { window: primary });
        app.update();
        // Bevy marks the window as closing, then despawns it on the next frame.
        app.update();

        assert!(app.world().get_entity(primary).is_err());
        assert!(app.world().get::<Window>(auxiliary).is_some());
        assert_eq!(app.should_exit(), Some(AppExit::Success));
    }

    #[test]
    fn headless_app_stays_alive_without_windows() {
        let mut app = App::new();
        app.add_plugins(viewer_window_plugin(None));
        for _ in 0..3 {
            app.update();
            assert!(app.should_exit().is_none());
        }
    }
}

#[cfg(feature = "viewer")]
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct EditorCameraSetup;

#[derive(Component, Debug, Reflect)]
struct MaterialGridCameraMarker;

#[cfg(feature = "viewer")]
#[derive(Clone, PartialEq)]
struct CameraSettingsSnapshot {
    camera_grid: bool,
    scene_type: ZeroverseSceneType,
    orbit_smoothness: f32,
    pan_smoothness: f32,
    zoom_smoothness: f32,
}

#[cfg(feature = "viewer")]
fn setup_camera(
    args: Res<BevyZeroverseConfig>,
    mut commands: Commands,
    windows: Query<(), With<Window>>,
    mut material_grid_cameras: Query<
        &mut Camera,
        (With<MaterialGridCameraMarker>, Without<EditorCameraMarker>),
    >,
    mut editor_cameras: Query<
        (Entity, &mut PanOrbitCamera, Option<&mut Camera>),
        With<EditorCameraMarker>,
    >,
    room_settings: Res<ZeroverseRoomSettings>,
    mut previous_settings: Local<Option<CameraSettingsSnapshot>>,
) {
    // A windowless capture app can inspect interactive scheduling without a
    // primary surface. Do not create an editor view targeting a missing window.
    if args.headless || windows.is_empty() {
        return;
    }

    let current_settings = CameraSettingsSnapshot {
        camera_grid: args.camera_grid,
        scene_type: args.scene_type.clone(),
        orbit_smoothness: args.orbit_smoothness,
        pan_smoothness: args.pan_smoothness,
        zoom_smoothness: args.zoom_smoothness,
    };

    if previous_settings.as_ref() == Some(&current_settings) {
        return;
    }
    *previous_settings = Some(current_settings);

    // A dedicated UI view keeps egui and capture images alive while the costly
    // editor scene view is disabled. Grid letterboxing has an opaque backdrop.
    let clear = if args.camera_grid {
        ClearColorConfig::Custom(Color::srgb(0.025, 0.028, 0.032))
    } else {
        // The UI camera has an LDR intermediate, while the editor uses HDR.
        // Loading the UI intermediate cannot preserve the editor's separate
        // buffer. Clear transparently and blend into the final window instead.
        ClearColorConfig::Custom(Color::NONE)
    };
    if let Ok(mut ui_camera) = material_grid_cameras.single_mut() {
        ui_camera.clear_color = clear;
    } else {
        commands.spawn((
            Camera2d,
            Camera {
                order: 1,
                clear_color: clear,
                output_mode: bevy::camera::CameraOutputMode::Write {
                    blend_state: Some(bevy::render::render_resource::BlendState::ALPHA_BLENDING),
                    clear_color: ClearColorConfig::None,
                },
                ..default()
            },
            bevy::render::view::Msaa::Off,
            MaterialGridCameraMarker,
            bevy::ui::IsDefaultUiCamera,
            bevy_egui::PrimaryEguiContext,
            Name::new("viewer_ui_camera"),
        ));
    }

    if args.camera_grid {
        if let Ok((_entity, _pan, Some(mut camera))) = editor_cameras.single_mut() {
            camera.is_active = false;
        }
        return;
    }

    if let Ok((_, mut pan, Some(mut camera))) = editor_cameras.single_mut() {
        camera.is_active = true;

        pan.orbit_smoothness = args.orbit_smoothness;
        pan.pan_smoothness = args.pan_smoothness;
        pan.zoom_smoothness = args.zoom_smoothness;
    } else {
        let radius = match args.scene_type {
            ZeroverseSceneType::Room | ZeroverseSceneType::SemanticRoom => {
                (match room_settings.room_size {
                    ScaleSampler::Bounded(_min, max) => max.max_element(),
                    ScaleSampler::Exact(size) => size.max_element(),
                }) * 2.0
            }
            _ => 3.5,
        }
        .into();

        let yaw = match args.scene_type {
            ZeroverseSceneType::Room | ZeroverseSceneType::SemanticRoom => 0.8,
            _ => 0.0,
        }
        .into();

        let pitch = match args.scene_type {
            ZeroverseSceneType::Room | ZeroverseSceneType::SemanticRoom => 0.8,
            _ => 0.0,
        }
        .into();

        commands.spawn((
            EditorCameraMarker::default(),
            PanOrbitCamera {
                focus: Vec3::ZERO,
                radius,
                pitch,
                yaw,
                allow_upside_down: true,
                orbit_smoothness: args.orbit_smoothness,
                pan_smoothness: args.pan_smoothness,
                zoom_smoothness: args.zoom_smoothness,
                ..default()
            },
        ));
    }
}

#[derive(Component)]
pub struct CameraGrid;

#[derive(Component)]
pub struct CameraGridMarker;

#[cfg(feature = "viewer")]
fn setup_camera_grid(
    mut commands: Commands,
    args: Res<BevyZeroverseConfig>,
    camera_grids: Query<Entity, With<CameraGrid>>,
    zeroverse_cameras: Query<(Entity, &RenderTarget), With<ZeroverseCamera>>,
    new_zeroverse_cameras: Query<Entity, (With<ZeroverseCamera>, Without<CameraGridMarker>)>,
    mut scene_loaded: MessageReader<SceneLoadedEvent>,
    mut previous_camera_grid: Local<Option<bool>>,
) {
    if zeroverse_cameras.is_empty() {
        return;
    }

    let camera_grid_changed = previous_camera_grid
        .map(|prev| prev != args.camera_grid)
        .unwrap_or(true);

    if scene_loaded.is_empty() && !camera_grid_changed && new_zeroverse_cameras.is_empty() {
        return;
    }
    scene_loaded.clear();
    *previous_camera_grid = Some(args.camera_grid);

    for entity in new_zeroverse_cameras.iter() {
        commands.entity(entity).insert(CameraGridMarker);
    }

    if !camera_grids.is_empty() {
        commands.entity(camera_grids.single().unwrap()).despawn();
    }

    if args.camera_grid {
        let camera_count = zeroverse_cameras.iter().count();
        let rows = (camera_count as f32).sqrt().ceil() as u16;
        let cols = (camera_count as f32 / rows as f32).ceil() as u16;

        commands
            .spawn((
                CameraGrid,
                Name::new("camera_grid"),
                Node {
                    display: Display::Grid,
                    width: Val::Percent(100.0),
                    height: Val::Percent(100.0),
                    grid_template_columns: RepeatedGridTrack::flex(cols, 1.0),
                    grid_template_rows: RepeatedGridTrack::flex(rows, 1.0),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.025, 0.028, 0.032)),
            ))
            .with_children(|builder| {
                for (_, target) in zeroverse_cameras.iter() {
                    let texture = match target.clone() {
                        RenderTarget::Image(texture) => texture,
                        _ => continue,
                    };

                    builder.spawn(ImageNode {
                        image: texture.handle,
                        ..default()
                    });
                }
            });
    }
}

#[derive(Component)]
pub struct MaterialGrid;

#[cfg(feature = "viewer")]
fn setup_material_grid(
    mut commands: Commands,
    args: Res<BevyZeroverseConfig>,
    standard_materials: Res<Assets<StandardMaterial>>,
    zeroverse_materials: Res<ZeroverseMaterials>,
    material_grids: Query<Entity, With<MaterialGrid>>,
    mut materials_loaded: MessageReader<MaterialsLoadedEvent>,
) {
    if materials_loaded.is_empty() {
        return;
    }
    materials_loaded.clear();

    if !material_grids.is_empty() {
        commands.entity(material_grids.single().unwrap()).despawn();
    }

    if args.material_grid {
        let material_count = zeroverse_materials.materials.len();
        let rows = (material_count as f32).sqrt().ceil() as u16;
        let cols = (material_count as f32 / rows as f32).ceil() as u16;

        commands
            .spawn((
                MaterialGrid,
                Name::new("material_grid"),
                Node {
                    display: Display::Grid,
                    width: Val::Percent(100.0),
                    height: Val::Percent(100.0),
                    grid_template_columns: RepeatedGridTrack::flex(cols, 1.0),
                    grid_template_rows: RepeatedGridTrack::flex(rows, 1.0),
                    ..default()
                },
            ))
            .with_children(|builder| {
                for material in &zeroverse_materials.materials {
                    let base_color_texture = standard_materials
                        .get(material)
                        .unwrap()
                        .base_color_texture
                        .clone()
                        .unwrap_or_default();

                    builder.spawn(ImageNode {
                        image: base_color_texture,
                        ..default()
                    });
                }
            });
    }
}

fn setup_scene(
    args: Res<BevyZeroverseConfig>,
    mut regenerate_event: MessageWriter<RegenerateSceneEvent>,
) {
    if args.initialize_scene {
        regenerate_event.write(RegenerateSceneEvent);
    } else {
        info!("skipping scene initialization.");
    }
}

#[allow(clippy::too_many_arguments)]
fn regenerate_scene_system(
    args: Res<BevyZeroverseConfig>,
    sampler: Option<Res<crate::sample::SamplerState>>,
    indoor: Option<Res<crate::scene::procedural_indoor::IndoorGenerationStatus>>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut regenerate_stopwatch: Local<Stopwatch>,
    mut regenerate_event: MessageWriter<RegenerateSceneEvent>,
) {
    if sampler.is_some_and(|state| state.enabled) {
        return;
    }
    if args.regenerate_ms > 0 {
        regenerate_stopwatch.tick(time.delta());
    }

    let mut regenerate_scene = args.regenerate_ms > 0
        && regenerate_stopwatch.elapsed().as_millis() > args.regenerate_ms as u128
        && !indoor.is_some_and(|status| status.busy());

    if args.keybinds {
        regenerate_scene |= keys.just_pressed(KeyCode::KeyR);
    }

    if regenerate_scene {
        regenerate_event.write(RegenerateSceneEvent);
        regenerate_stopwatch.reset();
    }
}

fn rotate_scene(
    time: Res<Time>,
    args: Res<BevyZeroverseConfig>,
    sampler: Option<Res<crate::sample::SamplerState>>,
    mut scene_roots: Query<&mut Transform, With<ZeroverseSceneRoot>>,
) {
    if args.yaw_speed == 0.0 || sampler.is_some_and(|state| state.enabled) {
        return;
    }

    for mut transform in scene_roots.iter_mut() {
        let delta_rot = args.yaw_speed * time.delta_secs();
        if delta_rot == 0.0 {
            continue;
        }
        transform.rotate(Quat::from_rotation_y(delta_rot));
    }
}

fn press_m_shuffle_materials_and_meshes(
    keys: Res<ButtonInput<KeyCode>>,
    mut shuffle_material_events: MessageWriter<ShuffleMaterialsEvent>,
    mut shuffle_meshes_events: MessageWriter<ShuffleMeshesEvent>,
) {
    if keys.just_pressed(KeyCode::KeyM) {
        shuffle_material_events.write(ShuffleMaterialsEvent);
        shuffle_meshes_events.write(ShuffleMeshesEvent);
    }
}

fn press_esc_close(keys: Res<ButtonInput<KeyCode>>, mut exit: MessageWriter<AppExit>) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
}

#[cfg(all(test, feature = "viewer"))]
mod viewer_tests;
