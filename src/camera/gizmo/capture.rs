//! Opt-in drawings inside the RGB raster of each logical dataset camera.
//! Each receiver gets a free layer. A camera glyph is visible to its peers only;
//! nonlinear source faces inherit the logical camera's layers and calibration.
use super::{color, scale, wireframe, MAX_RADIUS_METRES};
use crate::{
    annotation::obb::{compute_object_obbs, draw_box, ObjectObb},
    app::BevyZeroverseConfig,
    camera::{lens, CaptureCameraIndex, TrajectorySampler, ZeroverseCamera},
    render::RenderMode,
};
use bevy::{camera::visibility::RenderLayers, prelude::*};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Presentation annotations drawn into RGB. All four channels default off.
/// Numeric annotations and the editor's own overlay switches are independent.
#[derive(Clone, Copy, Debug, PartialEq, Reflect, Serialize, Deserialize, clap::Args)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, set_all))]
pub struct RgbGizmos {
    /// Draw semantic object oriented bounding boxes in capture RGB.
    #[arg(long = "rgb-object-obbs", action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true", default_value_t = false)]
    pub object_obbs: bool,
    /// Draw articulated skeletons in capture RGB, occluded by the environment.
    #[arg(long = "rgb-human-poses", action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true", default_value_t = false)]
    pub human_poses: bool,
    /// Draw other dataset cameras' calibrated lens outlines in capture RGB.
    #[arg(long = "rgb-camera-frustums", action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true", default_value_t = false)]
    pub camera_frustums: bool,
    /// Draw the dataset cameras' metric paths in capture RGB.
    #[arg(long = "rgb-camera-trajectories", action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true", default_value_t = false)]
    pub camera_trajectories: bool,
    /// Capture frustum glyph scale, independent of lens optics and path length.
    #[arg(long = "rgb-camera-gizmo-scale", default_value = "1", value_parser = scale::parse)]
    #[serde(default = "scale::default", deserialize_with = "scale::deserialize")]
    pub camera_scale: f32,
}

impl Default for RgbGizmos {
    fn default() -> Self {
        Self {
            object_obbs: false,
            human_poses: false,
            camera_frustums: false,
            camera_trajectories: false,
            camera_scale: scale::default(),
        }
    }
}

impl RgbGizmos {
    /// Whether the RGB pixels carry any presentation annotation.
    pub fn enabled(&self) -> bool {
        self.object_obbs || self.human_poses || self.camera_frustums || self.camera_trajectories
    }
}

#[cfg(feature = "python")]
#[pyo3::pymethods]
impl RgbGizmos {
    #[new]
    fn new() -> Self {
        Self::default()
    }
}

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct UpdateRgbGizmos;

pub(crate) fn install(app: &mut App) {
    app.add_systems(
        PostUpdate,
        sync.in_set(UpdateRgbGizmos)
            .after(compute_object_obbs)
            .after(bevy::transform::TransformSystems::Propagate)
            .before(bevy::camera::CameraUpdateSystems),
    );
}

struct Drawing {
    entity: Entity,
    handle: Handle<GizmoAsset>,
    layers: RenderLayers,
}

impl Drawing {
    fn update<'a>(
        slot: &mut Option<Self>,
        commands: &mut Commands,
        assets: &'a mut Assets<GizmoAsset>,
        layers: RenderLayers,
    ) -> &'a mut GizmoAsset {
        let drawing = slot.get_or_insert_with(|| {
            let handle = assets.add(GizmoAsset::default());
            let entity = commands
                .spawn((
                    Name::new("Capture RGB gizmos"),
                    Gizmo {
                        handle: handle.clone(),
                        line_config: bevy::gizmos::config::GizmoLineConfig {
                            width: 1.8,
                            ..default()
                        },
                        ..default()
                    },
                    layers.clone(),
                ))
                .id();
            Self {
                entity,
                handle,
                layers: layers.clone(),
            }
        });
        if drawing.layers != layers {
            commands.entity(drawing.entity).insert(layers.clone());
            drawing.layers = layers;
        }
        let asset = assets
            .get_mut(&drawing.handle)
            .expect("retained capture gizmo")
            .into_inner();
        asset.clear();
        asset
    }

    fn remove(slot: &mut Option<Self>, commands: &mut Commands) {
        if let Some(drawing) = slot.take() {
            commands.entity(drawing.entity).try_despawn();
        }
    }
}

struct Receiver {
    layer: usize,
    was_implicit: bool,
    glyph: Option<Drawing>,
    outline: Option<(lens::CameraCalibration, Vec<[Vec3; 2]>)>,
}

#[derive(Default)]
struct Drawings {
    receivers: HashMap<Entity, Receiver>,
    boxes: Option<Drawing>,
    paths: Option<Drawing>,
}

type Cameras<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static ZeroverseCamera,
        &'static GlobalTransform,
        &'static Projection,
        Option<&'static lens::LensCamera>,
        Option<&'static CaptureCameraIndex>,
        Option<&'static RenderLayers>,
        Option<&'static ChildOf>,
    ),
    With<Camera>,
>;

#[allow(clippy::too_many_arguments)]
fn sync(
    mut commands: Commands,
    config: Res<BevyZeroverseConfig>,
    mode: Res<RenderMode>,
    mut state: Local<Drawings>,
    mut assets: ResMut<Assets<GizmoAsset>>,
    cameras: Cameras,
    occupied_layers: Query<&RenderLayers>,
    globals: Query<&GlobalTransform>,
    obbs: Query<&ObjectObb>,
) {
    let options = config.rgb_gizmos;
    let active = *mode == RenderMode::Color
        && (options.camera_frustums || options.camera_trajectories || options.object_obbs);
    if !active && state.receivers.is_empty() {
        return;
    }
    state.receivers.retain(|&owner, entry| {
        if active && cameras.contains(owner) {
            return true;
        }
        Drawing::remove(&mut entry.glyph, &mut commands);
        if let Ok(layers) = occupied_layers.get(owner) {
            let remaining = layers.clone().without(entry.layer);
            if entry.was_implicit && remaining == RenderLayers::default() {
                commands.entity(owner).remove::<RenderLayers>();
            } else {
                commands.entity(owner).insert(remaining);
            }
        }
        false
    });
    if !active {
        Drawing::remove(&mut state.boxes, &mut commands);
        Drawing::remove(&mut state.paths, &mut commands);
        return;
    }

    let mut views: Vec<_> = cameras.iter().collect();
    views.sort_by_key(|view| (view.5.map_or(usize::MAX, |i| i.0), view.0.to_bits()));
    let mut occupied = None;
    for &(owner, .., layers, _) in &views {
        let entry = state.receivers.entry(owner).or_insert_with(|| {
            let used: &mut HashSet<usize> = occupied.get_or_insert_with(|| {
                occupied_layers
                    .iter()
                    .flat_map(RenderLayers::iter)
                    .collect()
            });
            let layer = (64..)
                .find(|i| used.insert(*i))
                .expect("free RGB render layer");
            Receiver {
                layer,
                was_implicit: layers.is_none(),
                glyph: None,
                outline: None,
            }
        });
        let layers = layers.cloned().unwrap_or_default();
        if !layers.iter().any(|l| l == entry.layer) {
            commands.entity(owner).insert(layers.with(entry.layer));
        }
    }
    let all = RenderLayers::from_layers(
        &state
            .receivers
            .values()
            .map(|r| r.layer)
            .collect::<Vec<_>>(),
    );
    let alpha = config.gizmos_alpha.clamp(0., 1.);
    if options.object_obbs && !views.is_empty() {
        let drawing = Drawing::update(&mut state.boxes, &mut commands, &mut assets, all.clone());
        for obb in &obbs {
            draw_box(drawing, obb, alpha);
        }
    } else {
        Drawing::remove(&mut state.boxes, &mut commands);
    }

    let mut fallback = views
        .iter()
        .filter_map(|v| v.5.map(|i| i.0))
        .max()
        .map_or(0, |i| i.saturating_add(1));
    let multiple = views.len() > 1;
    let views: Vec<_> = views
        .into_iter()
        .map(|view| {
            let index = view.5.map_or_else(
                || {
                    let id = fallback;
                    fallback = fallback.saturating_add(1);
                    id
                },
                |i| i.0,
            );
            (index, view)
        })
        .collect();
    let state = &mut *state;
    if options.camera_trajectories && !views.is_empty() {
        let drawing = Drawing::update(&mut state.paths, &mut commands, &mut assets, all.clone());
        for (index, (_, camera, _, _, _, _, _, parent)) in &views {
            if camera.override_transform.is_none()
                && !matches!(camera.trajectory, TrajectorySampler::Static { .. })
            {
                let transform = parent
                    .and_then(|p| globals.get(p.parent()).ok())
                    .map_or(Transform::IDENTITY, GlobalTransform::compute_transform);
                camera.trajectory.clone().draw_path(
                    drawing,
                    transform,
                    color(*index).with_alpha(alpha * 0.65),
                );
            }
        }
    } else {
        Drawing::remove(&mut state.paths, &mut commands);
    }
    for (index, (owner, camera, world, projection, angular, _, _, _)) in views {
        let entry = state
            .receivers
            .get_mut(&owner)
            .expect("assigned RGB render layer");
        if !options.camera_frustums || !multiple {
            Drawing::remove(&mut entry.glyph, &mut commands);
            continue;
        }
        let calibration = angular
            .map(|l| Ok(l.calibration.clone()))
            .unwrap_or_else(|| {
                lens::calibration(
                    projection,
                    camera.fisheye,
                    camera.principal_point_offset,
                    camera
                        .resolution
                        .unwrap_or(UVec2::new(config.width as u32, config.height as u32))
                        .to_array(),
                )
            });
        let Ok(calibration) = calibration else {
            Drawing::remove(&mut entry.glyph, &mut commands);
            continue;
        };
        if entry
            .outline
            .as_ref()
            .is_none_or(|(old, _)| *old != calibration)
        {
            entry.outline = wireframe(&calibration)
                .ok()
                .map(|lines| (calibration, lines));
        }
        let Some((_, lines)) = &entry.outline else {
            Drawing::remove(&mut entry.glyph, &mut commands);
            continue;
        };
        let drawing = Drawing::update(
            &mut entry.glyph,
            &mut commands,
            &mut assets,
            all.clone().without(entry.layer),
        );
        let pose = Mat4::from_rotation_translation(world.rotation(), world.translation());
        let radius = MAX_RADIUS_METRES
            * scale::validate(options.camera_scale).unwrap_or_else(|_| scale::default());
        super::runtime::draw(drawing, lines, pose, radius, color(index).with_alpha(alpha));
    }
}

#[cfg(test)]
mod tests;
