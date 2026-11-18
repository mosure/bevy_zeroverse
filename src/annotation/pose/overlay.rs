//! Optional editor and capture pose layer with its own depth buffer. Environment mesh
//! handles are reused in a color-write-disabled pass; people never write this
//! depth buffer. No per-joint CPU ray casting or always-on-top wall leakage.
use bevy::{
    camera::{visibility::RenderLayers, CameraOutputMode, RenderTarget},
    core_pipeline::tonemapping::Tonemapping,
    light::NotShadowCaster,
    mesh::MeshVertexBufferLayoutRef,
    pbr::{ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline},
    prelude::*,
    render::render_resource::*,
};
use std::collections::HashMap;

use crate::{
    app::BevyZeroverseConfig,
    camera::{lens::LensFace, ProcessedEditorCameraMarker, ZeroverseCamera},
    render::{semantic::SemanticLabel, RenderOnlyOverlay},
};

pub(crate) const LAYER: RenderLayers = RenderLayers::layer(2);

#[derive(Component)]
struct PoseOverlayCamera(Entity);

#[derive(Default, Resource)]
pub(super) struct OverlayState {
    cameras: HashMap<Entity, Entity>,
    objects: HashMap<Entity, Entity>,
    material: Handle<Occluder>,
}

type Occluder = ExtendedMaterial<StandardMaterial, DepthOnly>;

#[derive(Asset, AsBindGroup, TypePath, Clone, Debug, Default)]
pub(super) struct DepthOnly {}

impl MaterialExtension for DepthOnly {
    fn enable_shadows() -> bool {
        false
    }
    fn enable_prepass() -> bool {
        false
    }

    fn specialize(
        _: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _: &MeshVertexBufferLayoutRef,
        _: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        if let Some(fragment) = &mut descriptor.fragment {
            for target in fragment.targets.iter_mut().flatten() {
                target.write_mask = ColorWrites::empty();
            }
        }
        Ok(())
    }
}

pub(super) fn install(app: &mut App) {
    app.init_resource::<OverlayState>()
        .add_plugins(MaterialPlugin::<Occluder>::default())
        .add_systems(
            PostUpdate,
            sync.after(crate::camera::lens::sync_faces)
                .after(super::compute_human_poses)
                .after(crate::scene::procedural_indoor::humans::update_human_poses)
                .after(bevy::transform::TransformSystems::Propagate)
                .before(bevy::camera::CameraUpdateSystems)
                .after(bevy::camera::visibility::VisibilitySystems::VisibilityPropagate)
                .before(bevy::pbr::check_entities_needing_specialization::<Occluder>),
        )
        .add_systems(Last, sync_activity.after(crate::camera::lens::sync_faces));
}

// The sampler can deactivate a submitted RGB camera late in the frame.
fn sync_activity(
    sources: Query<&Camera, Without<PoseOverlayCamera>>,
    mut overlays: Query<(&PoseOverlayCamera, &mut Camera)>,
) {
    for (source, mut camera) in &mut overlays {
        let active = sources.get(source.0).is_ok_and(|c| c.is_active);
        if camera.is_active != active {
            camera.is_active = active;
        }
    }
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn sync(
    mut commands: Commands,
    config: Res<BevyZeroverseConfig>,
    mode: Res<crate::render::RenderMode>,
    sources: Query<
        (
            Entity,
            &Camera,
            &RenderTarget,
            &Projection,
            &GlobalTransform,
            Has<ProcessedEditorCameraMarker>,
        ),
        Or<(
            With<ProcessedEditorCameraMarker>,
            With<ZeroverseCamera>,
            With<LensFace>,
        )>,
    >,
    meshes: Query<
        (
            Entity,
            Ref<Mesh3d>,
            Ref<GlobalTransform>,
            &InheritedVisibility,
            Option<&RenderLayers>,
        ),
        Without<RenderOnlyOverlay>,
    >,
    hierarchy: Query<(
        Option<&SemanticLabel>,
        Option<&ChildOf>,
        Option<&super::HumanPose>,
    )>,
    poses: Query<(), With<super::HumanPose>>,
    mut state: ResMut<OverlayState>,
    mut materials: ResMut<Assets<Occluder>>,
) {
    let editor = config.draw_pose_gizmos && !config.camera_grid && !config.room_schematic;
    let capture = config.rgb_gizmos.human_poses && *mode == crate::render::RenderMode::Color;
    if !editor && !capture && state.cameras.is_empty() && state.objects.is_empty() {
        return;
    }
    let have_poses = !poses.is_empty();
    let enabled = |is_editor: bool| have_poses && if is_editor { editor } else { capture };
    state.cameras.retain(|&source, &mut overlay| {
        if sources
            .get(source)
            .is_ok_and(|(.., is_editor)| enabled(is_editor))
        {
            true
        } else {
            commands.entity(overlay).try_despawn();
            false
        }
    });
    for (source, camera, target, projection, transform, is_editor) in &sources {
        if !enabled(is_editor) {
            continue;
        }
        let overlay = *state.cameras.entry(source).or_insert_with(|| {
            commands
                .spawn((
                    Name::new("Pose overlay"),
                    PoseOverlayCamera(source),
                    Camera3d::default(),
                    Msaa::Off,
                    Tonemapping::None,
                    bevy::core_pipeline::tonemapping::DebandDither::Disabled,
                    LAYER,
                ))
                .id()
        });
        // Draw on the physical RGB source before lens resolve, supersampling
        // and readback. Copy the exact viewport and projection, including a
        // cropped fisheye face or a shifted pinhole principal point.
        commands.entity(overlay).insert((
            Camera {
                order: camera.order + 1,
                clear_color: ClearColorConfig::Custom(Color::NONE),
                output_mode: CameraOutputMode::Write {
                    blend_state: Some(BlendState::ALPHA_BLENDING),
                    clear_color: ClearColorConfig::None,
                },
                ..camera.clone()
            },
            target.clone(),
            projection.clone(),
            transform.compute_transform(),
            *transform,
        ));
    }
    if state.cameras.is_empty() {
        for (_, entity) in state.objects.drain() {
            commands.entity(entity).try_despawn();
        }
        return;
    }
    if state.material == Handle::default() {
        state.material = materials.add(Occluder {
            base: StandardMaterial {
                unlit: true,
                cull_mode: None,
                double_sided: true,
                ..default()
            },
            extension: DepthOnly {},
        });
    }
    let material = state.material.clone();
    let mut retained = std::collections::HashSet::with_capacity(state.objects.len());
    for (entity, mesh, transform, visible, layers) in &meshes {
        if !visible.get() || layers.is_some_and(|l| !l.intersects(&RenderLayers::default())) {
            continue;
        }
        let mut ancestor = entity;
        let mut person = false;
        while let Ok((label, parent, pose)) = hierarchy.get(ancestor) {
            if label == Some(&SemanticLabel::Person) || pose.is_some() {
                person = true;
                break;
            }
            let Some(parent) = parent else {
                break;
            };
            ancestor = parent.parent();
        }
        if person {
            continue;
        }
        retained.insert(entity);
        match state.objects.entry(entity) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(
                    commands
                        .spawn((
                            RenderOnlyOverlay,
                            NotShadowCaster,
                            LAYER,
                            MeshMaterial3d(material.clone()),
                            (*mesh).clone(),
                            transform.compute_transform(),
                            *transform,
                        ))
                        .id(),
                );
            }
            std::collections::hash_map::Entry::Occupied(entry) => {
                if mesh.is_changed() || transform.is_changed() {
                    commands.entity(*entry.get()).insert((
                        (*mesh).clone(),
                        transform.compute_transform(),
                        *transform,
                    ));
                }
            }
        }
    }
    state.objects.retain(|source, copy| {
        if retained.contains(source) {
            true
        } else {
            commands.entity(*copy).despawn();
            false
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotation::pose::HumanPose;
    use crate::render::RenderMode;

    #[test]
    fn capture_pose_pass_tracks_physical_targets_and_excludes_people_from_depth() {
        let mut app = App::new();
        app.init_resource::<OverlayState>()
            .init_resource::<Assets<Occluder>>()
            .init_resource::<BevyZeroverseConfig>()
            .insert_resource(RenderMode::Color)
            .add_systems(Update, (sync, sync_activity).chain());
        let w = app.world_mut();
        w.resource_mut::<BevyZeroverseConfig>()
            .rgb_gizmos
            .human_poses = true;
        let pinhole = w
            .spawn((
                Camera3d::default(),
                ZeroverseCamera::default(),
                RenderTarget::None {
                    size: UVec2::new(640, 480),
                },
                GlobalTransform::from_xyz(1., 2., 3.),
            ))
            .id();
        let face = w
            .spawn((
                Camera3d::default(),
                LensFace { owner: pinhole },
                RenderTarget::None {
                    size: UVec2::splat(320),
                },
                Projection::Perspective(PerspectiveProjection {
                    fov: 1.27,
                    ..default()
                }),
            ))
            .id();
        let editor = w
            .spawn((Camera3d::default(), ProcessedEditorCameraMarker))
            .id();
        let wall = w
            .spawn((Mesh3d::default(), InheritedVisibility::VISIBLE))
            .id();
        let human = w
            .spawn((
                HumanPose {
                    bone_positions: vec![Vec3::Y],
                    bone_rotations: vec![Quat::IDENTITY],
                },
                SemanticLabel::Person,
            ))
            .id();
        let coat = w
            .spawn((
                Mesh3d::default(),
                InheritedVisibility::VISIBLE,
                ChildOf(human),
            ))
            .id();
        app.update();
        let w = app.world_mut();
        let state = w.resource::<OverlayState>();
        assert_eq!(state.objects.len(), 1);
        assert!(state.objects.contains_key(&wall) && !state.objects.contains_key(&coat));
        assert_eq!(state.cameras.len(), 2);
        assert!(!state.cameras.contains_key(&editor));
        for source in [pinhole, face] {
            let overlay = state.cameras[&source];
            assert_eq!(
                w.get::<RenderTarget>(overlay).unwrap().normalize(None),
                w.get::<RenderTarget>(source).unwrap().normalize(None)
            );
            assert_eq!(
                w.get::<GlobalTransform>(overlay),
                w.get::<GlobalTransform>(source)
            );
            assert_eq!(
                w.get::<Projection>(overlay).unwrap().get_clip_from_view(),
                w.get::<Projection>(source).unwrap().get_clip_from_view()
            );
        }
        w.get_mut::<Camera>(face).unwrap().is_active = false;
        app.update();
        let w = app.world_mut();
        let overlay = w.resource::<OverlayState>().cameras[&face];
        assert!(!w.get::<Camera>(overlay).unwrap().is_active);
        w.insert_resource(RenderMode::Normal);
        app.update();
        let state = app.world().resource::<OverlayState>();
        assert!(state.objects.is_empty() && state.cameras.is_empty());
    }
}
