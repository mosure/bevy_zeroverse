//! Optional editor-only pose layer with its own depth buffer. Environment mesh
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
    camera::{EditorCameraMarker, ProcessedEditorCameraMarker},
    render::{semantic::SemanticLabel, RenderOnlyOverlay},
};

pub(crate) const LAYER: RenderLayers = RenderLayers::layer(2);

#[derive(Component)]
struct PoseOverlayCamera;

#[derive(Default, Resource)]
pub(super) struct OverlayState {
    camera: Option<Entity>,
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
            sync.after(bevy::transform::TransformSystems::Propagate)
                .before(bevy::camera::CameraUpdateSystems)
                .after(bevy::camera::visibility::VisibilitySystems::VisibilityPropagate)
                .before(bevy::pbr::check_entities_needing_specialization::<Occluder>),
        );
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn sync(
    mut commands: Commands,
    config: Res<BevyZeroverseConfig>,
    editors: Query<
        (&Camera, &RenderTarget, &Projection, &GlobalTransform),
        (With<EditorCameraMarker>, With<ProcessedEditorCameraMarker>),
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
    mut state: ResMut<OverlayState>,
    mut materials: ResMut<Assets<Occluder>>,
) {
    let editor = editors.iter().find(|(camera, ..)| camera.is_active);
    // The grid deliberately disables the editor's scene. Do not draw an
    // unrelated editor-space skeleton on top of its capture-camera tiles.
    let active = config.draw_pose_gizmos && !config.camera_grid && editor.is_some();
    if !active {
        if let Some(camera) = state.camera.take() {
            commands.entity(camera).despawn();
        }
        for (_, entity) in state.objects.drain() {
            commands.entity(entity).despawn();
        }
        return;
    }
    let (camera, target, projection, transform) = editor.unwrap();
    let camera_entity = *state.camera.get_or_insert_with(|| {
        commands
            .spawn((
                Name::new("pose overlay"),
                PoseOverlayCamera,
                Camera3d::default(),
                Msaa::Off,
                Tonemapping::None,
                bevy::core_pipeline::tonemapping::DebandDither::Disabled,
                LAYER,
            ))
            .id()
    });
    commands.entity(camera_entity).insert((
        Camera {
            order: camera.order + 1,
            viewport: camera.viewport.clone(),
            clear_color: ClearColorConfig::Custom(Color::NONE),
            output_mode: CameraOutputMode::Write {
                blend_state: Some(BlendState::ALPHA_BLENDING),
                clear_color: ClearColorConfig::None,
            },
            // Preserve computed target/viewport state; replacing it with
            // defaults after camera setup prevents extraction of this view.
            ..camera.clone()
        },
        target.clone(),
        projection.clone(),
        transform.compute_transform(),
        *transform,
    ));
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
