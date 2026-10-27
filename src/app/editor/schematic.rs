//! GPU-rendered live schematic. Static geometry is retained; playback updates
//! only a small colored triangle batch and text transforms, never an RGBA upload.
mod drawing;
mod readiness;
use super::*;
use crate::annotation::schematic::{Camera as PlanCamera, Overlay, Pose, RenderOptions, Schematic};
use crate::scene::procedural_indoor::{
    humans::{IndoorHumanInstance, HUMAN_BONE_PARENTS},
    layout::IndoorManifest,
};
use bevy::{
    camera::{visibility::RenderLayers, RenderTarget},
    render::{render_resource::TextureFormat, view::Msaa},
    sprite::Anchor,
};
pub(super) use readiness::install;
const LAYER: RenderLayers = RenderLayers::layer(3);
#[derive(Component)]
pub(super) struct SchematicView;
/// Model predictions for the live schematic and editor-world camera overlays.
/// Label camera predictions with their capture ID (e.g. `Pred C7`) to match the
/// solid capture glyph's color; predicted glyphs are dashed. Predictions supply
/// their own calibration and pose, never inferred from the ground-truth rig.
#[derive(Resource, Default)]
pub struct Predictions(pub Overlay);
#[derive(Resource, Default)]
pub(super) struct Preview {
    entity: Option<Entity>,
    root: Option<Entity>,
    texture: Option<Handle<Image>>,
    material: Option<Handle<ColorMaterial>>,
    meshes: Vec<Handle<Mesh>>,
    plan: Option<Schematic>,
    options: Option<RenderOptions>,
    transform: Mat4,
    last: Option<(Vec<PlanCamera>, Vec<Pose>, Overlay)>,
    static_labels: Vec<(Entity, String, f32)>,
    dynamic_labels: Vec<(Entity, String, f32)>,
}
impl Preview {
    fn clear(
        &mut self,
        commands: &mut Commands,
        images: &mut Assets<Image>,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<ColorMaterial>,
    ) {
        for entity in [self.entity.take(), self.root.take()].into_iter().flatten() {
            commands.entity(entity).despawn();
        }
        if let Some(image) = self.texture.take() {
            images.remove(image.id());
        }
        if let Some(material) = self.material.take() {
            materials.remove(material.id());
        }
        for mesh in self.meshes.drain(..) {
            meshes.remove(mesh.id());
        }
        self.plan = None;
        self.last = None;
        self.options = None;
        self.static_labels.clear();
        self.dynamic_labels.clear();
    }
}
#[allow(clippy::too_many_arguments)]
fn labels(
    commands: &mut Commands,
    root: Entity,
    entities: &mut Vec<(Entity, String, f32)>,
    labels: &[drawing::Label],
    size: Vec2,
    z: f32,
    font: Handle<Font>,
) {
    while entities.len() > labels.len() {
        commands.entity(entities.pop().unwrap().0).despawn();
    }
    for (i, label) in labels.iter().enumerate() {
        let tf = Transform::from_xyz(
            label.position.x - size.x * 0.5,
            size.y * 0.5 - label.position.y,
            z,
        );
        if let Some((entity, text, font_size)) = entities.get_mut(i) {
            commands
                .entity(*entity)
                .insert((tf, TextColor(label.color)));
            if text != &label.text || *font_size != label.size {
                commands.entity(*entity).insert((
                    Text2d::new(&label.text),
                    TextFont {
                        font: font.clone().into(),
                        font_size: label.size.into(),
                        ..default()
                    },
                ));
                text.clone_from(&label.text);
                *font_size = label.size;
            }
        } else {
            let entity = commands
                .spawn((
                    Text2d::new(&label.text),
                    TextFont {
                        font: font.clone().into(),
                        font_size: label.size.into(),
                        ..default()
                    },
                    TextColor(label.color),
                    Anchor::BOTTOM_LEFT,
                    tf,
                    LAYER,
                    ChildOf(root),
                ))
                .id();
            entities.push((entity, label.text.clone(), label.size));
        }
    }
}
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn update(
    mut commands: Commands,
    config: Res<BevyZeroverseConfig>,
    editor: Res<EditorState>,
    playback: Res<Playback>,
    scene: Option<Res<IndoorManifest>>,
    predictions: Res<Predictions>,
    mut preview: ResMut<Preview>,
    cameras: Query<(
        &crate::camera::CaptureCameraIndex,
        &GlobalTransform,
        &Projection,
        Option<&crate::camera::lens::LensCamera>,
    )>,
    roots: Query<&GlobalTransform, With<crate::scene::ZeroverseSceneRoot>>,
    humans: Query<(&IndoorHumanInstance, &crate::annotation::pose::HumanPose)>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    assets: Res<AssetServer>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
) {
    if !config.room_schematic || config.headless {
        if preview.entity.is_some() {
            preview.clear(&mut commands, &mut images, &mut meshes, &mut materials);
        }
        return;
    }
    let Some(scene) = scene.filter(|_| config.scene_type == ZeroverseSceneType::ProceduralIndoor)
    else {
        if preview.root.is_some() {
            preview.clear(&mut commands, &mut images, &mut meshes, &mut materials);
        }
        if preview.entity.is_none() {
            preview.entity=Some(commands.spawn((SchematicView,Node{align_items:AlignItems::Center,justify_content:JustifyContent::Center,..default()},BackgroundColor(Color::srgb(0.06,0.09,0.10)),ZIndex(1))).with_children(|p|{p.spawn((Text::new("Room schematic requires a Procedural Indoor scene.\nChoose Scene → Procedural Indoor, then Apply."),TextColor(INK),TextFont{font:assets.load(bevy::feathers::constants::fonts::REGULAR).into(),font_size:18.0.into(),..default()}));}).id());
        }
        return;
    };
    let (width, height) = windows.single().map_or((1024., 768.), |w| {
        (
            (w.width() - if editor.view.collapsed { 0. } else { PANEL }).max(128.),
            (w.height() - 96.).max(128.),
        )
    });
    let scale = (1600.0_f32 / width).min(1200. / height).min(1.);
    let options = RenderOptions {
        width: (width * scale).max(128.) as u32,
        height: (height * scale).max(128.) as u32,
        ..default()
    };
    let size = Vec2::new(options.width as f32, options.height as f32);
    let transform = roots.single().map_or(Mat4::IDENTITY, |root| {
        root.to_matrix() * Mat4::from_rotation_y(-scene.world_yaw)
    });
    if preview.options != Some(options) || preview.root.is_none() {
        // Publish complete mesh assets; the retained geometry is uploaded only
        // when its contents change. The render-side readiness hook handles
        // asynchronous preparation without continuously reuploading it.
        let Ok(mut initial) = Schematic::from_manifest(&scene, 0.) else {
            return;
        };
        initial.transform_geometry(transform);
        let (Ok(background), Ok(moving)) = (
            drawing::background(&initial, &options, &predictions.0),
            drawing::moving(&initial, &options, &predictions.0),
        ) else {
            return;
        };
        let initial_drawings = [background, moving];
        preview.clear(&mut commands, &mut images, &mut meshes, &mut materials);
        let root = commands
            .spawn((
                Transform::default(),
                Visibility::default(),
                Name::new("schematic_graph"),
            ))
            .id();
        let image = images.add(Image::new_target_texture(
            options.width,
            options.height,
            TextureFormat::Rgba8UnormSrgb,
            None,
        ));
        commands.spawn((
            Camera2d,
            Camera {
                order: -10,
                clear_color: ClearColorConfig::Custom(Color::srgb(0.953, 0.961, 0.945)),
                ..default()
            },
            RenderTarget::Image(image.clone().into()),
            Msaa::Sample4,
            LAYER,
            ChildOf(root),
            Name::new("schematic_2d_camera"),
        ));
        let material = materials.add(ColorMaterial::default());
        for (z, drawing) in initial_drawings.into_iter().enumerate() {
            let mesh = meshes.add(drawing.into_mesh());
            commands.spawn((
                Mesh2d(mesh.clone()),
                readiness::SchematicBatch,
                MeshMaterial2d(material.clone()),
                Transform::from_xyz(0., 0., z as f32),
                // Both batches are already clipped to the map viewport.
                bevy::camera::visibility::NoFrustumCulling,
                LAYER,
                ChildOf(root),
            ));
            preview.meshes.push(mesh);
        }
        preview.entity = Some(
            commands
                .spawn((
                    SchematicView,
                    Node::default(),
                    ImageNode {
                        image: image.clone(),
                        ..default()
                    },
                    ZIndex(1),
                ))
                .id(),
        );
        preview.root = Some(root);
        preview.texture = Some(image);
        preview.material = Some(material);
        preview.options = Some(options);
    }
    let rebuild = preview.plan.is_none()
        || scene.is_changed()
        || preview.transform != transform
        || predictions.is_changed();
    let font = assets.load(bevy::feathers::constants::fonts::REGULAR);
    let root = preview.root.unwrap();
    if rebuild {
        let Ok(mut plan) = Schematic::from_manifest(&scene, 0.) else {
            return;
        };
        plan.transform_geometry(transform);
        match drawing::background(&plan, &options, &predictions.0) {
            Ok(drawing) => {
                labels(
                    &mut commands,
                    root,
                    &mut preview.static_labels,
                    &drawing.labels,
                    size,
                    2.,
                    font.clone(),
                );
                if let Some(mut mesh) = meshes.get_mut(&preview.meshes[0]) {
                    *mesh = drawing.into_mesh();
                }
            }
            Err(e) => {
                warn!("schematic: {e}");
                return;
            }
        }
        preview.plan = Some(plan);
        preview.transform = transform;
        preview.last = None;
    }
    let mut plan = preview.plan.as_ref().unwrap().clone();
    plan.trajectory_progress = playback.progress.clamp(0., 1.);
    let mut live: Vec<_> = cameras.iter().collect();
    live.sort_by_key(|(i, ..)| i.0);
    let planned = std::mem::take(&mut plan.cameras);
    for (index, tf, projection, angular) in live {
        if let Some((lens, offset)) = crate::camera::pinhole::parameters(projection) {
            plan.cameras.push(PlanCamera {
                label: format!("C{}", index.0),
                world_from_view: tf.to_matrix().to_cols_array_2d(),
                fov_y: lens.fov,
                aspect: lens.aspect_ratio,
                principal_point_offset: Some(offset),
                calibration: angular.map(|l| l.calibration.clone()),
                path: planned
                    .get(index.0)
                    .map_or_else(Vec::new, |c| c.path.clone()),
            });
        }
    }
    plan.poses = humans
        .iter()
        .map(|(id, p)| Pose {
            label: format!("P{}", id.id),
            joints: p.bone_positions.iter().map(|v| v.to_array()).collect(),
            parents: HUMAN_BONE_PARENTS.to_vec(),
        })
        .collect();
    plan.poses.sort_by(|a, b| a.label.cmp(&b.label));
    let current = (
        plan.cameras.clone(),
        plan.poses.clone(),
        predictions.0.clone(),
    );
    if preview.last.as_ref() == Some(&current) {
        return;
    }
    match drawing::moving(&plan, &options, &predictions.0) {
        Ok(drawing) => {
            labels(
                &mut commands,
                root,
                &mut preview.dynamic_labels,
                &drawing.labels,
                size,
                3.,
                font,
            );
            if let Some(mut mesh) = meshes.get_mut(&preview.meshes[1]) {
                *mesh = drawing.into_mesh();
            }
            preview.last = Some(current);
        }
        Err(e) => warn!("schematic: {e}"),
    }
}
