//! Optional schematic viewport. Updates at most 8 Hz; caches unchanged diagrams.
use super::*;
use crate::annotation::schematic::{Camera as PlanCamera, Overlay, Pose, RenderOptions, Schematic};
use crate::scene::procedural_indoor::{
    humans::{IndoorHumanInstance, HUMAN_BONE_PARENTS},
    layout::IndoorManifest,
};
#[derive(Component)]
pub(super) struct SchematicView;
/// Applications can insert model predictions here to compare them in the viewer.
#[derive(Resource, Default)]
pub struct Predictions(pub Overlay);
#[derive(Resource, Default)]
pub(super) struct Preview {
    elapsed: f32,
    entity: Option<Entity>,
    texture: Option<Handle<Image>>,
    last: Option<(Schematic, Overlay, RenderOptions)>,
}
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn update(
    mut commands: Commands,
    config: Res<BevyZeroverseConfig>,
    time: Res<Time>,
    editor: Res<EditorState>,
    playback: Res<Playback>,
    scene: Option<Res<IndoorManifest>>,
    predictions: Res<Predictions>,
    mut preview: ResMut<Preview>,
    cameras: Query<(
        &crate::camera::CaptureCameraIndex,
        &GlobalTransform,
        &Projection,
    )>,
    roots: Query<&GlobalTransform, With<crate::scene::ZeroverseSceneRoot>>,
    humans: Query<(&IndoorHumanInstance, &crate::annotation::pose::HumanPose)>,
    mut images: ResMut<Assets<Image>>,
    assets: Res<AssetServer>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
) {
    if !config.room_schematic || config.headless {
        if let Some(e) = preview.entity.take() {
            commands.entity(e).despawn();
        }
        if let Some(handle) = preview.texture.take() {
            images.remove(handle.id());
        }
        preview.last = None;
        return;
    }
    preview.elapsed += time.delta_secs();
    if preview.entity.is_some() && preview.elapsed < 0.125 {
        return;
    }
    preview.elapsed = 0.;
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
    let Some(scene) = scene.filter(|_| config.scene_type == ZeroverseSceneType::ProceduralIndoor)
    else {
        if let Some(handle) = preview.texture.take() {
            images.remove(handle.id());
            if let Some(e) = preview.entity.take() {
                commands.entity(e).despawn();
            }
            preview.last = None;
        }
        if preview.entity.is_none() {
            preview.entity=Some(commands.spawn((SchematicView,Node{align_items:AlignItems::Center,justify_content:JustifyContent::Center,..default()},BackgroundColor(Color::srgb(0.06,0.09,0.10)),ZIndex(1))).with_children(|p|{p.spawn((Text::new("Room schematic requires a Procedural Indoor scene.\nChoose Scene → Procedural Indoor, then Apply."),TextColor(INK),TextFont{font:assets.load(bevy::feathers::constants::fonts::REGULAR).into(),font_size:18.0.into(),..default()}));}).id());
        }
        return;
    };
    let Ok(mut plan) = Schematic::from_manifest(&scene, playback.progress.clamp(0., 1.)) else {
        return;
    };
    if let Ok(root) = roots.single() {
        plan.transform_geometry(root.to_matrix() * Mat4::from_rotation_y(-scene.world_yaw));
    }
    let mut live: Vec<_> = cameras.iter().collect();
    live.sort_by_key(|(i, _, _)| i.0);
    let planned = std::mem::take(&mut plan.cameras);
    for (index, tf, projection) in live {
        if let Projection::Perspective(lens) = projection {
            plan.cameras.push(PlanCamera {
                label: format!("C{}", index.0),
                world_from_view: tf.to_matrix().to_cols_array_2d(),
                fov_y: lens.fov,
                aspect: lens.aspect_ratio,
                calibration: None,
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
    let current = (plan, predictions.0.clone(), options);
    if preview.last.as_ref() == Some(&current) {
        return;
    }
    let pixels = match current.0.rgba(&options, &predictions.0) {
        Ok(p) => p,
        Err(e) => {
            warn!("schematic: {e}");
            return;
        }
    };
    let image = Image::new(
        bevy::render::render_resource::Extent3d {
            width: options.width,
            height: options.height,
            depth_or_array_layers: 1,
        },
        bevy::render::render_resource::TextureDimension::D2,
        pixels,
        bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
        bevy::asset::RenderAssetUsages::default(),
    );
    if let Some(handle) = &preview.texture {
        if let Some(mut existing) = images.get_mut(handle) {
            *existing = image;
        }
    } else {
        if let Some(e) = preview.entity.take() {
            commands.entity(e).despawn();
        }
        let handle = images.add(image);
        preview.entity = Some(
            commands
                .spawn((
                    SchematicView,
                    Node { ..default() },
                    ImageNode {
                        image: handle.clone(),
                        ..default()
                    },
                    ZIndex(1),
                ))
                .id(),
        );
        preview.texture = Some(handle);
    }
    preview.last = Some(current);
}
