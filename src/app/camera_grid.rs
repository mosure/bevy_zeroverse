//! Filter calibrated images at the actual on-screen pixel footprint. A grid
//! tile can be substantially smaller than the capture even with SSAA enabled.
use super::*;
use bevy::{
    asset::{load_internal_asset, uuid_handle},
    render::render_resource::AsBindGroup,
    shader::ShaderRef,
};

const SHADER: Handle<Shader> = uuid_handle!("c9ad960b-3d55-4e9d-9158-5e54f6505881");

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub(super) struct GridImage {
    #[texture(0, filterable = false)]
    pub image: Handle<Image>,
    #[uniform(1)]
    pub filter_rgb: u32,
}

impl UiMaterial for GridImage {
    fn fragment_shader() -> ShaderRef {
        SHADER.into()
    }
}

pub(super) fn install(app: &mut App) {
    load_internal_asset!(app, SHADER, "camera_grid.wgsl", Shader::from_wgsl);
    app.add_plugins(UiMaterialPlugin::<GridImage>::default());
    app.add_systems(PostUpdate, (setup_camera_grid, update_images).chain());
}

#[allow(clippy::type_complexity)]
pub(super) fn update_images(
    mode: Res<RenderMode>,
    cameras: Query<(
        &RenderTarget,
        Option<&crate::render::ground_truth::GroundTruthCamera>,
        Option<&crate::render::ssaa::SsaaCamera>,
        Option<&crate::camera::lens::LensCamera>,
    )>,
    mut materials: ResMut<Assets<GridImage>>,
    mut images: Query<(&CameraGridSource, &mut ImageNode, &MaterialNode<GridImage>)>,
) {
    for (source, mut image, material) in &mut images {
        let Ok((RenderTarget::Image(target), gt, ssaa, lens)) = cameras.get(source.0) else {
            continue;
        };
        let handle = if let Some(ssaa) = ssaa {
            &ssaa.output
        } else if *mode == RenderMode::CoVisibility {
            gt.and_then(|g| g.co_visibility.as_ref())
                .and_then(|c| c.preview.as_ref())
                .unwrap_or(&target.handle)
        } else if let Some(lens) = lens {
            &lens.output
        } else {
            &target.handle
        };
        let filter_rgb = u32::from(*mode == RenderMode::Color);
        if materials
            .get(material.id())
            .is_some_and(|m| m.image != *handle || m.filter_rgb != filter_rgb)
        {
            let mut m = materials.get_mut(material.id()).unwrap();
            m.image = handle.clone();
            m.filter_rgb = filter_rgb;
        }
        if image.image != *handle {
            image.image = handle.clone();
        }
    }
}

fn setup_camera_grid(
    mut commands: Commands,
    args: Res<BevyZeroverseConfig>,
    mut materials: ResMut<Assets<GridImage>>,
    camera_grids: Query<Entity, With<CameraGrid>>,
    zeroverse_cameras: Query<
        (
            Entity,
            &RenderTarget,
            Option<&crate::camera::CaptureCameraIndex>,
        ),
        With<ZeroverseCamera>,
    >,
    new_zeroverse_cameras: Query<Entity, (With<ZeroverseCamera>, Without<CameraGridMarker>)>,
    mut scene_loaded: MessageReader<SceneLoadedEvent>,
    mut previous_camera_grid: Local<Option<bool>>,
) {
    if zeroverse_cameras.is_empty() {
        return;
    }

    let camera_grid_changed = previous_camera_grid
        .map(|prev| prev != (args.camera_grid && !args.room_schematic))
        .unwrap_or(true);

    if scene_loaded.is_empty() && !camera_grid_changed && new_zeroverse_cameras.is_empty() {
        return;
    }
    scene_loaded.clear();
    *previous_camera_grid = Some(args.camera_grid && !args.room_schematic);

    for entity in new_zeroverse_cameras.iter() {
        commands.entity(entity).insert(CameraGridMarker);
    }

    if !camera_grids.is_empty() {
        commands.entity(camera_grids.single().unwrap()).despawn();
    }

    if args.camera_grid && !args.room_schematic {
        let mut cameras: Vec<_> = zeroverse_cameras.iter().collect();
        cameras.sort_by_key(|(entity, _, index)| {
            (index.map_or(usize::MAX, |i| i.0), entity.to_bits())
        });
        let camera_count = cameras.len();
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
                for (entity, target, _) in cameras {
                    let texture = match target.clone() {
                        RenderTarget::Image(texture) => texture,
                        _ => continue,
                    };

                    builder.spawn((
                        CameraGridSource(entity),
                        MaterialNode(materials.add(GridImage {
                            image: texture.handle.clone(),
                            filter_rgb: 1,
                        })),
                        // Keep Bevy's intrinsic image sizing and grid aspect.
                        // The material draws the tile; a transparent ImageNode is never extracted.
                        ImageNode {
                            color: Color::NONE,
                            image: texture.handle,
                            ..default()
                        },
                    ));
                }
            });
    }
}
