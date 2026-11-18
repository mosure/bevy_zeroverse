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

#[derive(bevy::ecs::query::QueryData)]
struct GridCamera {
    entity: Entity,
    target: &'static RenderTarget,
    index: Option<&'static crate::camera::CaptureCameraIndex>,
    marked: Has<CameraGridMarker>,
}

fn setup_camera_grid(
    mut commands: Commands,
    args: Res<BevyZeroverseConfig>,
    mut materials: ResMut<Assets<GridImage>>,
    camera_grids: Query<Entity, With<CameraGrid>>,
    zeroverse_cameras: Query<GridCamera, With<ZeroverseCamera>>,
    mut scene_loaded: MessageReader<SceneLoadedEvent>,
    mut previous_camera_grid: Local<Option<(bool, usize)>>,
) {
    if zeroverse_cameras.is_empty() {
        for grid in &camera_grids {
            commands.entity(grid).despawn();
        }
        *previous_camera_grid = None;
        return;
    }

    let state = (
        args.camera_grid && !args.room_schematic,
        zeroverse_cameras.iter().len(),
    );
    if scene_loaded.is_empty()
        && *previous_camera_grid == Some(state)
        && zeroverse_cameras.iter().all(|camera| camera.marked)
    {
        return;
    }
    scene_loaded.clear();
    *previous_camera_grid = Some(state);

    for camera in &zeroverse_cameras {
        if !camera.marked {
            commands.entity(camera.entity).insert(CameraGridMarker);
        }
    }

    for grid in &camera_grids {
        commands.entity(grid).despawn();
    }

    if state.0 {
        let mut cameras: Vec<_> = zeroverse_cameras.iter().collect();
        cameras.sort_by_key(|camera| {
            (
                camera.index.map_or(usize::MAX, |i| i.0),
                camera.entity.to_bits(),
            )
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
                for camera in cameras {
                    let RenderTarget::Image(texture) = camera.target else {
                        continue;
                    };

                    builder.spawn((
                        CameraGridSource(camera.entity),
                        MaterialNode(materials.add(GridImage {
                            image: texture.handle.clone(),
                            filter_rgb: 1,
                        })),
                        // Keep Bevy's intrinsic image sizing and grid aspect.
                        // The material draws the tile; a transparent ImageNode is never extracted.
                        ImageNode {
                            color: Color::NONE,
                            image: texture.handle.clone(),
                            ..default()
                        },
                    ));
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_follow_live_capture_targets_including_the_last_removal() {
        let mut app = App::new();
        app.init_resource::<Assets<GridImage>>()
            .add_message::<SceneLoadedEvent>()
            .insert_resource(BevyZeroverseConfig {
                camera_grid: true,
                ..default()
            })
            .add_systems(Update, setup_camera_grid);
        let entities = [0, 1].map(|index| {
            app.world_mut()
                .spawn((
                    ZeroverseCamera::default(),
                    RenderTarget::Image(Handle::<Image>::default().into()),
                    crate::camera::CaptureCameraIndex(index),
                ))
                .id()
        });
        for count in [2, 1, 0] {
            app.update();
            let world = app.world_mut();
            assert_eq!(
                world.query::<&CameraGridSource>().iter(world).count(),
                count
            );
            assert_eq!(
                world.query::<&CameraGrid>().iter(world).count(),
                usize::from(count > 0)
            );
            if count > 0 {
                world.despawn(entities[count - 1]);
            }
        }
    }
}
