//! Match interactive RGB source density to the physical grid tile, including
//! HiDPI and SSAA. Never resize calibrated outputs, numeric annotations or a
//! dataset camera. The full source density is restored before any capture.
use super::{LensCamera, LensFace};
use bevy::{camera::RenderTarget, prelude::*, render::render_resource::Extent3d};

pub(super) fn install(app: &mut App) {
    // Read the previous completed UI layout, before camera/frustum extraction.
    // Run after Update can attach a copier or enable sampling: the same frame
    // must restore capture density. Do not depend on the later UI layout.
    app.add_systems(
        PostUpdate,
        resize
            .after(crate::sample::camera_activity::CaptureCameraGate)
            .before(bevy::camera::CameraUpdateSystems),
    );
}

fn source_side(lens: &LensCamera, tile: Option<Vec2>, configured_ssaa: u32) -> u32 {
    let Some(tile) = tile.filter(|s| s.is_finite() && s.min_element() >= 1.) else {
        return lens.native_side;
    };
    let factor = crate::render::ssaa::effective_factor(tile.ceil().as_uvec2(), configured_ssaa);
    let output = Vec2::from_array(lens.calibration.image_size.map(|v| v as f32));
    // The greater axis preserves density even if the UI stretches the image.
    let scale =
        ((tile / output).max_element() * factor as f32 / lens.native_factor as f32).clamp(0., 1.);
    // Round upwards, never below the displayed sample density. Quantization
    // also avoids texture churn for single-pixel changes in panel layout.
    ((lens.native_side as f32 * scale).ceil() as u32)
        .max(32)
        .div_ceil(16)
        .saturating_mul(16)
        .min(lens.native_side)
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn resize(
    config: Res<crate::app::BevyZeroverseConfig>,
    sampler: Option<Res<crate::sample::SamplerState>>,
    grids: Query<(&crate::app::CameraGridSource, &ComputedNode)>,
    copiers: Query<(), With<crate::io::image_copy::ImageCopier>>,
    mut lenses: Query<(Entity, &mut LensCamera)>,
    mut faces: Query<(&LensFace, &RenderTarget, &mut Projection)>,
    mut images: ResMut<Assets<Image>>,
) {
    let interactive = config.editor
        && config.camera_grid
        && !config.room_schematic
        && !config.headless
        && !config.image_copiers
        && copiers.is_empty()
        && !sampler.is_some_and(|s| s.enabled);
    for (entity, mut lens) in &mut lenses {
        let tile = interactive
            .then(|| {
                grids
                    .iter()
                    .find_map(|(source, node)| (source.0 == entity).then_some(node.size()))
            })
            .flatten();
        let side = source_side(&lens, tile, config.ssaa_factor);
        let Some(front) = images.get(&lens.faces[0]) else {
            continue;
        };
        if front.width() == side {
            continue;
        }
        let regions = super::raster::sensor_regions(side, lens.face_scale, &lens.calibration);
        for (i, handle) in lens.faces.iter().enumerate() {
            if i > 0 && *handle == lens.faces[0] {
                continue;
            }
            if let Some(mut image) = images.get_mut(handle) {
                image.resize(Extent3d {
                    width: regions[i].width(),
                    height: regions[i].height(),
                    depth_or_array_layers: 1,
                });
            }
        }
        for (face, target, mut projection) in &mut faces {
            if face.owner != entity {
                continue;
            }
            let RenderTarget::Image(target) = target else {
                continue;
            };
            let Some(index) = lens.faces.iter().position(|h| *h == target.handle) else {
                continue;
            };
            let Projection::Custom(p) = &mut *projection else {
                continue;
            };
            if let Some(p) = p.get_mut::<super::raster::FaceProjection>() {
                p.region = regions[index];
                p.full_side = side;
            }
        }
        lens.regions = regions;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_resize_is_stable_and_restores_full_sources_for_capture() {
        let mut app = App::new();
        app.insert_resource(crate::app::BevyZeroverseConfig {
            camera_grid: true,
            ..default()
        });
        app.init_resource::<Assets<Image>>();
        let mut lens = LensCamera::test_camera();
        lens.native_side = 1480;
        lens.face_scale = 0.96;
        lens.calibration = super::super::LensSampling {
            probability: 1.,
            fov_degrees: [180., 180.],
            ..default()
        }
        .sample(0, 0)
        .unwrap()
        .calibration([1280, 720], [0., 0.])
        .unwrap();
        let image = |size: UVec2| {
            let mut image = Image {
                data: None,
                ..default()
            };
            image.resize(Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            });
            image
        };
        {
            let mut assets = app.world_mut().resource_mut::<Assets<Image>>();
            lens.output = assets.add(image(UVec2::new(1280, 720)));
            lens.faces = std::array::from_fn(|_| assets.add(image(UVec2::splat(1480))));
            lens.faces[5] = lens.faces[0].clone();
        }
        let original = lens.clone();
        let owner = app.world_mut().spawn(lens).id();
        let node = app
            .world_mut()
            .spawn((
                crate::app::CameraGridSource(owner),
                ComputedNode {
                    size: Vec2::new(450., 253.),
                    ..default()
                },
            ))
            .id();
        let original_projection = super::super::raster::FaceProjection {
            perspective: PerspectiveProjection {
                fov: 1.6,
                ..default()
            },
            full_side: 1480,
            region: URect::from_corners(UVec2::ZERO, UVec2::splat(1480)),
        };
        let face = app
            .world_mut()
            .spawn((
                LensFace { owner },
                RenderTarget::Image(original.faces[1].clone().into()),
                Projection::custom(original_projection.clone()),
            ))
            .id();
        install(&mut app);
        app.update();
        let side = |app: &App| {
            app.world()
                .resource::<Assets<Image>>()
                .get(&original.faces[0])
                .unwrap()
                .width()
        };
        assert_eq!(side(&app), 1056);
        let tick = app
            .world()
            .entity(owner)
            .get_ref::<LensCamera>()
            .unwrap()
            .last_changed();
        app.update();
        assert_eq!(
            app.world()
                .entity(owner)
                .get_ref::<LensCamera>()
                .unwrap()
                .last_changed(),
            tick
        );
        use bevy::camera::CameraProjection;
        let Projection::Custom(projection) = app.world().get::<Projection>(face).unwrap() else {
            panic!()
        };
        let projection = projection
            .get::<super::super::raster::FaceProjection>()
            .unwrap();
        assert_eq!(projection.full_side, 1056);
        assert_eq!(
            projection.get_frustum_corners(0.1, 20.),
            original_projection.get_frustum_corners(0.1, 20.)
        );
        for headless in [true, false] {
            app.world_mut()
                .resource_mut::<crate::app::BevyZeroverseConfig>()
                .headless = headless;
            app.update();
            assert_eq!(side(&app), if headless { 1480 } else { 1056 });
            let lens = app.world().get::<LensCamera>(owner).unwrap();
            assert_eq!(lens.output, original.output);
            assert_eq!(lens.calibration, original.calibration);
            assert_eq!(
                app.world()
                    .resource::<Assets<Image>>()
                    .get(&lens.output)
                    .unwrap()
                    .size(),
                UVec2::new(1280, 720)
            );
        }
        app.insert_resource(crate::sample::SamplerState::default());
        app.update();
        assert_eq!(side(&app), 1480);
        app.world_mut()
            .remove_resource::<crate::sample::SamplerState>();
        app.world_mut().get_mut::<ComputedNode>(node).unwrap().size = Vec2::new(900., 506.);
        app.update();
        assert_eq!(side(&app), 1480);
        app.world_mut().get_mut::<ComputedNode>(node).unwrap().size = Vec2::new(450., 253.);
        app.update();
        assert_eq!(side(&app), 1056);
        // A sampler attached by Update must get full density on that very
        // frame, not one frame after its first render/copy request.
        app.add_systems(
            Update,
            (|mut commands: Commands| {
                commands.insert_resource(crate::sample::SamplerState::default());
            })
            .run_if(bevy::ecs::schedule::common_conditions::run_once),
        );
        app.update();
        assert_eq!(side(&app), 1480);
    }

    #[test]
    fn physical_tile_density_retains_ssaa_and_never_changes_calibration() {
        let mut lens = LensCamera::test_camera();
        lens.native_side = 1480;
        lens.calibration.image_size = [1280, 720];
        let original = lens.calibration.clone();
        assert_eq!(source_side(&lens, None, 0), 1480);
        // 450 x 253 physical pixels receives 2x automatic antialiasing.
        let side = source_side(&lens, Some(Vec2::new(450., 253.)), 0);
        assert_eq!(side, 1056);
        assert!(side as f32 >= 1480. * 450. / 1280. * 2.);
        assert_eq!(source_side(&lens, Some(Vec2::new(900., 506.)), 0), 1480);
        assert_eq!(source_side(&lens, Some(Vec2::splat(0.)), 0), 1480);
        assert_eq!(source_side(&lens, Some(Vec2::splat(f32::NAN)), 0), 1480);
        assert_eq!(source_side(&lens, Some(Vec2::new(1280., 720.)), 4), 1480);
        assert_eq!(lens.calibration, original);
        lens.native_factor = 2;
        assert_eq!(source_side(&lens, Some(Vec2::new(640., 360.)), 2), 752);
    }
}
