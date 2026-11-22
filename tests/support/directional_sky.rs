//! Independent check of the finite sky mesh against the material's actual radiance texture.
use bevy::{camera::Exposure, prelude::*};
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::{CaptureCameraIndex, PerspectiveSampler, ZeroverseCamera},
    headless::{create_app, setup_globals},
    render::{color::RgbTonemapping, RenderMode},
    scene::{
        procedural_indoor::{
            architecture,
            layout::{IndoorLayout, IndoorManifest},
            materials::{IndoorMaterials, Surface},
        },
        SceneAabbNode, ZeroverseSceneRoot, ZeroverseSceneType,
    },
};

#[derive(Resource)]
struct SkyTexture {
    texture: Handle<Image>,
    scale: Vec3,
}

impl SkyTexture {
    fn sample(&self, image: &Image, ray: Vec3) -> Vec3 {
        let horizontal = ray.xz() / (ray.x.abs() + ray.y.max(0.) + ray.z.abs()).max(1e-8);
        let uv = (Vec2::new(horizontal.x + horizontal.y, horizontal.x - horizontal.y) + 1.) * 0.5;
        let last = image.size() - 1;
        let texel = uv.clamp(Vec2::ZERO, Vec2::ONE) * last.as_vec2();
        let lo = texel.floor().as_uvec2();
        let hi = (lo + 1).min(last);
        let weight = texel.fract();
        let at = |x, y| {
            let rgb = image.get_color_at(x, y).unwrap().to_linear();
            Vec3::new(rgb.red, rgb.green, rgb.blue)
        };
        self.scale
            * at(lo.x, lo.y)
                .lerp(at(hi.x, lo.y), weight.x)
                .lerp(at(lo.x, hi.y).lerp(at(hi.x, hi.y), weight.x), weight.y)
    }
}

#[test]
#[ignore = "requires a native graphics adapter for linear RGB readback"]
fn tagged_sky_matches_world_rays_with_room_rotation_and_camera_translation() {
    const SIDE: usize = 64;
    let scene =
        IndoorManifest::generate_with_humans(73161, IndoorLayout::Cafe, 0.65, 0, 0.).unwrap();
    let rotation = Quat::from_rotation_y(1.3);
    std::env::set_var("BEVY_ASSET_ROOT", env!("CARGO_MANIFEST_DIR"));
    setup_globals(None);
    let mut app = create_app(
        None,
        Some(BevyZeroverseConfig {
            scene_type: ZeroverseSceneType::Custom,
            initialize_scene: false,
            headless: true,
            editor: false,
            gizmos: false,
            keybinds: false,
            image_copiers: true,
            num_cameras: 3,
            playback_steps: 1,
            width: SIDE as f32,
            height: SIDE as f32,
            hdr: true,
            tonemapping: RgbTonemapping::None,
            ssaa_factor: 1,
            render_modes: vec![RenderMode::Color],
            ..default()
        }),
        false,
    );
    app.add_systems(
        Startup,
        move |mut commands: Commands,
              mut images: ResMut<Assets<Image>>,
              mut materials: ResMut<Assets<StandardMaterial>>,
              mut meshes: ResMut<Assets<Mesh>>| {
            let root = commands
                .spawn((
                    ZeroverseSceneRoot,
                    SceneAabbNode,
                    Transform::IDENTITY,
                    Visibility::default(),
                ))
                .id();
            let mut assembly = architecture::architecture(&scene);
            assembly
                .parts
                .retain(|(_, label), _| label == "sky#exterior");
            assert_eq!(assembly.parts.len(), 1);
            let set = IndoorMaterials::build_for_parts(
                &scene,
                &mut *images,
                &mut *materials,
                [(Surface::Light, "sky#exterior")],
            );
            let material = materials
                .get(&set.for_part(Surface::Light, "sky#exterior"))
                .unwrap();
            commands.insert_resource(SkyTexture {
                texture: material.emissive_texture.clone().unwrap(),
                scale: Vec3::new(
                    material.emissive.red,
                    material.emissive.green,
                    material.emissive.blue,
                ),
            });
            let shell = commands
                .spawn((
                    Transform::from_rotation(rotation),
                    Visibility::default(),
                    ChildOf(root),
                ))
                .id();
            assembly.spawn(shell, &mut commands, &mut meshes, &set);
            for (i, (origin, direction)) in [
                (Vec3::ZERO, Vec3::new(-0.6, 0.2, -1.)),
                (Vec3::new(6., 2., -4.), Vec3::new(-0.6, 0.2, -1.)),
                (Vec3::ZERO, Vec3::new(1., 0.85, -0.35)),
            ]
            .into_iter()
            .enumerate()
            {
                commands.spawn((
                    CaptureCameraIndex(i),
                    ZeroverseCamera {
                        override_transform: Some(
                            Transform::from_translation(origin)
                                .looking_to(direction.normalize(), Vec3::Y),
                        ),
                        perspective_sampler: PerspectiveSampler::exact(92.),
                        ..default()
                    },
                    ChildOf(root),
                ));
            }
        },
    );
    app.add_systems(
        Last,
        |mut commands: Commands, cameras: Query<Entity, With<Camera3d>>| {
            for entity in &cameras {
                commands
                    .entity(entity)
                    .insert((
                        Exposure::INDOOR,
                        bevy::core_pipeline::tonemapping::Tonemapping::None,
                        bevy::core_pipeline::tonemapping::DebandDither::Disabled,
                    ))
                    .remove::<(
                        bevy::post_process::bloom::Bloom,
                        bevy::anti_alias::fxaa::Fxaa,
                    )>();
            }
        },
    );
    app.finish();
    app.cleanup();
    app.insert_resource(bevy::light::GlobalAmbientLight::NONE);
    let sample = super::capture(&mut app);
    let sky = app.world().resource::<SkyTexture>();
    let img = app
        .world()
        .resource::<Assets<Image>>()
        .get(&sky.texture)
        .unwrap();
    assert_eq!(sample.views.len(), 3);
    for view in &sample.views {
        let calibration = view.calibration.as_ref().unwrap();
        let world = Mat4::from_cols_array_2d(&view.world_from_view);
        let pixels: &[[f32; 4]] = bytemuck::cast_slice(&view.color);
        for y in (3..SIDE - 3).step_by(4) {
            for x in (3..SIDE - 3).step_by(4) {
                let ray = calibration
                    .unproject([x as f32 + 0.5, y as f32 + 0.5], 1.)
                    .unwrap();
                let local = rotation.inverse() * world.transform_vector3(Vec3::from(ray));
                let reference = sky.sample(img, local) * Exposure::INDOOR.exposure();
                let measured = Vec4::from_array(pixels[y * SIDE + x]).truncate();
                let error = (measured / reference - 1.).abs().max_element();
                assert!(
                    error < 0.016,
                    "pixel ({x},{y}), RGB {measured:?}, expected {reference:?}"
                );
            }
        }
    }
    let a: &[f32] = bytemuck::cast_slice(&sample.views[0].color);
    let b: &[f32] = bytemuck::cast_slice(&sample.views[1].color);
    assert!(a
        .iter()
        .zip(b)
        .all(|(x, y)| (x - y).abs() < 0.002 * x.abs().max(1.)));
}
