//! A foreground mullion cannot be a source of transmitted light. Compare
//! identical camera paths on either side of that single visibility change.
use bevy::{core_pipeline::prepass::DepthPrepass, light::NotShadowCaster, prelude::*};
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::{CaptureCameraIndex, PerspectiveSampler, TrajectorySampler, ZeroverseCamera},
    headless::{create_app, setup_globals},
    render::{color::RgbTonemapping, semantic::SemanticLabel, RenderMode},
    scene::{SceneAabbNode, ZeroverseSceneRoot, ZeroverseSceneType},
};

#[test]
#[ignore = "native moving-camera transmission readback; runs on software Vulkan"]
fn foreground_mullion_does_not_flicker_through_frosted_glass() {
    const SIDE: usize = 96;
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
            num_cameras: 1,
            playback_steps: 3,
            playback_step: 0.5,
            width: SIDE as f32,
            height: SIDE as f32,
            tonemapping: RgbTonemapping::None,
            ssaa_factor: 1,
            render_modes: vec![RenderMode::Color],
            ..default()
        }),
        false,
    );
    let world = app.world_mut();
    let root = world
        .spawn((
            ZeroverseSceneRoot,
            SceneAabbNode,
            Transform::IDENTITY,
            Visibility::default(),
        ))
        .id();
    let mut slab = |size: Vec3, position: Vec3, material, label| {
        let mesh = world
            .resource_mut::<Assets<Mesh>>()
            .add(Cuboid::from_size(size));
        let material = world
            .resource_mut::<Assets<StandardMaterial>>()
            .add(material);
        world
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material),
                Transform::from_translation(position),
                label,
                NotShadowCaster,
                ChildOf(root),
            ))
            .id()
    };
    let unlit = |gray| StandardMaterial {
        base_color: Color::linear_rgb(gray, gray, gray),
        unlit: true,
        ..default()
    };
    slab(
        Vec3::new(12., 8., 0.04),
        Vec3::Z * -3.,
        unlit(0.7),
        SemanticLabel::Wall,
    );
    slab(
        Vec3::new(4., 8., 0.04),
        Vec3::new(3.15, 0., -2.9),
        unlit(0.),
        SemanticLabel::Wall,
    );
    slab(
        Vec3::new(6., 4., 0.01),
        Vec3::ZERO,
        StandardMaterial {
            specular_transmission: 1.,
            perceptual_roughness: 0.47,
            ior: 1.5,
            thickness: 0.009,
            ..default()
        },
        SemanticLabel::Window,
    );
    let mullion = slab(
        Vec3::new(0.12, 3., 0.02),
        Vec3::Z * 0.35,
        unlit(0.),
        SemanticLabel::Wall,
    );
    world.entity_mut(mullion).insert(Visibility::Hidden);
    world.spawn((
        CaptureCameraIndex(0),
        ZeroverseCamera {
            perspective_sampler: PerspectiveSampler::exact(52.),
            trajectory: TrajectorySampler::WaypointPath {
                positions: vec![Vec3::new(-0.025, 0., 1.55), Vec3::new(0.025, 0., 1.55)],
                rotations: vec![Quat::IDENTITY; 2],
            },
            ..default()
        },
        DepthPrepass,
        ChildOf(root),
    ));
    // Bloom deliberately spreads dark/light boundaries across pixels. Isolate
    // the optical filter from presentation effects for this radiance oracle.
    app.add_systems(
        Last,
        |mut commands: Commands, cameras: Query<Entity, With<Camera3d>>| {
            for entity in &cameras {
                commands
                    .entity(entity)
                    .insert(bevy::core_pipeline::tonemapping::DebandDither::Disabled)
                    .remove::<(
                        bevy::post_process::bloom::Bloom,
                        bevy::anti_alias::fxaa::Fxaa,
                    )>();
            }
        },
    );
    app.finish();
    app.cleanup();
    let reference = super::capture(&mut app);
    app.world_mut()
        .entity_mut(mullion)
        .insert(Visibility::Inherited);
    let occluded = super::capture(&mut app);
    assert_eq!(reference.views.len(), 3);
    assert_eq!(occluded.views.len(), 3);
    assert_ne!(
        reference.views[0].world_from_view,
        reference.views[2].world_from_view
    );
    for (glass, control) in occluded.views.iter().zip(&reference.views) {
        assert_eq!(glass.world_from_view, control.world_from_view);
        let image: &[[f32; 4]] = bytemuck::cast_slice(&glass.color);
        let clean: &[[f32; 4]] = bytemuck::cast_slice(&control.color);
        assert!(image
            .iter()
            .flatten()
            .chain(clean.iter().flatten())
            .all(|v| v.is_finite()));
        let middle = SIDE / 2 * SIDE;
        assert!(
            image[middle + 48][0] + 0.2 < clean[middle + 48][0],
            "mullion must be visible"
        );
        assert!(
            clean[middle + 87][0] + 0.2 < clean[middle + 33][0],
            "black behind the glass must remain dark"
        );
        // Inside the scattering radius on the bright side, outside the actual
        // silhouette at every pose. There is no physical shadow or radiance change.
        for y in 30..66 {
            for x in 35..39 {
                let i = y * SIDE + x;
                assert!(
                    (image[i][0] - clean[i][0]).abs() < 0.003,
                    "foreground leaked into transmitted light at ({x},{y}): {:?} vs {:?}",
                    image[i],
                    clean[i]
                );
            }
        }
    }
}
