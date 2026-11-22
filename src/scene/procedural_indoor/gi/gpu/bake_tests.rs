//! Exercise the production upload, compute pass and irradiance texture.
use super::*;
use bevy::render::{
    gpu_readback::{Readback, ReadbackComplete},
    pipelined_rendering::PipelinedRenderingPlugin,
    RenderPlugin,
};
use std::{sync::OnceLock, time::Duration};

fn bake_lobes(app: &mut App, scene: &BakeScene) -> Vec<Vec3> {
    let settings = BakeSettings {
        spacing: 4.,
        rays_per_probe: 512,
        diffuse_bounces: 1,
        ..default()
    };
    let (request, _, stats) = prepare(
        scene,
        settings,
        17,
        app.world_mut().resource_mut::<Assets<Image>>().as_mut(),
    );
    assert_eq!(stats.probes, 8);
    assert_eq!(stats.transport_bytes as u64, request.gpu_buffer_bytes());
    assert_eq!(request.input.params.budget[3] as usize, scene.lights.len());
    let ready = request.readiness.clone();
    let image = request.image.clone();
    let extent = app
        .world()
        .resource::<Assets<Image>>()
        .get(&image)
        .unwrap()
        .texture_descriptor
        .size;
    app.insert_resource(request);
    let deadline = Instant::now() + Duration::from_secs(30);
    while !ready.ready() {
        app.update();
        assert_eq!(ready.failure(), None);
        assert!(Instant::now() < deadline, "GI bake timed out");
    }

    let bytes = Arc::new(OnceLock::new());
    let output = Arc::clone(&bytes);
    let entity = app
        .world_mut()
        .spawn(Readback::texture(image))
        .observe(move |event: On<ReadbackComplete>| {
            let _ = output.set(event.data.clone());
        })
        .id();
    let bytes = loop {
        app.update();
        if let Some(bytes) = bytes.get() {
            break bytes;
        }
        assert!(Instant::now() < deadline, "GI texture readback timed out");
        std::thread::sleep(Duration::from_millis(1));
    };
    app.world_mut().despawn(entity);
    let row_bytes = extent.width as usize * 8;
    let pitch = RenderDevice::align_copy_bytes_per_row(row_bytes);
    assert_eq!(
        bytes.len(),
        pitch * extent.height as usize * extent.depth_or_array_layers as usize
    );
    let mut lobes = Vec::new();
    for row in bytes.chunks_exact(pitch) {
        for pixel in row[..row_bytes].as_chunks::<8>().0 {
            let rgba = std::array::from_fn::<_, 4, _>(|i| {
                half::f16::from_le_bytes([pixel[2 * i], pixel[2 * i + 1]]).to_f32()
            });
            assert_eq!(
                rgba[3], 1.,
                "every lobe must have been written by the compute shader"
            );
            let rgb = Vec3::new(rgba[0], rgba[1], rgba[2]);
            assert!(rgb.is_finite() && rgb.min_element() >= 0.);
            lobes.push(rgb);
        }
    }
    assert_eq!(lobes.len(), 6 * stats.probes);
    lobes
}

#[test]
#[ignore = "requires a native graphics adapter; validates GI dispatch and RGBA16F readback"]
fn gpu_bakes_sky_unlit_and_locally_lit_geometry() {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(RenderPlugin {
                synchronous_pipeline_compilation: true,
                ..default()
            })
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: bevy::window::ExitCondition::DontExit,
                ..default()
            })
            .disable::<bevy::winit::WinitPlugin>()
            .disable::<PipelinedRenderingPlugin>(),
    )
    .init_resource::<crate::scene::procedural_indoor::preparation::residency::FutureAssets>()
    .add_plugins(GpuGiPlugin);
    app.finish();
    app.cleanup();

    let mut scene = super::super::tests::empty();
    // One probe is exactly at (0,0,0), inside the empty BVH root bounds.
    scene.bounds_max = Vec3::splat(3.);
    let sky = bake_lobes(&mut app, &scene);
    // 2x2x2 probes. The texture packs the six lobes as +X/-X, +Y/-Y, +Z/-Z.
    for (i, rgb) in sky.iter().enumerate() {
        let face = i / 16 * 2 + i / 4 % 2;
        let expected = [0.5, 0.5, 1., 0., 0.5, 0.5][face];
        assert!((*rgb - Vec3::splat(expected)).abs().max_element() < 0.015);
    }

    let mut walls = Assembly::default();
    walls.box_part(Surface::Paint, "room", Vec3::ZERO, Vec3::splat(8.), 0.);
    for geometry in walls.parts.values() {
        scene.add_geometry(geometry, Transform::IDENTITY, 0);
    }
    scene.materials.push(DiffuseMaterial {
        albedo: Vec3::ONE,
        emission: Vec3::ZERO,
        uv_scale: Vec2::ONE,
        texture: None,
        address: Default::default(),
        textured_emission: false,
    });
    drop(scene.build_bvh());
    let dark = bake_lobes(&mut app, &scene);
    assert!(dark.iter().all(|&lobe| lobe == Vec3::ZERO));

    scene.lights.push(LocalLight {
        direction: Vec3::NEG_Y,
        position: Vec3::ZERO,
        color: Vec3::X,
        candela: 50.,
        range: 20.,
        spot: false,
        inner_cos: 1.,
        outer_cos: 0.,
    });
    let lit = bake_lobes(&mut app, &scene);
    assert!(lit
        .iter()
        .all(|lobe| lobe.x > 0.005 && lobe.y == 0. && lobe.z == 0.));

    // Same texture and UVs, two actual hardware address contracts. Every wall
    // is emissive so the read-back six lobes directly identify the sampled texel.
    let mut indexed = super::super::tests::empty();
    indexed.bounds_max = Vec3::splat(3.);
    let uv = Vec2::new(-0.25, 1.25);
    for mut geometry in walls.parts.into_values() {
        geometry.uvs.fill(uv.to_array());
        indexed.add_geometry(&geometry, Transform::IDENTITY, 0);
    }
    drop(indexed.build_bvh());
    indexed.materials.push(DiffuseMaterial {
        albedo: Vec3::ONE,
        emission: Vec3::ONE,
        uv_scale: Vec2::ONE,
        texture: Some((2, vec![Vec3::X, Vec3::Y, Vec3::Z, Vec3::ONE])),
        address: Default::default(),
        textured_emission: true,
    });
    use bevy::image::{ImageAddressMode, ImageSamplerDescriptor};
    for (u, v, expected) in [
        (ImageAddressMode::Repeat, ImageAddressMode::Repeat, Vec3::Y),
        (
            ImageAddressMode::ClampToEdge,
            ImageAddressMode::ClampToEdge,
            Vec3::Z,
        ),
        (
            ImageAddressMode::MirrorRepeat,
            ImageAddressMode::Repeat,
            Vec3::X,
        ),
        (
            ImageAddressMode::ClampToBorder,
            ImageAddressMode::Repeat,
            Vec3::ZERO,
        ),
    ] {
        indexed.materials[0].address = TextureAddress::from_descriptor(&ImageSamplerDescriptor {
            address_mode_u: u,
            address_mode_v: v,
            ..default()
        });
        assert_eq!(indexed.materials[0].albedo(uv), expected);
        for lobe in bake_lobes(&mut app, &indexed) {
            assert!(
                (lobe - expected).abs().max_element() < 0.02,
                "{u:?}/{v:?}: {lobe:?}"
            );
        }
    }
}
