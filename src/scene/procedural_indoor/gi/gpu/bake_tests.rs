//! Exercise the production upload, compute pass and irradiance texture.
use super::*;
use bevy::render::{
    gpu_readback::{Readback, ReadbackComplete},
    pipelined_rendering::PipelinedRenderingPlugin,
    RenderPlugin,
};
use std::{sync::OnceLock, time::Duration};

fn bake_lobes(app: &mut App, scene: &BakeScene, rays_per_probe: u32) -> Vec<Vec3> {
    let settings = BakeSettings {
        spacing: 4.,
        rays_per_probe,
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
    let sky = bake_lobes(&mut app, &scene, 512);
    // 2x2x2 probes. The texture packs the six lobes as +X/-X, +Y/-Y, +Z/-Z.
    for (i, rgb) in sky.iter().enumerate() {
        let face = i / 16 * 2 + i / 4 % 2;
        let expected = [0.5, 0.5, 1., 0., 0.5, 0.5][face];
        assert!((*rgb - Vec3::splat(expected)).abs().max_element() < 0.015);
    }

    let room = IndoorManifest::generate_with_humans(
        73161,
        crate::scene::procedural_indoor::layout::IndoorLayout::Cafe,
        0.65,
        0,
        0.,
    )
    .unwrap();
    scene.sky_field = crate::scene::procedural_indoor::sky::bake(&room)
        .unwrap()
        .field;
    // The CPU and GPU rotate their Fibonacci rays with different PRNGs.
    // Integrate to convergence before comparing the uploaded directional map.
    let rays_per_probe = 16_384;
    let actual = bake_lobes(&mut app, &scene, rays_per_probe);
    let cpu = scene
        .bake(
            BakeSettings {
                spacing: 4.,
                rays_per_probe,
                diffuse_bounces: 1,
                ..default()
            },
            17,
        )
        .image();
    let expected = cpu.data.unwrap();
    assert_eq!(expected.len(), actual.len() * 8);
    for (rgb, pixel) in actual.into_iter().zip(expected.as_chunks::<8>().0) {
        for (i, v) in rgb.to_array().into_iter().enumerate() {
            let reference = half::f16::from_le_bytes([pixel[2 * i], pixel[2 * i + 1]]).to_f32();
            assert!(
                (v - reference).abs() <= 0.003 * reference.max(1.),
                "directional sky GPU {v}, CPU {reference}"
            );
        }
    }
    // The same anisotropic sky reaches every probe in empty space. At a low
    // budget their estimates differ. Resolve must reduce their spatial variance
    // without transferring light between lobes or changing the total energy.
    let bounds = (scene.bounds_min, scene.bounds_max);
    scene.bounds_min = Vec3::new(-0.7, 0.4, -0.7);
    scene.bounds_max = Vec3::new(0.7, 2.2, 0.7);
    let raw = bake_lobes(&mut app, &scene, 32);
    let hall = IndoorManifest::generate_with_humans(
        58103,
        crate::scene::procedural_indoor::layout::IndoorLayout::Gymnasium,
        0.65,
        0,
        0.,
    )
    .unwrap();
    scene.probe_domain = ProbeDomain::from_manifest(&hall);
    let filtered = bake_lobes(&mut app, &scene, 32);
    let face = |i: usize| i / 16 * 2 + i / 4 % 2;
    let mut total_spread = [0.; 2];
    for f in 0..6 {
        let mut means = [Vec3::ZERO; 2];
        for (k, source) in [&raw, &filtered].into_iter().enumerate() {
            let cells = || {
                source
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| face(*i) == f)
                    .map(|(_, v)| *v)
            };
            means[k] = cells().sum::<Vec3>() / 8.;
            total_spread[k] += cells().map(|v| v.distance_squared(means[k])).sum::<f32>();
        }
        // The texture stores half floats after the conservative exchange.
        assert!(means[0].abs_diff_eq(means[1], 0.001 * means[0].max_element().max(1.)));
    }
    assert!(
        total_spread[1] < total_spread[0] * 0.9,
        "raw/filtered sky variance {total_spread:?}"
    );
    (scene.bounds_min, scene.bounds_max) = bounds;
    scene.probe_domain = None;
    scene.sky_field = default();

    // A downward-facing receiver sees only light reflected off the floor.
    // It must receive that light even with a one-surface transport budget.
    let mut floor = super::super::tests::daylight_floor();
    floor.bounds_max = Vec3::splat(3.);
    let reflected = floor.sky * floor.materials[0].albedo;
    for (i, rgb) in bake_lobes(&mut app, &floor, 1024).into_iter().enumerate() {
        let face = i / 16 * 2 + i / 4 % 2;
        let expected = match face {
            2 => floor.sky,
            3 => reflected,
            _ => 0.5 * (floor.sky + reflected),
        };
        assert!(
            rgb.abs_diff_eq(expected, 0.015),
            "lobe {face}: {rgb} vs {expected}"
        );
    }

    let mut walls = Assembly::default();
    walls.box_part(Surface::Paint, "room", Vec3::ZERO, Vec3::splat(8.), 0.);
    for geometry in walls.parts.values() {
        scene.add_geometry(geometry, Transform::IDENTITY, 0);
    }
    scene.materials.push(DiffuseMaterial {
        albedo: Vec3::ONE,
        emission: Vec3::ZERO,
        uv_transform: Affine2::IDENTITY,
        base_map: None,
        emission_map: EmissionMap::Uniform,
    });
    drop(scene.build_bvh());
    let dark = bake_lobes(&mut app, &scene, 512);
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
    let lit = bake_lobes(&mut app, &scene, 512);
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
        uv_transform: Affine2::IDENTITY,
        base_map: Some(TransportMap {
            size: 2,
            pixels: [Vec3::X, Vec3::Y, Vec3::Z, Vec3::ONE].into(),
            address: default(),
        }),
        emission_map: EmissionMap::SharedWithBase,
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
        indexed.materials[0].base_map.as_mut().unwrap().address =
            TextureAddress::from_descriptor(&ImageSamplerDescriptor {
                address_mode_u: u,
                address_mode_v: v,
                ..default()
            });
        assert_eq!(indexed.materials[0].response(uv).0, expected);
        for lobe in bake_lobes(&mut app, &indexed, 512) {
            assert!(
                (lobe - expected).abs().max_element() < 0.02,
                "{u:?}/{v:?}: {lobe:?}"
            );
        }
    }
    // The off-diagonal basis and phase map (-0.25, 1.25) to (0.25, 0.25)
    // before the border test. Only the red texel can light this enclosure.
    indexed.materials[0].uv_transform = Affine2::from_cols(
        Vec2::new(0., 1.25),
        Vec2::new(-0.5, 0.),
        Vec2::new(0.875, 0.5625),
    );
    assert_eq!(indexed.materials[0].response(uv).0, Vec3::X);
    for lobe in bake_lobes(&mut app, &indexed, 512) {
        assert!(lobe.abs_diff_eq(Vec3::X, 0.02), "affine atlas lobe {lobe}");
    }
    // A dark cover reflects very little, but its emissive pixels keep their
    // radiance. A second map may have its own sampler, or no base map at all.
    indexed.materials[0].uv_transform = Affine2::IDENTITY;
    indexed.materials[0].albedo = Vec3::ZERO;
    indexed.materials[0].emission = Vec3::new(2., 3., 4.);
    indexed.materials[0].base_map.as_mut().unwrap().address =
        TextureAddress::from_descriptor(&ImageSamplerDescriptor {
            address_mode_u: ImageAddressMode::ClampToEdge,
            address_mode_v: ImageAddressMode::ClampToEdge,
            ..default()
        });
    let mut separate = indexed.materials[0].base_map.clone().unwrap();
    separate.address = TextureAddress::default();
    for (mode, (map, expected)) in [
        (EmissionMap::SharedWithBase, 4. * Vec3::Z),
        (EmissionMap::Separate(separate.clone()), 3. * Vec3::Y),
        (EmissionMap::Separate(separate), 3. * Vec3::Y),
    ]
    .into_iter()
    .enumerate()
    {
        if mode == 2 {
            indexed.materials[0].base_map = None;
        }
        indexed.materials[0].emission_map = map;
        assert_eq!(indexed.materials[0].response(uv), (Vec3::ZERO, expected));
        for lobe in bake_lobes(&mut app, &indexed, 512) {
            assert!(
                lobe.abs_diff_eq(expected, 0.025),
                "map {mode}: {lobe} vs {expected}"
            );
        }
    }
}
