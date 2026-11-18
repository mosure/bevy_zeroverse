use super::*;
use crate::{app::BevyZeroverseConfig, headless::create_app, io::image_copy::ImageCopier};
use std::time::{Duration, Instant};

fn test_app() -> App {
    let mut app = create_app(
        None,
        Some(BevyZeroverseConfig {
            headless: true,
            editor: false,
            initialize_scene: false,
            image_copiers: true,
            num_cameras: 0,
            gizmos: false,
            ..default()
        }),
        false,
    );
    app.finish();
    app.cleanup();
    app
}

fn fixture(
    app: &mut App,
    device: &RenderDevice,
    size: UVec2,
    factor: u32,
    values: &[f32],
    profile: Option<CameraSensorProfile>,
) -> ImageCopier {
    let high = size * factor;
    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    let mut source = Image::new_target_texture(high.x, high.y, TextureFormat::Rgba32Float, None);
    source.data = Some(bytemuck::cast_slice(values).to_vec());
    source.texture_descriptor.usage |= TextureUsages::COPY_DST;
    let source = images.add(source);
    let mut output = Image::new_target_texture(size.x, size.y, TextureFormat::Rgba32Float, None);
    output.data = None;
    output.copy_on_resize = false;
    output.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let output = images.add(output);
    let dummy = images.add(Image::new_target_texture(
        1,
        1,
        TextureFormat::Rgba8Unorm,
        None,
    ));
    let copier = ImageCopier::for_targets(
        vec![output.clone()],
        Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        TextureFormat::Rgba32Float,
        device,
    );
    copier.request(1);
    let mut entity = app.world_mut().spawn((
        Camera2d,
        bevy::camera::RenderTarget::Image(dummy.into()),
        SsaaCamera::new(source, output, size, factor),
        copier.clone(),
    ));
    if let Some(profile) = profile {
        entity.insert(profile);
    }
    copier
}

fn wait(app: &mut App, copiers: &[&ImageCopier]) {
    let start = Instant::now();
    while copiers.iter().any(|c| !c.ready(1)) {
        app.update();
        for copier in copiers {
            assert!(copier.failure().is_none(), "{:?}", copier.failure());
        }
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "RGB resolve stalled"
        );
    }
}

#[test]
#[ignore = "requires native GPU; exact integer-area filter oracle"]
fn gpu_box_matches_cpu_linear_area_average_at_all_factors() {
    let mut app = test_app();
    let device = app.world().resource::<RenderDevice>().clone();
    let mut checks = Vec::new();
    for factor in 2..=4 {
        let size = UVec2::new(3, 5);
        let high = size * factor;
        let mut values = Vec::new();
        for y in 0..high.y {
            for x in 0..high.x {
                values.extend([
                    ((x + y) % 2) as f32,
                    x as f32 / 32.,
                    y as f32 / 32.,
                    ((x * 3 + y) % 8) as f32 / 8.,
                ]);
            }
        }
        let copier = fixture(&mut app, &device, size, factor, &values, None);
        checks.push((factor, size, high, values, copier));
    }
    wait(&mut app, &checks.iter().map(|c| &c.4).collect::<Vec<_>>());
    for (factor, size, high, input, copier) in checks {
        let packet = copier.take(1).unwrap();
        let actual: &[f32] = bytemuck::cast_slice(&packet.planes[0]);
        assert_eq!(actual.len(), (size.x * size.y * 4) as usize);
        for y in 0..size.y {
            for x in 0..size.x {
                for channel in 0..4 {
                    let mut expected = 0.;
                    for dy in 0..factor {
                        for dx in 0..factor {
                            expected += input[(((y * factor + dy) * high.x + x * factor + dx) * 4
                                + channel) as usize];
                        }
                    }
                    expected /= (factor * factor) as f32;
                    let got = actual[((y * size.x + x) * 4 + channel) as usize];
                    assert!(
                        (got - expected).abs() < 1e-6,
                        "factor {factor}, ({x},{y}) channel {channel}: {got} != {expected}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires native GPU; sensor optics and signal-dependent noise oracle"]
fn gpu_sensor_preserves_masks_and_obeys_image_formation() {
    let mut app = test_app();
    let device = app.world().resource::<RenderDevice>().clone();
    let size = UVec2::splat(128);
    let policy = super::super::sensor::SensorArtifactSettings {
        probability: 1.,
        ..default()
    };
    let mut clean = policy.profile(99, 2).unwrap();
    clean.blur_axes_px512 = [0., 0.];
    clean.smudge_mix = 0.;
    clean.flare_gain = 0.;
    clean.read_std = 0.;
    clean.shot_std = 0.;
    let mut ghost = clean;
    ghost.flare_gain = 0.25;
    ghost.ghost_scale = 1.;
    ghost.ghost_radius_px512 = 2.;
    let mut smear = clean;
    smear.smudge_mix = 0.7;
    smear.smudge_center_uv = [0.25, 0.5];
    smear.smudge_radii = [0.22, 0.22];
    smear.smudge_direction = [1., 0.];
    smear.scatter_px512 = 6.;
    let mut blur = clean;
    blur.blur_direction = [1., 0.];
    blur.blur_axes_px512 = [5., 3.];
    let mut noise = clean;
    noise.read_std = 0.003;
    noise.shot_std = 0.028;
    let mut chart = Vec::new();
    let mut flat = Vec::new();
    for y in 0..size.y {
        for x in 0..size.x {
            let value = if (94..=106).contains(&x) && (57..=70).contains(&y) {
                1.
            } else {
                0.10 + ((x + y) % 2) as f32 * 0.08
            };
            let a = if x < 4 {
                0.
            } else if x == 4 {
                0.25
            } else {
                1.
            };
            chart.extend([value * a, value * a, value * a, a]);
            let v = if x < 64 { 0.1 } else { 0.55 };
            flat.extend([v * a, v * a, v * a, a]);
        }
    }
    let cases = [
        (&chart, ghost),
        (&chart, smear),
        (&chart, blur),
        (&flat, noise),
        (&flat, noise),
    ];
    let copiers: Vec<_> = cases
        .iter()
        .map(|(data, profile)| fixture(&mut app, &device, size, 1, data, Some(*profile)))
        .collect();
    wait(&mut app, &copiers.iter().collect::<Vec<_>>());
    let outputs: Vec<Vec<f32>> = copiers
        .iter()
        .map(|c| bytemuck::cast_slice::<u8, f32>(&c.take(1).unwrap().planes[0]).to_vec())
        .collect();
    for ((input, _), out) in cases.iter().zip(&outputs) {
        for (a, b) in input.as_chunks::<4>().0.iter().zip(out.as_chunks::<4>().0) {
            assert_eq!(a[3], b[3]);
            if a[3] == 0. {
                assert_eq!(a, b);
            }
        }
    }
    let index = |x, y| ((y * 128 + x) * 4) as usize;
    let delta = |case: usize, x, y| outputs[case][index(x, y)] - chart[index(x, y)];
    assert!(delta(0, 27, 64) > 0.1 && delta(0, 64, 82).abs() < 1e-6);
    assert!(delta(1, 32, 64).abs() > 0.01 && delta(1, 79, 64).abs() < 1e-6);
    assert!(delta(2, 78, 64).abs() > 0.01);
    assert_eq!(outputs[3], outputs[4]);
    for (range, signal) in [(8..64, 0.1_f32), (65..126, 0.55_f32)] {
        let errors: Vec<_> = (0..128)
            .flat_map(|y| range.clone().map(move |x| index(x, y)))
            .map(|i| outputs[3][i] - signal)
            .collect();
        let mean = errors.iter().sum::<f32>() / errors.len() as f32;
        let observed = errors.iter().map(|e| (e - mean).powi(2)).sum::<f32>() / errors.len() as f32;
        let expected = noise.read_std.powi(2) + noise.shot_std.powi(2) * signal;
        assert!(
            mean.abs() < 0.0009 && (observed / expected - 1.).abs() < 0.08,
            "{mean}, {observed}, {expected}"
        );
    }
}
