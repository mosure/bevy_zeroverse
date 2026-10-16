use super::*;
use crate::{app::BevyZeroverseConfig, headless::create_app, io::image_copy::ImageCopier};
use std::time::{Duration, Instant};

#[test]
#[ignore = "requires native GPU; exact integer-area filter oracle"]
fn gpu_box_matches_cpu_linear_area_average_at_all_factors() {
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
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        let mut source =
            Image::new_target_texture(high.x, high.y, TextureFormat::Rgba32Float, None);
        source.data = Some(bytemuck::cast_slice(&values).to_vec());
        source.texture_descriptor.usage |= TextureUsages::COPY_DST;
        let source = images.add(source);
        let mut output =
            Image::new_target_texture(size.x, size.y, TextureFormat::Rgba32Float, None);
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
            &device,
        );
        copier.request(1);
        app.world_mut().spawn((
            Camera2d,
            bevy::camera::RenderTarget::Image(dummy.into()),
            SsaaCamera::new(source, output, size, factor),
            copier.clone(),
        ));
        checks.push((factor, size, high, values, copier));
    }
    let start = Instant::now();
    while checks.iter().any(|(_, _, _, _, c)| !c.ready(1)) {
        app.update();
        for (_, _, _, _, copier) in &checks {
            assert!(copier.failure().is_none(), "{:?}", copier.failure());
        }
        assert!(
            start.elapsed() < Duration::from_secs(90),
            "SSAA resolve stalled"
        );
    }
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
