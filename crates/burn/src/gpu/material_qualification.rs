//! Hardware qualification of material baking at the actual capture boundary.
use super::tests::{equal_plane, setup_test_assets, values};
use super::*;
use bevy_zeroverse::render::{RenderMode, glass::AnnotationGlass};

fn save_color(path: impl AsRef<std::path::Path>, data: &[f32], size: u32) -> Result<()> {
    let rgba: Vec<u8> = data
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| {
            p[..3]
                .iter()
                .map(|v| {
                    let v = v.clamp(0., 1.);
                    let s = if v <= 0.0031308 {
                        v * 12.92
                    } else {
                        1.055 * v.powf(1. / 2.4) - 0.055
                    };
                    (s * 255.).round() as u8
                })
                .chain(std::iter::once(255))
                .collect::<Vec<_>>()
        })
        .collect();
    image::save_buffer(path, &rgba, size, size, image::ColorType::Rgba8)?;
    Ok(())
}

#[test]
#[ignore = "native GPU: through-glass exterior coverage, primary-room bounds, and RGB-only exposure changes"]
fn finite_exterior_and_exposure_preserve_capture_contracts() -> Result<()> {
    use bevy_zeroverse::render::{ground_truth::semantic_id, semantic::SemanticLabel};
    setup_test_assets();
    let output = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../out/gpu_materials/exterior_exposure");
    std::fs::create_dir_all(&output)?;
    let mut renderer = Renderer::with_transport(
        BevyZeroverseConfig {
            scene_type: ZeroverseSceneType::ProceduralIndoor,
            indoor_seed: Some(0),
            annotation_glass: AnnotationGlass::Through,
            num_cameras: 3,
            width: 512.,
            height: 512.,
            playback_steps: 2,
            indoor_human_density: 0.25,
            depth_format: bevy_zeroverse::render::depth::DepthFormat::Linear,
            render_modes: vec![
                RenderMode::Color,
                RenderMode::Depth,
                RenderMode::Normal,
                RenderMode::Position,
                RenderMode::Semantic,
                RenderMode::OpticalFlow,
                RenderMode::CoVisibility,
            ],
            ..default()
        },
        Transport::Wgpu,
    )?;
    let mut reports = Vec::new();
    let sky_id = semantic_id(&SemanticLabel::Sky) as f32;
    let mut sky_total = 0;
    for seed in (0..8).chain([57_541_088, 57_491_588]) {
        let sample = renderer.sample_seed(seed)?;
        assert!(
            renderer
                .app
                .world()
                .resource::<procedural_indoor::materials::gpu::MaterialBakeStatistics>()
                .gpu_materials
                > 0,
            "default configuration must bake on this capable device"
        );
        // A sky closure must not become the dataset's voxel/reconstruction region.
        let [lo, hi] = sample.metadata.aabb;
        assert!(hi[0] - lo[0] < 30. && hi[2] - lo[2] < 30. && hi[1] - lo[1] < 8.);
        for (index, view) in sample.views.iter().enumerate() {
            let geometry = values(view.world_depth.clone().unwrap());
            let labels = values(view.normal_semantic.clone().unwrap());
            let mut sky_pixels = 0;
            let mut max_depth = 0f32;
            for (p, n) in geometry
                .as_chunks::<4>()
                .0
                .iter()
                .zip(labels.as_chunks::<4>().0)
            {
                assert!(
                    p.iter().all(|x| x.is_finite()) && p[3] > 0.,
                    "seed {seed} view {index}: invalid exterior depth {p:?}"
                );
                let length_squared = n[..3].iter().map(|x| (2. * x - 1.).powi(2)).sum::<f32>();
                assert!(
                    n[3] > 0. && (0.98..1.02).contains(&length_squared),
                    "seed {seed} view {index}: invalid normal/semantic {n:?}"
                );
                max_depth = max_depth.max(p[3]);
                sky_pixels += usize::from(n[3] == sky_id);
            }
            assert!(
                max_depth < 50.,
                "seed {seed}: finite background exceeds the existing camera range"
            );
            if sky_pixels > 0 && index < 3 {
                save_color(
                    output.join(format!("room{seed}_view{index}.png")),
                    &values(view.color.clone().unwrap()),
                    512,
                )?;
                let bytes: Vec<u8> = geometry
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .flat_map(|p| {
                        let d = (p[3] / 50. * 255.).round() as u8;
                        [d, d, d, 255]
                    })
                    .collect();
                image::save_buffer(
                    output.join(format!("room{seed}_view{index}_depth.png")),
                    &bytes,
                    512,
                    512,
                    image::ColorType::Rgba8,
                )?;
            }
            sky_total += sky_pixels;
            reports.push(serde_json::json!({"seed":seed,"view":index,"invalid_depth_pixels":0,"sky_pixels":sky_pixels,"max_depth_m":max_depth}));
        }
    }
    assert!(
        sky_total > 1000,
        "window views must actually exercise the sky closure"
    );
    let mut reference: Option<GpuSample> = None;
    let mut luminance = Vec::new();
    for offset in [-1., 0., 1.] {
        let policy = Some(serde_json::json!({"exposure_ev100_offset":offset}).to_string());
        renderer.config.indoor_appearance = policy.clone();
        renderer
            .app
            .world_mut()
            .resource_mut::<BevyZeroverseConfig>()
            .indoor_appearance = policy;
        let sample = renderer.sample_seed(202)?;
        let mut total = 0f64;
        for (index, view) in sample.views.iter().enumerate() {
            let rgb = values(view.color.clone().unwrap());
            assert!(rgb.iter().all(|v| v.is_finite()));
            total += rgb
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| f64::from(p[0] * 0.2126 + p[1] * 0.7152 + p[2] * 0.0722))
                .sum::<f64>();
            if index == 1 {
                save_color(output.join(format!("exposure_{offset}.png")), &rgb, 512)?;
            }
            if let Some(before) = &reference {
                let a = &before.views[index];
                equal_plane(
                    values(a.world_depth.clone().unwrap()),
                    values(view.world_depth.clone().unwrap()),
                    "exposure position/depth",
                );
                equal_plane(
                    values(a.normal_semantic.clone().unwrap()),
                    values(view.normal_semantic.clone().unwrap()),
                    "exposure normals/semantics",
                );
                equal_plane(
                    values(a.co_visibility.clone().unwrap()),
                    values(view.co_visibility.clone().unwrap()),
                    "exposure visibility",
                );
                equal_plane(
                    values(a.optical_flow.clone().unwrap()),
                    values(view.optical_flow.clone().unwrap()),
                    "exposure flow",
                );
            }
        }
        luminance.push(total / (512. * 512. * sample.views.len() as f64));
        reference = Some(sample);
    }
    assert!(
        luminance[0] > luminance[1] * 1.10 && luminance[1] > luminance[2] * 1.10,
        "exposure must monotonically darken RGB: {luminance:?}"
    );
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"views":reports,"exposure_ev100_offsets":[-1,0,1],"mean_linear_luminance":luminance,"exposure_annotation_differences":0}),
        )?,
    )?;
    Ok(())
}

#[test]
#[ignore = "native GPU: matched full-quality three-view RGB and all rendered annotations with CPU/GPU materials"]
fn baked_materials_preserve_completed_scene_annotations() -> Result<()> {
    setup_test_assets();
    let output = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../out/gpu_materials");
    std::fs::create_dir_all(&output)?;
    let mut renderer = Renderer::with_transport(
        BevyZeroverseConfig {
            scene_type: ZeroverseSceneType::ProceduralIndoor,
            indoor_seed: Some(6),
            indoor_gpu_materials: false,
            num_cameras: 3,
            width: 512.,
            height: 512.,
            playback_steps: 2,
            indoor_human_density: 0.25,
            depth_format: bevy_zeroverse::render::depth::DepthFormat::Linear,
            render_modes: vec![
                RenderMode::Color,
                RenderMode::Depth,
                RenderMode::Normal,
                RenderMode::Position,
                RenderMode::Semantic,
                RenderMode::OpticalFlow,
                RenderMode::CoVisibility,
            ],
            ..default()
        },
        Transport::Wgpu,
    )?;
    let mut report = Vec::new();
    for seed in [6, 202, 400, 401] {
        for glass in [AnnotationGlass::Surface, AnnotationGlass::Through] {
            renderer.config.annotation_glass = glass;
            renderer
                .app
                .world_mut()
                .resource_mut::<BevyZeroverseConfig>()
                .annotation_glass = glass;
            renderer.config.indoor_gpu_materials = false;
            renderer
                .app
                .world_mut()
                .resource_mut::<BevyZeroverseConfig>()
                .indoor_gpu_materials = false;
            let cpu = renderer.sample_seed(seed)?;
            if seed == 6 && glass == AnnotationGlass::Surface {
                let replay = renderer.sample_seed(seed)?;
                for (a, b) in cpu.views.iter().zip(&replay.views) {
                    equal_plane(
                        values(a.normal_semantic.clone().unwrap()),
                        values(b.normal_semantic.clone().unwrap()),
                        "CPU replay geometric normals/semantics",
                    );
                }
            }

            renderer.config.indoor_gpu_materials = true;
            renderer
                .app
                .world_mut()
                .resource_mut::<BevyZeroverseConfig>()
                .indoor_gpu_materials = true;
            let gpu = renderer.sample_seed(seed)?;
            let statistics = *renderer
                .app
                .world()
                .resource::<procedural_indoor::materials::gpu::MaterialBakeStatistics>(
            );
            assert!(
                statistics.gpu_materials >= 3,
                "GPU path was not exercised: {statistics:?}"
            );
            assert_eq!(cpu.metadata.aabb, gpu.metadata.aabb);
            assert_eq!(cpu.views.len(), 6);
            for (index, (a, b)) in cpu.views.iter().zip(&gpu.views).enumerate() {
                assert_eq!(
                    cpu.metadata.views[index].world_from_view,
                    gpu.metadata.views[index].world_from_view
                );
                equal_plane(
                    values(a.world_depth.clone().unwrap()),
                    values(b.world_depth.clone().unwrap()),
                    "material bake position/depth",
                );
                equal_plane(
                    values(a.normal_semantic.clone().unwrap()),
                    values(b.normal_semantic.clone().unwrap()),
                    "material bake geometric normals/semantics",
                );
                equal_plane(
                    values(a.co_visibility.clone().unwrap()),
                    values(b.co_visibility.clone().unwrap()),
                    "material bake co-visibility",
                );
                equal_plane(
                    values(a.optical_flow.clone().unwrap()),
                    values(b.optical_flow.clone().unwrap()),
                    "material bake physical-time flow",
                );
                let a = values(a.color.clone().unwrap());
                let b = values(b.color.clone().unwrap());
                let mut squared = 0.;
                let mut max = 0f32;
                for (x, y) in a.iter().zip(&b) {
                    assert!(x.is_finite() && y.is_finite());
                    let d = (x - y).abs();
                    max = max.max(d);
                    squared += f64::from(d).powi(2);
                }
                let rmse = (squared / a.len() as f64).sqrt();
                assert!(
                    rmse < 0.004 && max < 0.20,
                    "seed {seed} view {index} {glass:?}: RGB rmse {rmse} max {max}"
                );
                if seed == 202 && index < 3 && glass == AnnotationGlass::Surface {
                    for (tag, data) in [("cpu", a), ("gpu", b)] {
                        // Captured RGB is tonemapped linear; encode sRGB for visual inspection only.
                        let rgba: Vec<u8> = data
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .flat_map(|p| {
                                let c = p[..3].iter().map(|v| {
                                    let v = v.clamp(0., 1.);
                                    let s = if v <= 0.0031308 {
                                        v * 12.92
                                    } else {
                                        1.055 * v.powf(1. / 2.4) - 0.055
                                    };
                                    (s * 255.).round() as u8
                                });
                                c.chain(std::iter::once(255)).collect::<Vec<_>>()
                            })
                            .collect();
                        image::save_buffer(
                            output.join(format!("room202_view{index}_{tag}.png")),
                            &rgba,
                            512,
                            512,
                            image::ColorType::Rgba8,
                        )?;
                    }
                }
                report.push(serde_json::json!({"seed":seed,"glass":glass,"material_bake":statistics,"view":index,"rgb_rmse":rmse,"rgb_max_absolute":max,"annotation_differing_values":0}));
            }
        }
    }

    std::fs::write(
        output.join("render_parity.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(())
}
