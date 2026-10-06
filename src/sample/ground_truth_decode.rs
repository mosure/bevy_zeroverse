//! Convert consumed native geometry planes into the established dataset layout.
//! Source allocations are reused; scalar equations and error checks stay exact.
use super::View;
use crate::{app::BevyZeroverseConfig, render::RenderMode, scene::SceneAabb};
use bevy::prelude::*;

/// Expand exact geometry attachments into the established dataset plane format.
/// World position and axial depth are reconstructed from the same raster sample.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn unpack_ground_truth(
    view: &mut View,
    planes: Vec<Vec<u8>>,
    modes: &[RenderMode],
    aabb: &SceneAabb,
    config: &BevyZeroverseConfig,
) -> Result<(), String> {
    if planes.len() != 3
        || planes
            .iter()
            .any(|p| p.len() != planes[0].len() || p.len() % 16 != 0)
    {
        return Err("malformed float32 geometry attachments".into());
    }
    let mut planes = planes.into_iter();
    let color = planes.next().unwrap();
    let mut world_depth = planes.next().unwrap();
    let mut normal_semantic = planes.next().unwrap();
    if modes.contains(&RenderMode::Color) {
        view.color = color;
    }
    let count = world_depth.len();
    let depth_enabled = modes.contains(&RenderMode::Depth);
    let normal_enabled = modes.contains(&RenderMode::Normal);
    let position_enabled = modes.contains(&RenderMode::Position);
    let semantic_enabled = modes.contains(&RenderMode::Semantic);
    // Position and normals overwrite their consumed source planes after each
    // original pixel is read. Depth can own world_depth when position is absent.
    // Only two separate allocations are needed when all annotations are requested.
    for (enabled, output) in [
        (depth_enabled && position_enabled, &mut view.depth),
        (semantic_enabled, &mut view.semantic),
    ] {
        if enabled {
            output.clear();
            output.reserve(count);
        }
    }
    let range = (aabb.max - aabb.min).max(Vec3::splat(1e-5));
    let camera_position = Mat4::from_cols_array_2d(&view.world_from_view)
        .w_axis
        .truncate();
    let append = |target: &mut Vec<u8>, values: [f32; 4]| {
        target.extend_from_slice(bytemuck::cast_slice(&values));
    };
    let palette: [Option<[f32; 4]>; 41] = std::array::from_fn(|id| {
        crate::render::ground_truth::semantic_label(id as u32)
            .map(|label| label.color().to_linear().to_f32_array())
    });
    for (wd, ns) in world_depth
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(normal_semantic.as_chunks_mut::<16>().0.iter_mut())
    {
        let read = |pixel: &[u8]| {
            std::array::from_fn::<f32, 4, _>(|i| {
                f32::from_ne_bytes(pixel[i * 4..i * 4 + 4].try_into().unwrap())
            })
        };
        let w = read(wd);
        let n = read(ns);
        if w.iter().chain(n.iter()).any(|v| !v.is_finite()) {
            return Err("non-finite float32 ground truth".into());
        }
        let hit = w[3] > 0.0;
        let alpha = if hit { 1.0 } else { 0.0 };
        if depth_enabled {
            let d = if config.z_depth {
                w[3]
            } else {
                (Vec3::new(w[0], w[1], w[2]) - camera_position).length()
            };
            let rgb = match config.depth_format {
                crate::render::depth::DepthFormat::Linear => [d; 3],
                crate::render::depth::DepthFormat::Normalized => [d / view.far; 3],
                crate::render::depth::DepthFormat::Colorized => {
                    let z = (view.near / w[3].max(view.near)).clamp(0.0, 1.0);
                    let smooth = |x: f32| {
                        let t = x.clamp(0.0, 1.0);
                        t * t * (3.0 - 2.0 * t)
                    };
                    [
                        smooth(2.0 * z - 1.0),
                        1.0 - (z - 0.5).abs() * 2.0,
                        1.0 - smooth(2.0 * z),
                    ]
                }
            };
            let depth = if hit {
                [rgb[0], rgb[1], rgb[2], alpha]
            } else {
                [0.0; 4]
            };
            if position_enabled {
                append(&mut view.depth, depth);
            } else {
                wd.copy_from_slice(bytemuck::bytes_of(&depth));
            }
        }
        if normal_enabled {
            ns.copy_from_slice(bytemuck::bytes_of(&[n[0], n[1], n[2], alpha]));
        }
        if position_enabled {
            // Context remains visible beyond the reconstruction region. Do not
            // collapse its positions onto the crop boundary: decoding must still
            // agree with depth and camera projection for every valid hit.
            let p = (Vec3::new(w[0], w[1], w[2]) - aabb.min) / range;
            let position = if hit {
                [p.x, p.y, p.z, alpha]
            } else {
                [0.0; 4]
            };
            wd.copy_from_slice(bytemuck::bytes_of(&position));
        }
        if semantic_enabled {
            let rgb = if hit {
                if n[3].fract() != 0.0 {
                    return Err("fractional geometry semantic ID".into());
                }
                palette
                    .get(n[3] as usize)
                    .copied()
                    .flatten()
                    .ok_or_else(|| format!("unknown geometry semantic ID {}", n[3]))?
            } else {
                [0.0; 4]
            };
            append(&mut view.semantic, [rgb[0], rgb[1], rgb[2], alpha]);
        }
    }
    if position_enabled {
        view.position = world_depth;
    } else if depth_enabled {
        view.depth = world_depth;
    }
    if normal_enabled {
        view.normal = normal_semantic;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Historical scalar path is retained as a byte oracle. It allocates each
    // output separately and never mutates its source planes.
    /// Expand exact geometry attachments into the established dataset plane format.
    /// World position and axial depth are reconstructed from the same raster sample.
    fn append_only_reference(
        view: &mut View,
        planes: Vec<Vec<u8>>,
        modes: &[RenderMode],
        aabb: &SceneAabb,
        config: &BevyZeroverseConfig,
    ) -> Result<(), String> {
        if planes.len() != 3
            || planes
                .iter()
                .any(|p| p.len() != planes[0].len() || p.len() % 16 != 0)
        {
            return Err("malformed float32 geometry attachments".into());
        }
        let mut planes = planes.into_iter();
        let color = planes.next().unwrap();
        let world_depth = planes.next().unwrap();
        let normal_semantic = planes.next().unwrap();
        if modes.contains(&RenderMode::Color) {
            view.color = color;
        }
        let count = world_depth.len();
        for (mode, output) in [
            (RenderMode::Depth, &mut view.depth),
            (RenderMode::Normal, &mut view.normal),
            (RenderMode::Position, &mut view.position),
            (RenderMode::Semantic, &mut view.semantic),
        ] {
            if modes.contains(&mode) {
                output.clear();
                output.reserve(count);
            }
        }
        let range = (aabb.max - aabb.min).max(Vec3::splat(1e-5));
        let camera_position = Mat4::from_cols_array_2d(&view.world_from_view)
            .w_axis
            .truncate();
        let append = |target: &mut Vec<u8>, values: [f32; 4]| {
            target.extend_from_slice(bytemuck::cast_slice(&values));
        };
        let depth_enabled = modes.contains(&RenderMode::Depth);
        let normal_enabled = modes.contains(&RenderMode::Normal);
        let position_enabled = modes.contains(&RenderMode::Position);
        let semantic_enabled = modes.contains(&RenderMode::Semantic);
        let palette: [Option<[f32; 4]>; 41] = std::array::from_fn(|id| {
            crate::render::ground_truth::semantic_label(id as u32)
                .map(|label| label.color().to_linear().to_f32_array())
        });
        for (wd, ns) in world_depth
            .as_chunks::<16>()
            .0
            .iter()
            .zip(normal_semantic.as_chunks::<16>().0.iter())
        {
            let read = |pixel: &[u8]| {
                std::array::from_fn::<f32, 4, _>(|i| {
                    f32::from_ne_bytes(pixel[i * 4..i * 4 + 4].try_into().unwrap())
                })
            };
            let w = read(wd);
            let n = read(ns);
            if w.iter().chain(n.iter()).any(|v| !v.is_finite()) {
                return Err("non-finite float32 ground truth".into());
            }
            let hit = w[3] > 0.0;
            let alpha = if hit { 1.0 } else { 0.0 };
            if depth_enabled {
                let d = if config.z_depth {
                    w[3]
                } else {
                    (Vec3::new(w[0], w[1], w[2]) - camera_position).length()
                };
                let rgb = match config.depth_format {
                    crate::render::depth::DepthFormat::Linear => [d; 3],
                    crate::render::depth::DepthFormat::Normalized => [d / view.far; 3],
                    crate::render::depth::DepthFormat::Colorized => {
                        let z = (view.near / w[3].max(view.near)).clamp(0.0, 1.0);
                        let smooth = |x: f32| {
                            let t = x.clamp(0.0, 1.0);
                            t * t * (3.0 - 2.0 * t)
                        };
                        [
                            smooth(2.0 * z - 1.0),
                            1.0 - (z - 0.5).abs() * 2.0,
                            1.0 - smooth(2.0 * z),
                        ]
                    }
                };
                append(
                    &mut view.depth,
                    if hit {
                        [rgb[0], rgb[1], rgb[2], alpha]
                    } else {
                        [0.0; 4]
                    },
                );
            }
            if normal_enabled {
                append(&mut view.normal, [n[0], n[1], n[2], alpha]);
            }
            if position_enabled {
                // Context remains visible beyond the reconstruction region. Do not
                // collapse its positions onto the crop boundary: decoding must still
                // agree with depth and camera projection for every valid hit.
                let p = (Vec3::new(w[0], w[1], w[2]) - aabb.min) / range;
                append(
                    &mut view.position,
                    if hit {
                        [p.x, p.y, p.z, alpha]
                    } else {
                        [0.0; 4]
                    },
                );
            }
            if semantic_enabled {
                let rgb = if hit {
                    if n[3].fract() != 0.0 {
                        return Err("fractional geometry semantic ID".into());
                    }
                    palette
                        .get(n[3] as usize)
                        .copied()
                        .flatten()
                        .ok_or_else(|| format!("unknown geometry semantic ID {}", n[3]))?
                } else {
                    [0.0; 4]
                };
                append(&mut view.semantic, [rgb[0], rgb[1], rgb[2], alpha]);
            }
        }
        Ok(())
    }

    fn planes() -> Vec<Vec<u8>> {
        let hits = [
            [0.0_f32, 1., -2., 2.],
            [-8., 2., -10., 10.],
            [8., 7., -8., 8.],
            [-0., 0., -0., 0.],
            [1., -1., 0.5, -1.],
        ];
        let normals = [
            [0.5_f32, 1., 0.5, 1.],
            [1., 0.5, 0.5, 2.],
            [-0., 0.5, 0.5, 40.],
            [-0., 0., 0.5, 0.],
            [0., -0., 0.5, 0.],
        ];
        vec![
            (0..hits.len() * 16).map(|i| i as u8).collect(),
            bytemuck::cast_slice(&hits).to_vec(),
            bytemuck::cast_slice(&normals).to_vec(),
        ]
    }

    fn view() -> View {
        View {
            color: vec![3, 5, 7],
            depth: vec![11, 13],
            normal: vec![17],
            position: vec![19, 23],
            semantic: vec![29, 31],
            world_from_view: Mat4::from_translation(Vec3::new(1., 2., 3.)).to_cols_array_2d(),
            near: 0.1,
            far: 50.,
            ..default()
        }
    }

    #[test]
    fn reused_ground_truth_matches_scalar_bytes_for_every_mode_subset() {
        let aabb = SceneAabb {
            min: Vec3::new(-2., 0., -3.),
            max: Vec3::new(2., 4., 3.),
        };
        let available = [
            RenderMode::Color,
            RenderMode::Depth,
            RenderMode::Normal,
            RenderMode::Position,
            RenderMode::Semantic,
        ];
        for bits in 0..1usize << available.len() {
            let modes: Vec<_> = available
                .iter()
                .enumerate()
                .filter(|(i, _)| bits & (1 << i) != 0)
                .map(|(_, mode)| mode.clone())
                .collect();
            for z_depth in [false, true] {
                for depth_format in [
                    crate::render::depth::DepthFormat::Linear,
                    crate::render::depth::DepthFormat::Normalized,
                    crate::render::depth::DepthFormat::Colorized,
                ] {
                    let config = BevyZeroverseConfig {
                        z_depth,
                        depth_format,
                        ..default()
                    };
                    for source in [planes(), vec![vec![], vec![], vec![]]] {
                        let mut expected = view();
                        let mut actual = expected.clone();
                        let reference = append_only_reference(
                            &mut expected,
                            source.clone(),
                            &modes,
                            &aabb,
                            &config,
                        );
                        let reused =
                            unpack_ground_truth(&mut actual, source, &modes, &aabb, &config);
                        assert_eq!(reused, reference);
                        assert_eq!(
                            actual, expected,
                            "bits={bits}, z_depth={z_depth}, depth={depth_format:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn output_ownership_reuses_consumed_geometry_allocations() {
        let aabb = SceneAabb {
            min: -Vec3::ONE,
            max: Vec3::ONE,
        };
        for position in [false, true] {
            let source = planes();
            let world_pointer = source[1].as_ptr();
            let normal_pointer = source[2].as_ptr();
            let mut modes = vec![RenderMode::Depth, RenderMode::Normal];
            if position {
                modes.push(RenderMode::Position);
            }
            let mut actual = view();
            unpack_ground_truth(&mut actual, source, &modes, &aabb, &default()).unwrap();
            assert_eq!(
                if position {
                    actual.position.as_ptr()
                } else {
                    actual.depth.as_ptr()
                },
                world_pointer
            );
            assert_eq!(actual.normal.as_ptr(), normal_pointer);
        }
    }

    #[test]
    fn malformed_and_nonfinite_ground_truth_keep_their_rejection_contract() {
        let aabb = SceneAabb {
            min: -Vec3::ONE,
            max: Vec3::ONE,
        };
        let modes = [
            RenderMode::Depth,
            RenderMode::Normal,
            RenderMode::Position,
            RenderMode::Semantic,
        ];
        let mut invalid = vec![
            vec![],
            vec![vec![0; 16]; 2],
            vec![vec![0; 17]; 3],
            vec![vec![0; 16], vec![0; 16], vec![0; 15]],
        ];
        for (plane, pixel, field, value) in [
            (1, 0, 0, f32::NAN),
            (2, 3, 2, f32::INFINITY),
            (2, 0, 3, 0.5),
            (2, 0, 3, 0.),
            (2, 0, 3, 41.),
        ] {
            let mut source = planes();
            let offset = pixel * 16 + field * 4;
            source[plane][offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
            invalid.push(source);
        }
        for source in invalid {
            let expected =
                append_only_reference(&mut view(), source.clone(), &modes, &aabb, &default());
            let actual = unpack_ground_truth(&mut view(), source, &modes, &aabb, &default());
            assert!(expected.is_err());
            assert_eq!(actual, expected);
        }
        // Unsupported semantic IDs are checked only when semantic output is
        // requested; all finite channels are still checked for RGB-only capture.
        let mut source = planes();
        source[2][12..16].copy_from_slice(&41_f32.to_ne_bytes());
        assert!(
            unpack_ground_truth(&mut view(), source, &[RenderMode::Color], &aabb, &default())
                .is_ok()
        );
    }
}
