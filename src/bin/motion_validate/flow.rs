use anyhow::{ensure, Result};
use bevy::prelude::*;
use bevy_zeroverse::{render::semantic::SemanticLabel, sample::Sample};
use std::path::Path;

/// Independent camera-only reprojection separates actor deformation from camera
/// motion. Non-person surfaces in these generated scenes are static controls.
pub fn validate(
    sample: &Sample,
    width: u32,
    height: u32,
    directory: &Path,
) -> Result<serde_json::Value> {
    let mut reports = Vec::new();
    let mut dynamic_person_pixels = 0;
    let person = SemanticLabel::Person.color().to_linear().to_f32_array();
    let count = width as usize * height as usize;
    for (i, source) in sample.views.iter().enumerate() {
        ensure!(
            source.optical_flow.len() == count * 16 && source.motion_vectors.len() == count * 16,
            "flow dimensions"
        );
        let flow: &[[f32; 4]] = bytemuck::cast_slice(&source.optical_flow);
        let motion: &[[f32; 4]] = bytemuck::cast_slice(&source.motion_vectors);
        let depth: &[[f32; 4]] = bytemuck::cast_slice(&source.depth);
        let semantic: &[[f32; 4]] = bytemuck::cast_slice(&source.semantic);
        let next = sample.views.get(i + sample.view_dim as usize);
        let mut valid = 0;
        let mut visible = 0;
        let mut moving_people = 0;
        let mut static_errors = Vec::new();
        let mut expected_static_valid = 0;
        let mut missing_static_valid = 0;
        let mut magnitude = 0.0_f32;
        let world_from_view = Mat4::from_cols_array_2d(&source.world_from_view);
        let focal = height as f32 / (2.0 * (source.fovy * 0.5).tan());
        let target_projection = next.map(|target| {
            (
                Mat4::from_cols_array_2d(&target.world_from_view).inverse(),
                height as f32 / (2.0 * (target.fovy * 0.5).tan()),
                target.near,
                target.far,
            )
        });
        let mut rgb = Vec::with_capacity(count * 3);
        for (pixel, (&p, &m)) in flow.iter().zip(motion).enumerate() {
            ensure!(
                p.iter().chain(m.iter()).all(|v| v.is_finite()),
                "nonfinite flow"
            );
            ensure!(
                [0.0, 1.0].contains(&p[2]) && [0.0, 1.0].contains(&p[3]) && p[3] <= p[2],
                "invalid flow masks"
            );
            ensure!(
                (p[0] / width as f32 - m[0]).abs() < 1e-7
                    && (p[1] / height as f32 - m[1]).abs() < 1e-7
                    && p[2..] == m[2..],
                "pixel/normalized flow disagreement"
            );
            if next.is_none() {
                ensure!(p == [0.0; 4], "terminal flow must be invalid");
            }
            let is_person = semantic[pixel][..3] == person[..3];
            let camera_only = target_projection.and_then(|(view, target_focal, near, far)| {
                let x = (pixel % width as usize) as f32 + 0.5;
                let y = (pixel / width as usize) as f32 + 0.5;
                let z = depth[pixel][0];
                let point = world_from_view.transform_point3(Vec3::new(
                    (x - width as f32 * 0.5) * z / focal,
                    (height as f32 * 0.5 - y) * z / focal,
                    -z,
                ));
                let q = view.transform_point3(point);
                (z > 0.0 && -q.z > near + 1e-4 && -q.z < far - 1e-4).then(|| {
                    Vec2::new(
                        q.x / -q.z * target_focal + width as f32 * 0.5 - x,
                        -q.y / -q.z * target_focal + height as f32 * 0.5 - y,
                    )
                })
            });
            if !is_person && camera_only.is_some() {
                expected_static_valid += 1;
                missing_static_valid += usize::from(p[2] == 0.0);
            }
            if p[2] == 0.0 {
                ensure!(p[..2] == [0.0; 2], "invalid vectors must be zero");
                rgb.extend_from_slice(&[0, 0, 0]);
                continue;
            }
            valid += 1;
            visible += usize::from(p[3] == 1.0);
            magnitude = magnitude.max(p[0].hypot(p[1]));
            if let Some(camera_only) = camera_only {
                let residual = camera_only.distance(Vec2::new(p[0], p[1]));
                if is_person {
                    moving_people += usize::from(residual > 0.05);
                } else {
                    static_errors.push(residual);
                }
            }
            let hue = p[1].atan2(p[0]).to_degrees().rem_euclid(360.0);
            let color = Color::hsv(
                hue,
                (p[0].hypot(p[1]) / 40.0).min(1.0),
                if p[3] > 0.0 { 1.0 } else { 0.45 },
            )
            .to_srgba();
            rgb.extend_from_slice(&[
                (color.red * 255.0) as u8,
                (color.green * 255.0) as u8,
                (color.blue * 255.0) as u8,
            ]);
        }
        static_errors.sort_by(f32::total_cmp);
        let p99 = static_errors
            .get(static_errors.len() * 99 / 100)
            .copied()
            .unwrap_or(0.0);
        ensure!(
            p99 < 0.03,
            "static control flow disagrees with camera reprojection: p99={p99}px"
        );
        ensure!(
            missing_static_valid == 0,
            "view {i} lost {missing_static_valid}/{expected_static_valid} static correspondences"
        );
        dynamic_person_pixels += moving_people;
        image::save_buffer(
            directory.join(format!("flow_{i:02}.png")),
            &rgb,
            width,
            height,
            image::ColorType::Rgb8,
        )?;
        std::fs::write(
            directory.join(format!("flow_{i:02}.f32")),
            &source.optical_flow,
        )?;
        reports.push(serde_json::json!({"view":i,"valid_pixels":valid,"visible_pixels":visible,"moving_person_pixels":moving_people,"maximum_displacement_pixels":magnitude,"static_camera_reprojection_p99_pixels":p99,"expected_static_valid_pixels":expected_static_valid,"missing_static_valid_pixels":missing_static_valid}));
    }
    Ok(serde_json::json!({"moving_person_pixels":dynamic_person_pixels,"views":reports}))
}
