//! Editor-only rendering; outlines are cached by calibration, never by motion.
use super::*;
use crate::{
    app::BevyZeroverseConfig,
    camera::{
        CaptureCameraIndex, EditorCameraGizmoConfigGroup, EditorCameraMarker, ZeroverseCamera,
    },
    scene::ZeroverseSceneRoot,
};
use bevy::gizmos::config::GizmoConfigGroup;
use std::collections::HashMap;

#[derive(Default, Reflect, GizmoConfigGroup)]
pub struct PredictionCameraGizmoConfigGroup;

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn draw_camera_gizmo(
    args: Res<BevyZeroverseConfig>,
    mut gizmos: Gizmos<EditorCameraGizmoConfigGroup>,
    mut predicted: Gizmos<PredictionCameraGizmoConfigGroup>,
    mut cameras: Query<
        (
            Entity,
            &GlobalTransform,
            &Projection,
            &mut ZeroverseCamera,
            Option<&super::super::lens::LensCamera>,
            Option<&CaptureCameraIndex>,
        ),
        (With<Camera>, Without<EditorCameraMarker>),
    >,
    editor: Query<
        (&Camera, &Projection, &GlobalTransform),
        (With<EditorCameraMarker>, Without<ZeroverseCamera>),
    >,
    scene: Query<&GlobalTransform, With<ZeroverseSceneRoot>>,
    mut outlines: Local<HashMap<Entity, (CameraCalibration, Vec<[Vec3; 2]>)>>,
    #[cfg(feature = "viewer")] predictions: Option<Res<crate::app::editor::schematic::Predictions>>,
    #[cfg(feature = "viewer")] mut prediction_outlines: Local<
        HashMap<usize, (CameraCalibration, Vec<[Vec3; 2]>)>,
    >,
) {
    if !args.gizmos || args.camera_grid || args.room_schematic {
        return;
    }
    let Some((editor_camera, editor_projection, editor_transform)) =
        editor.iter().find(|(c, ..)| c.is_active)
    else {
        return;
    };
    let viewport = editor_camera
        .logical_viewport_size()
        .unwrap_or(Vec2::splat(768.));
    let view_from_world = editor_transform.to_matrix().inverse();
    let billboard = editor_transform.rotation();
    let alpha = args.gizmos_alpha.clamp(0., 1.);
    let scene_transform = scene
        .single()
        .map_or(Transform::IDENTITY, GlobalTransform::compute_transform);
    outlines.retain(|entity, _| cameras.get(*entity).is_ok());
    // CaptureCameraIndex is authoritative. Deterministic entity ordering is
    // only a fallback for custom integrations without capture IDs.
    let mut ids: Vec<_> = cameras
        .iter()
        .map(|(e, .., index)| (e, index.map(|i| i.0)))
        .collect();
    ids.sort_by_key(|(e, index)| (index.unwrap_or(usize::MAX), e.to_bits()));
    let next_id = ids
        .iter()
        .filter_map(|(_, i)| *i)
        .max()
        .map_or(0, |i| i.saturating_add(1));
    let mut anonymous = 0;
    for (entity, index) in ids {
        let index = index.unwrap_or_else(|| {
            let id = next_id.saturating_add(anonymous);
            anonymous += 1;
            id
        });
        let Ok((_, global, projection, mut camera, angular, _)) = cameras.get_mut(entity) else {
            continue;
        };
        let c = angular
            .map(|a| Ok(a.calibration.clone()))
            .unwrap_or_else(|| {
                super::super::lens::calibration(
                    projection,
                    camera.fisheye,
                    camera.principal_point_offset,
                    camera.resolution.unwrap_or(UVec2::splat(512)).to_array(),
                )
            });
        let Ok(c) = c else {
            continue;
        };
        if outlines.get(&entity).is_none_or(|(old, _)| old != &c) {
            let Ok(lines) = wireframe(&c) else {
                continue;
            };
            outlines.insert(entity, (c, lines));
        }
        let pose = global.compute_transform();
        // Camera glyph scale is presentation-only, independent of scene-root scale.
        let matrix = Mat4::from_rotation_translation(pose.rotation, pose.translation);
        let radius = display_radius(
            editor_projection,
            view_from_world,
            viewport,
            pose.translation,
        );
        let color = color(index).with_alpha(alpha);
        draw(&mut gizmos, &outlines[&entity].1, matrix, radius, color);
        label(
            &mut gizmos,
            &format!("C{index}"),
            pose.translation,
            billboard,
            radius,
            color,
            false,
        );
        camera
            .trajectory
            .draw_gizmos(&mut gizmos, scene_transform, color.with_alpha(alpha * 0.45));
    }
    #[cfg(feature = "viewer")]
    {
        let cameras = predictions
            .as_ref()
            .map_or(&[][..], |p| p.0.cameras.as_slice());
        prediction_outlines.retain(|slot, _| *slot < cameras.len());
        for (fallback, camera) in cameras.iter().enumerate() {
            let matrix = Mat4::from_cols_array_2d(&camera.world_from_view);
            if !rigid_pose(matrix) {
                continue;
            }
            let Ok(c) = camera.display_calibration() else {
                continue;
            };
            if prediction_outlines
                .get(&fallback)
                .is_none_or(|(old, _)| old != &c)
            {
                let Ok(lines) = prediction_wireframe(&c) else {
                    continue;
                };
                prediction_outlines.insert(fallback, (c, lines));
            }
            let origin = matrix.transform_point3(Vec3::ZERO);
            let index = color_index(&camera.label, fallback);
            let color = color(index).with_alpha(alpha);
            let radius = display_radius(editor_projection, view_from_world, viewport, origin);
            draw(
                &mut predicted,
                &prediction_outlines[&fallback].1,
                matrix,
                radius,
                color,
            );
            // Text stays solid and readable; line style distinguishes predictions.
            label(
                &mut gizmos,
                &format!("Pred C{index}"),
                origin,
                billboard,
                radius,
                color,
                true,
            );
        }
    }
    #[cfg(not(feature = "viewer"))]
    let _ = &mut predicted;
}

fn draw<G: GizmoConfigGroup>(
    gizmos: &mut Gizmos<G>,
    lines: &[[Vec3; 2]],
    pose: Mat4,
    radius: f32,
    color: Color,
) {
    if radius <= 1e-6 {
        return;
    }
    for &[a, b] in lines {
        gizmos.line(
            pose.transform_point3(a * radius),
            pose.transform_point3(b * radius),
            color,
        );
    }
    let origin = pose.transform_point3(Vec3::ZERO);
    gizmos
        .arrow(
            origin,
            pose.transform_point3(Vec3::NEG_Z * radius * 0.8),
            color,
        )
        .with_tip_length(radius * 0.12);
    gizmos.sphere(origin, radius * 0.065, color).resolution(8);
}
fn label<G: GizmoConfigGroup>(
    gizmos: &mut Gizmos<G>,
    label: &str,
    origin: Vec3,
    billboard: Quat,
    radius: f32,
    color: Color,
    predicted: bool,
) {
    if radius <= 1e-6 {
        return;
    }
    let offset = billboard
        * Vec3::new(
            radius * 0.12,
            radius * if predicted { 1.9 } else { 1.2 },
            0.,
        );
    gizmos.text(
        Isometry3d::new(origin + offset, billboard),
        label,
        radius * 0.55,
        Vec2::ZERO,
        color,
    );
}
