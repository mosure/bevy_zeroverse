//! Pinhole projection with an explicit optical center. The same matrix drives
//! RGB, geometry attachments, visibility, optical flow, culling and calibration.
use crate::calibration::CameraCalibration;
use bevy::{
    camera::{CameraProjection, SubCameraView},
    math::Vec3A,
    prelude::*,
};

#[derive(Debug, Clone)]
pub struct ShiftedPerspective {
    pub perspective: PerspectiveProjection,
    /// Displacement from image center as fractions of width/height, right/down.
    pub offset: Vec2,
}

impl CameraProjection for ShiftedPerspective {
    fn get_clip_from_view(&self) -> Mat4 {
        let mut m = self.perspective.get_clip_from_view();
        m.z_axis.x -= 2. * self.offset.x;
        m.z_axis.y += 2. * self.offset.y;
        m
    }
    fn get_clip_from_view_for_sub(&self, sub: &SubCameraView) -> Mat4 {
        let mut m = self.perspective.get_clip_from_view_for_sub(sub);
        let ratio = sub.full_size.as_vec2() / sub.size.as_vec2();
        m.z_axis.x -= 2. * self.offset.x * ratio.x;
        m.z_axis.y += 2. * self.offset.y * ratio.y;
        m
    }
    fn update(&mut self, width: f32, height: f32) {
        self.perspective.update(width, height);
    }
    fn far(&self) -> f32 {
        self.perspective.far
    }
    fn get_frustum_corners(&self, near: f32, far: f32) -> [Vec3A; 8] {
        let scale = (self.perspective.fov * 0.5).tan();
        self.perspective.get_frustum_corners(near, far).map(|p| {
            p + Vec3A::new(
                2. * self.offset.x * self.perspective.aspect_ratio,
                -2. * self.offset.y,
                0.,
            ) * p.z
                * scale
        })
    }
}

pub fn projection(perspective: PerspectiveProjection, offset: Vec2) -> Projection {
    if offset == Vec2::ZERO {
        Projection::Perspective(perspective)
    } else {
        Projection::custom(ShiftedPerspective {
            perspective,
            offset,
        })
    }
}

/// Unsupported custom/orthographic projections remain explicit, not mislabeled
/// as a centered pinhole. Callers must not infer intrinsics from FOV alone.
pub(crate) fn parameters(projection: &Projection) -> Option<(&PerspectiveProjection, Vec2)> {
    match projection {
        Projection::Perspective(p) => Some((p, Vec2::ZERO)),
        Projection::Custom(p) => p
            .get::<ShiftedPerspective>()
            .map(|p| (&p.perspective, p.offset)),
        _ => None,
    }
}

pub(crate) fn calibration(
    projection: &Projection,
    width: u32,
    height: u32,
) -> Result<CameraCalibration, String> {
    let (p, offset) = parameters(projection).ok_or("unsupported capture projection")?;
    let mut c = CameraCalibration::centered_pinhole(width, height, p.fov, p.aspect_ratio)?;
    c.k[0][2] += offset.x * width as f32;
    c.k[1][2] += offset.y * height as f32;
    c.validate()?;
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ultra_wide_projection_matches_pixel_calibration_through_image_edges() {
        for size in [UVec2::new(768, 512), UVec2::new(320, 640)] {
            for degrees in [108_f32, 125., 145., 150.] {
                for offset in [Vec2::ZERO, Vec2::new(0.2, -0.2)] {
                    let mut lens = ShiftedPerspective {
                        perspective: PerspectiveProjection {
                            fov: degrees.to_radians(),
                            ..default()
                        },
                        offset,
                    };
                    lens.update(size.x as f32, size.y as f32);
                    let labels =
                        calibration(&Projection::custom(lens.clone()), size.x, size.y).unwrap();
                    for pixel in [
                        Vec2::splat(0.5),
                        size.as_vec2() - 0.5,
                        Vec2::new(0.5, size.y as f32 - 0.5),
                        size.as_vec2() * 0.5,
                    ] {
                        for depth in [0.1, 1., 50.] {
                            let q = Vec3::from_array(
                                labels.unproject(pixel.to_array(), depth).unwrap(),
                            );
                            let ndc = lens.get_clip_from_view().project_point3(q);
                            let projected =
                                Vec2::new(ndc.x + 1., 1. - ndc.y) * size.as_vec2() * 0.5;
                            assert!(
                                (projected - pixel).length() < 0.001,
                                "FOV {degrees}: {projected} != {pixel}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn seedable_optical_centers_preserve_room_and_visibility_contracts() {
        use crate::scene::procedural_indoor::{
            cameras::{multiview::OverlapGraph, CameraSettings},
            layout::{IndoorLayout, IndoorManifest},
            validation::validate_layout,
        };
        for seed in 0..8 {
            let original =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.)
                    .unwrap();
            for aspect in [0.75, 1.6] {
                for graph in [OverlapGraph::Reference, OverlapGraph::Connected] {
                    let mut settings = CameraSettings {
                        principal_point_jitter: Vec2::splat(0.15),
                        ..default()
                    };
                    settings.multiview.as_mut().unwrap().graph = graph;
                    let mut scene = original.clone();
                    scene.resample_cameras(4, settings.clone(), aspect).unwrap();
                    validate_layout(&scene).unwrap();
                    assert_eq!(original.objects, scene.objects);
                    assert_eq!(original.envelope, scene.envelope);
                    let cameras = scene.cameras.clone();
                    scene.resample_cameras(4, settings, aspect).unwrap();
                    assert_eq!(cameras, scene.cameras);
                    for c in &cameras {
                        let offset = c.principal_point_offset.unwrap();
                        assert!(
                            offset.abs().cmple(Vec2::splat(0.15)).all() && offset != Vec2::ZERO
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn shifted_matrices_intrinsics_corners_and_subviews_agree() {
        for size in [UVec2::new(640, 480), UVec2::new(320, 720)] {
            for offset in [Vec2::ZERO, Vec2::new(0.15, -0.12), Vec2::new(-0.2, 0.2)] {
                let mut p = ShiftedPerspective {
                    perspective: Default::default(),
                    offset,
                };
                p.update(size.x as f32, size.y as f32);
                let c = calibration(&Projection::custom(p.clone()), size.x, size.y).unwrap();
                for q in [Vec3::new(0., 0., -2.), Vec3::new(0.2, -0.3, -1.5)] {
                    let ndc = p.get_clip_from_view().project_point3(q);
                    let pixel = Vec2::new(ndc.x + 1., 1. - ndc.y) * size.as_vec2() * 0.5;
                    assert!(
                        (pixel - Vec2::from_array(c.project(q.to_array()).unwrap())).length()
                            < 0.0002
                    );
                }
                for (i, q) in p.get_frustum_corners(-0.1, -10.).iter().enumerate() {
                    let ndc = p.get_clip_from_view().project_point3((*q).into());
                    let expected = [
                        Vec2::new(1., -1.),
                        Vec2::ONE,
                        Vec2::new(-1., 1.),
                        -Vec2::ONE,
                    ][i % 4];
                    assert!((ndc.xy() - expected).length() < 1e-5);
                }
                let sub = SubCameraView {
                    full_size: size,
                    offset: Vec2::new(10., 20.),
                    size: size / 2,
                };
                let q = Vec3::new(0.1, -0.2, -2.);
                let ndc = p.get_clip_from_view_for_sub(&sub).project_point3(q);
                let pixel = Vec2::new(ndc.x + 1., 1. - ndc.y) * sub.size.as_vec2() * 0.5;
                assert!(
                    (pixel + sub.offset - Vec2::from_array(c.project(q.to_array()).unwrap()))
                        .length()
                        < 0.0002
                );
            }
        }
    }
}
