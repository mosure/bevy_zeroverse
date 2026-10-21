//! Source-view coverage for angular RGB. Crop unused *directions*, never reduce
//! angular sampling density or change the calibrated output field of view.
use bevy::{
    camera::{CameraProjection, SubCameraView},
    math::Vec3A,
    prelude::*,
};
mod density;
pub(super) use density::density;

/// Pixel regions of the original guarded square faces. Side views need only
/// the portion intersecting the lens cone, plus screen-space effect context.
pub(super) fn regions(side: u32, scale: f32, max_angle: f32) -> [URect; 6] {
    let full = URect::from_corners(UVec2::ZERO, UVec2::splat(side));
    let mut regions = [full; 6];
    if max_angle <= std::f32::consts::FRAC_PI_4 {
        return regions;
    }
    // For a side face, a is the coordinate toward the optical axis and b is
    // transverse. Its cone boundary is a = cot(theta) * sqrt(1 + b*b).
    // The negative (rearward) bound must include the most extreme b, not b=0.
    let cot = max_angle.cos() / max_angle.sin();
    let bound = if cot >= 0. {
        cot
    } else {
        cot * (1. + scale.recip().powi(2)).sqrt()
    };
    let context = (side as f32 / 16.).ceil().max(32.);
    let cut = ((0.5 + 0.5 * bound * scale) * side as f32 - context)
        .floor()
        .clamp(0., side.saturating_sub(4) as f32) as u32;
    // Same orientation ordering as runtime::orientations. Snap to 16 pixels
    // so cropped SSAO/bloom downsampling keeps the full-face texel phase.
    let cut = cut / 16 * 16;
    regions[1].max.x = side - cut;
    regions[2].min.x = cut;
    regions[3].min.y = cut;
    regions[4].max.y = side - cut;
    regions
}

/// The raster projection is cropped, while directional-shadow cascade fitting
/// retains the original guard-band frustum. Cropping must not change shadow
/// coverage/resolution or introduce new cascade seams between source faces.
#[derive(Debug, Clone)]
pub(super) struct FaceProjection {
    pub perspective: PerspectiveProjection,
    pub region: URect,
    pub full_side: u32,
}
impl CameraProjection for FaceProjection {
    fn get_clip_from_view(&self) -> Mat4 {
        self.perspective.get_clip_from_view_for_sub(&SubCameraView {
            full_size: UVec2::splat(self.full_side),
            offset: self.region.min.as_vec2(),
            size: self.region.size(),
        })
    }
    fn get_clip_from_view_for_sub(&self, sub: &SubCameraView) -> Mat4 {
        self.perspective.get_clip_from_view_for_sub(&SubCameraView {
            full_size: UVec2::splat(self.full_side),
            offset: self.region.min.as_vec2() + sub.offset,
            size: sub.size,
        })
    }
    fn update(&mut self, _: f32, _: f32) {}
    fn far(&self) -> f32 {
        self.perspective.far
    }
    fn get_frustum_corners(&self, near: f32, far: f32) -> [Vec3A; 8] {
        self.perspective.get_frustum_corners(near, far)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cropped_faces_contain_the_entire_angular_domain_and_filter_footprint() {
        for side in [64, 257, 512, 1551, 4096] {
            let scale = (side - 4) as f32 / side as f32;
            for degrees in [40_f32, 90., 91., 110., 140., 180., 210., 220.] {
                let theta = degrees.to_radians() * 0.5;
                let regions = regions(side, scale, theta);
                for (face, rotation) in super::super::runtime::orientations()
                    .iter()
                    .enumerate()
                    .take(5)
                {
                    for y in 0..=96 {
                        for x in 0..=96 {
                            let uv = Vec2::new(x as f32, y as f32) / 96.;
                            let local = (uv * 2. - Vec2::ONE) / scale * Vec2::new(1., -1.);
                            let d = *rotation * local.extend(-1.).normalize();
                            if (-d.z).clamp(-1., 1.).acos() > theta {
                                continue;
                            }
                            let p = uv * side as f32;
                            let region = regions[face];
                            // Source-square outer edges retain the existing guard;
                            // every new crop edge has ample filter context.
                            if region.min.x > 0 {
                                assert!(p.x >= region.min.x as f32 + 2.);
                            }
                            if region.min.y > 0 {
                                assert!(p.y >= region.min.y as f32 + 2.);
                            }
                            if region.max.x < side {
                                assert!(p.x <= region.max.x as f32 - 2.);
                            }
                            if region.max.y < side {
                                assert!(p.y <= region.max.y as f32 - 2.);
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn crop_preserves_rays_texel_density_and_shadow_cascade_frustum() {
        let side = 1552;
        let scale = 0.96;
        let full = PerspectiveProjection {
            fov: 2. * (1_f32 / scale).atan(),
            aspect_ratio: 1.,
            ..default()
        };
        let regions = regions(side, scale, std::f32::consts::FRAC_PI_2);
        let pixels: u32 = regions[..5].iter().map(|r| r.width() * r.height()).sum();
        assert!(pixels < side * side * 4);
        for region in regions.into_iter().take(5) {
            let p = FaceProjection {
                perspective: full.clone(),
                region,
                full_side: side,
            };
            assert_eq!(
                full.get_frustum_corners(-0.1, -20.),
                p.get_frustum_corners(-0.1, -20.)
            );
            for v in [Vec3::new(0.2, -0.7, -2.), Vec3::new(-0.4, 0.3, -1.)] {
                let pixels = |m: Mat4, size: Vec2| {
                    let ndc = m.project_point3(v);
                    (ndc.xy() * Vec2::new(1., -1.) + Vec2::ONE) * 0.5 * size
                };
                let old = pixels(full.get_clip_from_view(), Vec2::splat(side as f32));
                let new =
                    pixels(p.get_clip_from_view(), region.size().as_vec2()) + region.min.as_vec2();
                assert!(old.abs_diff_eq(new, 0.001));
            }
        }
    }
}
