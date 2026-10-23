//! A mounted camera's downward overview must contain useful room geometry.
//! The >180-degree sensor periphery legitimately sees upper walls and ceiling;
//! counting it against floor coverage biases admission against circular lenses.
use super::*;
use crate::scene::procedural_indoor::{
    cameras::{coverage::VisibilityView, multiview::CHECK_TIMES},
    layout::IndoorManifest,
    program::region::PrimaryRegion,
};

pub(super) fn suitable(
    scene: &IndoorManifest,
    camera: &IndoorCamera,
    views: &[VisibilityView],
) -> bool {
    if views.len() != CHECK_TIMES.len() {
        return false;
    }
    let region = scene.primary_region();
    let normal = camera.ceiling_mount.as_ref().unwrap().inward_normal;
    CHECK_TIMES.into_iter().zip(views).all(|(t, view)| {
        overview(
            &region,
            camera.transform_at(t).translation,
            normal,
            view.surfaces(),
        )
    })
}

fn overview(
    region: &PrimaryRegion<'_>,
    eye: Vec3,
    inward: Vec3,
    points: impl Iterator<Item = Vec3>,
) -> bool {
    let mut hits = 0;
    let mut below = 0;
    let mut lo = Vec2::splat(f32::INFINITY);
    let mut hi = Vec2::splat(f32::NEG_INFINITY);
    for point in points {
        // A common central 140-degree cone measures overview quality for both
        // circular and cropped sensors. No capture rays or overlap samples are
        // discarded: this is solely the mount-content admission predicate.
        if (point - eye).normalize_or_zero().dot(inward) < 70_f32.to_radians().cos() {
            continue;
        }
        hits += 1;
        if point.y < eye.y - 0.7 && region.contains_projection_with_tolerance(point.xz(), 1e-4) {
            below += 1;
            lo = lo.min(point.xz());
            hi = hi.max(point.xz());
        }
    }
    hits >= 12 && below * 2 >= hits && (hi - lo).min_element() >= 1.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::layout::IndoorLayout;

    #[test]
    fn peripheral_ceiling_is_valid_context_but_obstructed_overviews_fail() {
        let mut scene =
            IndoorManifest::generate_with_humans(3, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        scene.program = None;
        scene.envelope = None;
        scene.room_size = Vec3::new(8., 4., 8.);
        let region = scene.primary_region();
        let eye = Vec3::new(0., 3.8, 0.);
        let floor: Vec<_> = (0..4)
            .flat_map(|x| (0..4).map(move |z| Vec3::new(x as f32 - 1.5, 0., z as f32 - 1.5)))
            .collect();
        let context = (0..80).map(|i| {
            let angle = i as f32 * std::f32::consts::TAU / 80.;
            eye + Vec3::new(angle.cos() * 3., 0.1, angle.sin() * 3.)
        });
        assert!(overview(
            &region,
            eye,
            Vec3::NEG_Y,
            floor.iter().copied().chain(context)
        ));
        assert!(!overview(
            &region,
            eye,
            Vec3::NEG_Y,
            floor.iter().map(|p| eye + (*p - eye) * 0.1)
        ));
        assert!(!overview(
            &region,
            eye,
            Vec3::NEG_Y,
            floor.iter().map(|p| p.with_y(0.) + Vec3::X * 20.)
        ));
        assert!(!overview(
            &region,
            eye,
            Vec3::NEG_Y,
            floor.iter().take(11).copied()
        ));
        assert!(!overview(
            &region,
            eye,
            Vec3::NEG_Y,
            floor.iter().map(|p| *p * 0.1)
        ));
    }
}
