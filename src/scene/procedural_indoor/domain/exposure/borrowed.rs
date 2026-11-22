//! Meter the two successive apertures between outdoors and the primary room.
//! Raster and baked transport use the actual geometry. This room-average estimate
//! accounts for fixed glazing and leaves the exposure independent of door motion.
use super::*;
use crate::scene::procedural_indoor::layout::AdjoiningSpace;

pub(super) fn apertures(scene: &IndoorManifest, sun: Vec3) -> Vec2 {
    let Some(w) = scene
        .adjoining_elevation
        .as_ref()
        .and_then(|p| p.windows.as_ref())
    else {
        return Vec2::ZERO;
    };
    let Some((a, b)) = w.endpoints(scene) else {
        return Vec2::ZERO;
    };
    let bay = scene.adjoining_space();
    let outline = bay.footprint(scene.room_size);
    let shared_z = scene.room_size.z * 0.5;
    let shared = architecture::interior_glazing::shared_facade(
        scene,
        architecture::interior_glazing::partition_height(scene),
    );
    let frame = polygon::wall_frame(a, b);
    let outward = frame.rotation * Vec3::NEG_Z;
    let incidence = outward.dot(sun).max(0.);
    let portals: Vec<_> = shared
        .openings
        .iter()
        .map(|o| {
            // Shared facade X runs opposite the room's X axis.
            let lo = Vec2::new(-o.max.x, o.min.y);
            let hi = Vec2::new(-o.min.x, o.max.y);
            (lo, hi, transmitted_opening(&shared, o, sun.z.max(0.)))
        })
        .collect();
    let mut total = Vec2::ZERO;
    for o in &w.facade.openings {
        let center = frame.transform_point(((o.min + o.max) * 0.5).extend(0.));
        let source = transmitted_opening(&w.facade, o, incidence);
        let beams = if incidence > 0. && sun.z > 0.0001 {
            solar_footprints(o, frame, shared_z, bay, sun)
        } else {
            Vec::new()
        };
        let mut admitted = Vec2::ZERO;
        for &(lo, hi, clear) in &portals {
            let mid = (lo + hi) * 0.5;
            let inner = mid.extend(shared_z);
            let delta = center - inner;
            let dir = delta.normalize();
            let projected = clear.x * dir.z.max(0.) * outward.dot(dir).max(0.);
            if projected > 0.
                && polygon::segment_inside(
                    &outline,
                    inner.lerp(center, 0.001).xz(),
                    center.lerp(inner, 0.001).xz(),
                    0.,
                )
            {
                // Bounded patch view-factor approximation. Diverges neither at
                // grazing angles nor when the receiver subtends a large solid angle.
                admitted.x += source.x * projected
                    / (std::f32::consts::PI * delta.length_squared() + projected);
            }
            if !beams.is_empty() {
                let overlap: f32 = beams
                    .iter()
                    .map(|p| polygon::area(&polygon::clip_box(p, lo, hi)).abs())
                    .sum();
                // Integrate solar energy on the shared wall, with the projected
                // area and clear glass/frame fraction of both installations.
                admitted.y += overlap * clear.y / (hi - lo).element_product() * source.y
                    / (o.area() * incidence);
            }
        }
        total += admitted.min(source);
    }
    total
}

fn solar_footprints(
    o: &WindowOpening,
    frame: Transform,
    shared_z: f32,
    bay: AdjoiningSpace,
    sun: Vec3,
) -> Vec<Vec<Vec2>> {
    let rect = [
        o.min,
        Vec2::new(o.max.x, o.min.y),
        o.max,
        Vec2::new(o.min.x, o.max.y),
    ];
    let at = |p: Vec2| frame.transform_point(p.extend(0.));
    let project = |p: Vec2| {
        let w = at(p);
        w.xy() - sun.xy() * ((w.z - shared_z) / sun.z)
    };
    let Some(ret) = bay.side_return else {
        return vec![rect.map(project).to_vec()];
    };
    let split_z = shared_z + bay.depth_m;
    let axis = (frame.rotation * Vec3::X).z;
    let offset = split_z - frame.translation.z;
    // A wall parallel to the partition belongs wholly to one side of the step.
    let (near, far) = if axis.abs() < 1e-5 {
        if offset >= -1e-5 {
            (rect.to_vec(), Vec::new())
        } else {
            (Vec::new(), rect.to_vec())
        }
    } else {
        (
            polygon::clip(&rect, Vec2::new(-axis, 0.), -offset),
            polygon::clip(&rect, Vec2::new(axis, 0.), offset),
        )
    };
    let mut pieces = Vec::with_capacity(2);
    if near.len() >= 3 {
        pieces.push(near.into_iter().map(project).collect());
    }
    if far.len() >= 3 {
        let far: Vec<_> = far.into_iter().map(project).collect();
        // A ray from the extension must cross the step on its occupied side.
        let sign = if ret.positive_x { 1. } else { -1. };
        let limit = ret.split_x_m - bay.depth_m * sun.x / sun.z;
        let visible = polygon::clip(&far, Vec2::X * sign, limit * sign);
        if visible.len() >= 3 {
            pieces.push(visible);
        }
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::layout::{AdjoiningReturn, IndoorLayout};

    #[test]
    fn direct_sun_cannot_cut_across_the_missing_corner_of_a_stepped_bay() {
        let opening = |width: f32| WindowOpening {
            min: Vec2::new(-width * 0.5, 1.8),
            max: Vec2::new(width * 0.5, 2.8),
            columns: 1,
            transom: 0.,
            transoms: vec![],
            operable: false,
            glazing: None,
        };
        for sign in [-1., 1.] {
            let bay = AdjoiningSpace {
                depth_m: 3.,
                side_return: Some(AdjoiningReturn {
                    split_x_m: 0.,
                    positive_x: sign > 0.,
                    extra_depth_m: 2.,
                }),
                ..default()
            };
            let projected_area = |o: &WindowOpening, frame, sun, bay| {
                solar_footprints(o, frame, 0., bay, sun)
                    .iter()
                    .map(|p| polygon::area(p).abs())
                    .sum::<f32>()
            };
            let rear = Transform::from_xyz(1.2 * sign, 0., 5.)
                .with_rotation(Quat::from_rotation_y(std::f32::consts::PI));
            let sun = Vec3::new(sign * 0.65, 0.12, 1.).normalize();
            // Over a 2 m extension the ray shifts 1.30 m horizontally. Of the
            // rear opening at |x|=0.20..2.20 only |x|=1.30..2.20 is visible.
            assert!((projected_area(&opening(2.), rear, sun, bay) - 0.90).abs() < 1e-5);
            assert!(
                (projected_area(&opening(2.), rear, sun, AdjoiningSpace::default()) - 2.).abs()
                    < 1e-5
            );
            let side = Transform::from_xyz(sign, 0., 3.)
                .with_rotation(Quat::from_rotation_y(-sign * std::f32::consts::FRAC_PI_2));
            let sun = Vec3::new(sign, 0.1, 1.).normalize();
            // The pane on the side wall spans z=1..5. At z>4 the ray reaches
            // the opaque return before it can reach the shared partition.
            assert!((projected_area(&opening(4.), side, sun, bay) - 3.).abs() < 1e-5);
        }
    }

    #[test]
    fn a_daylit_bay_can_supply_a_room_with_no_other_exterior_openings() {
        let mut scene =
            IndoorManifest::generate_with_humans(44316, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        for wall in &mut scene.envelope.as_mut().unwrap().walls {
            wall.facade = None;
        }
        scene.meter_indoor_lighting();
        let lit = scene.domain().unwrap().photometry.meter.unwrap();
        scene.adjoining_elevation = None;
        scene.meter_indoor_lighting();
        let enclosed = scene.domain().unwrap().photometry.meter.unwrap();
        assert!(lit.diffuse_daylight_lux > 0.);
        assert_eq!(lit.electric_lux, enclosed.electric_lux);
        assert_eq!(enclosed.diffuse_daylight_lux, 0.);
        assert_eq!(enclosed.direct_daylight_lux, 0.);
    }

    #[test]
    fn metering_requires_two_openings_and_scales_with_their_orientation() {
        let mut lit = 0;
        for seed in 44300..44348 {
            let mut scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.)
                    .unwrap();
            let sun = Vec3::new(0.18, 0.15, 1.).normalize();
            let area = apertures(&scene, sun);
            assert!(area.is_finite() && area.min_element() >= 0.);
            assert_eq!(apertures(&scene, -sun).y, 0.);
            lit += usize::from(area.x > 0. && area.y > 0.);
            scene.adjoining_elevation = None;
            assert_eq!(apertures(&scene, sun), Vec2::ZERO);
        }
        assert!(lit >= 2, "{lit} fixed solar sightlines");
    }
}
