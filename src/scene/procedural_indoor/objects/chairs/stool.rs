//! Counter, dining and low stools fitted to their support surface.
use super::*;
mod base;
mod fit;
#[cfg(test)]
mod tests;
pub(super) use fit::clearance_boxes;
pub use fit::SeatedSupport;
pub(crate) use fit::{fitted_envelope, same_floor, FIT_VERSION};

#[derive(Debug, Clone, serde::Serialize)]
pub struct StoolProgram {
    pub seat_height_m: f32,
    pub back_gap_m: f32,
    pub back_height_m: f32,
    pub back_curve_m: f32,
    pub radius_fraction: f32,
    pub seat_aspect: f32,
    pub seat_roundness: f32,
    pub padding_m: f32,
    pub base: u8,
    pub legs: u32,
    pub foot_spread: f32,
    pub footrest_height_m: f32,
    pub phase: f32,
    pub cover: Surface,
    pub frame: Surface,
}

pub fn parameters(o: &IndoorObject) -> StoolProgram {
    parameters_at(o, bevy_zeroverse_capture::GENERATOR_VERSION)
}

pub(crate) fn parameters_at(o: &IndoorObject, version: u32) -> StoolProgram {
    let mut r = stream(o.seed, 6431);
    let base = r.random_range(0..3);
    let current = version >= FIT_VERSION;
    let height = if current { fit::seat(o) } else { 0.47 };
    let mut back = stream(o.seed, 0x5354_4241_434b);
    let gap = if current {
        back.random_range(0.066..0.15)
    } else {
        0.13
    };
    let mut p = StoolProgram {
        seat_height_m: height,
        back_gap_m: gap,
        back_height_m: if current {
            back.random_range(0.105..(0.305_f32 - gap).min(0.213))
        } else {
            0.14
        },
        back_curve_m: if current {
            back.random_range(0.025..0.060)
        } else {
            0.045
        },
        radius_fraction: r.random_range(0.30..0.38),
        seat_aspect: r.random_range(0.72..1.0),
        seat_roundness: r.random_range(2.0..7.0),
        padding_m: r.random_range(0.03..0.055),
        base,
        legs: r.random_range(3..=4),
        foot_spread: r.random_range(0.95..1.17),
        footrest_height_m: r.random_range(0.15..0.24) + (height - 0.47) * 0.88,
        phase: r.random_range(0.0..TAU),
        cover: [
            Surface::Wood,
            Surface::Leather,
            Surface::FabricAlt,
            Surface::Plastic,
        ][r.random_range(0..4)],
        frame: if base == 0 && r.random_bool(0.55) {
            Surface::WoodEdge
        } else {
            [Surface::Chrome, Surface::Metal][r.random_range(0..2)]
        },
    };
    if current {
        let thickness = (p.padding_m - 0.030) / 0.025;
        p.padding_m = match p.cover {
            Surface::Wood => 0.019 + 0.017 * thickness,
            Surface::Plastic => 0.018 + 0.020 * thickness,
            _ => 0.039 + 0.028 * thickness,
        };
    }
    p
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    let p = parameters_at(o, version);
    let radius = o.size.x.min(o.size.z) * p.radius_fraction;
    let half = Vec3::new(radius, 0., radius * p.seat_aspect);
    if matches!(p.cover, Surface::Wood | Surface::Plastic) {
        let outline: Vec<_> = (0..48)
            .map(|i| {
                let t = i as f32 * TAU / 48.;
                Vec2::new(
                    t.cos().signum() * t.cos().abs().powf(2. / p.seat_roundness) * half.x,
                    t.sin().signum() * t.sin().abs().powf(2. / p.seat_roundness) * half.z,
                )
            })
            .collect();
        a.part(p.cover, "chair").profile_slab(
            &outline,
            p.padding_m,
            0.006,
            Transform::from_xyz(0., p.seat_height_m - p.padding_m * 0.5, 0.),
        );
    } else {
        // The compressed center stays on the human seating datum; increasing
        // the crown by the dish depth keeps its underside on the supports.
        let dish = 0.004;
        a.part(p.cover, "chair").cushion(
            Vec3::new(half.x * 2., p.padding_m + dish, half.z * 2.),
            super::super::super::geometry::CushionProfile {
                roundness: p.seat_roundness,
                crown: 3.5,
                dish,
            },
            Transform::from_xyz(0., p.seat_height_m - (p.padding_m - dish) * 0.5, 0.),
        );
    }
    let underside = p.seat_height_m - p.padding_m;
    base::build(a, &p, radius, underside, version);
    if version >= FIT_VERSION && !is_backless(o) {
        let z = radius * p.seat_aspect * 0.70;
        let bottom = p.seat_height_m + p.back_gap_m;
        let h = p.back_height_m;
        let t = 0.70;
        let post_x = radius * 0.80 * t;
        // Terminate both posts inside the rear skin at panel mid-height.
        let socket_z = z + p.back_curve_m * (1. - t * t) + 0.5 * h * 0.045 + 0.014;
        for side in [-1., 1.] {
            a.part(p.frame, "chair").rod(
                Vec3::new(side * post_x, underside, z * 0.83),
                Vec3::new(side * post_x, bottom + h * 0.5, socket_z),
                0.012,
            );
        }
        a.part(p.cover, "chair").chair_back_contour(
            radius * 1.6,
            h,
            [p.back_curve_m, 0., 0.045, 0.025, 0., 0.],
            Transform::from_xyz(0., bottom, z),
        );
    } else if !is_backless(o) {
        for side in [-1., 1.] {
            a.part(p.frame, "chair").rod(
                Vec3::new(side * radius * 0.60, underside, radius * 0.55),
                Vec3::new(side * radius * 0.60, 0.68, radius * 0.76),
                0.012,
            );
        }
        a.part(p.cover, "chair").chair_back_contour(
            radius * 1.5,
            0.14,
            [0.045, 0., 0.03, 0.023, 0.006, 0.],
            Transform::from_xyz(0., 0.60, radius * 0.76),
        );
    }
}
