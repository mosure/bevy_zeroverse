//! Backless and low-back stools share the human seating datum of task chairs.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct StoolProgram {
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
    let mut r = stream(o.seed, 6431);
    let base = r.random_range(0..3);
    StoolProgram {
        radius_fraction: r.random_range(0.30..0.38),
        seat_aspect: r.random_range(0.72..1.0),
        seat_roundness: r.random_range(2.0..7.0),
        padding_m: r.random_range(0.03..0.055),
        base,
        legs: r.random_range(3..=4),
        foot_spread: r.random_range(0.95..1.17),
        footrest_height_m: r.random_range(0.15..0.24),
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
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let radius = o.size.x.min(o.size.z) * p.radius_fraction;
    let half = Vec3::new(radius, 0., radius * p.seat_aspect);
    let outline: Vec<_> = (0..48)
        .map(|i| {
            let t = i as f32 * TAU / 48.;
            Vec2::new(
                t.cos().signum() * t.cos().abs().powf(2. / p.seat_roundness) * half.x,
                t.sin().signum() * t.sin().abs().powf(2. / p.seat_roundness) * half.z,
            )
        })
        .collect();
    if matches!(p.cover, Surface::Wood | Surface::Plastic) {
        a.part(p.cover, "chair").profile_slab(
            &outline,
            p.padding_m,
            0.006,
            Transform::from_xyz(0., 0.47 - p.padding_m * 0.5, 0.),
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
            Transform::from_xyz(0., 0.47 - (p.padding_m - dish) * 0.5, 0.),
        );
    }
    let underside = 0.47 - p.padding_m;
    let foot_radius = radius * p.foot_spread;
    match p.base {
        0 => {
            for i in 0..p.legs {
                let t = p.phase + i as f32 * TAU / p.legs as f32;
                let dir = Vec3::new(t.cos(), 0., t.sin());
                let foot = dir * foot_radius + Vec3::Y * 0.016;
                let top = dir * radius * 0.60 + Vec3::Y * underside;
                a.part(p.frame, "chair").rod(foot, top, 0.016);
                a.part(Surface::Rubber, "chair").cylinder(
                    0.020,
                    0.016,
                    Transform::from_translation(foot.with_y(0.008)),
                );
            }
            // The ring meets the actual splayed legs at this height.
            let t = (p.footrest_height_m - 0.016) / (underside - 0.016);
            ring(
                a,
                p.frame,
                foot_radius * (1. - t) + radius * 0.60 * t,
                p.footrest_height_m,
            );
        }
        1 => {
            a.part(p.frame, "chair").lathe(
                &[
                    (0., 0.),
                    (foot_radius, 0.),
                    (foot_radius, 0.015),
                    (foot_radius * 0.70, 0.035),
                    (0.032, 0.085),
                    (0.032, underside),
                    (0., underside),
                ],
                32,
                Transform::IDENTITY,
            );
            let rr = radius * 0.76;
            ring(a, p.frame, rr, p.footrest_height_m);
            for i in 0..3 {
                let angle = p.phase + i as f32 * TAU / 3.;
                a.part(p.frame, "chair").rod(
                    Vec3::Y * p.footrest_height_m,
                    Vec3::new(angle.cos() * rr, p.footrest_height_m, angle.sin() * rr),
                    0.010,
                );
            }
        }
        _ => {
            for side in [-1., 1.] {
                let x = side * radius * 0.66;
                let z = radius * p.seat_aspect * 0.65;
                let path = [
                    Vec3::new(x, underside, -z),
                    Vec3::new(x, 0.022, -z),
                    Vec3::new(x, 0.022, z),
                    Vec3::new(x, underside, z),
                ];
                a.part(p.frame, "chair").tube(&path, 0.016, 10);
                for zz in [-z, z] {
                    a.box_part(
                        Surface::Rubber,
                        "chair",
                        Vec3::new(x, 0.006, zz),
                        Vec3::new(0.042, 0.012, 0.042),
                        0.004,
                    );
                }
            }
            a.part(p.frame, "chair").rod(
                Vec3::new(
                    -radius * 0.66,
                    p.footrest_height_m,
                    radius * p.seat_aspect * 0.65,
                ),
                Vec3::new(
                    radius * 0.66,
                    p.footrest_height_m,
                    radius * p.seat_aspect * 0.65,
                ),
                0.012,
            );
        }
    }
    if !is_backless(o) {
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

fn ring(a: &mut Assembly, material: Surface, r: f32, h: f32) {
    let path: Vec<_> = (0..=40)
        .map(|i| {
            let t = (i % 40) as f32 * TAU / 40.;
            Vec3::new(t.cos() * r, h, t.sin() * r)
        })
        .collect();
    a.part(material, "chair").tube(&path, 0.009, 8);
}
