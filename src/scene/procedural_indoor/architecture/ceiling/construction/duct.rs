//! Fabricated trunk, joints, takeoffs and fitted terminal faces.
use super::services::{route, suspended};
use super::*;

fn archived_grille(a: &mut Assembly, f: &Frame, p: Vec3, length: f32, width: f32) {
    f.block(
        a,
        Surface::Metal,
        p + Vec3::Y * 0.014,
        Vec3::new(length, 0.028, width),
        0.003,
    );
    // Recess, frame and separate angled louvers. The dark recess is raised
    // below the housing bottom instead of competing with its face.
    f.block(
        a,
        Surface::Ink,
        p - Vec3::Y * 0.002,
        Vec3::new(length - 0.012, 0.003, width - 0.012),
        0.,
    );
    let count = (length / 0.025).round().clamp(3., 12.) as usize;
    for i in 0..count {
        let x = (i as f32 + 0.5) / count as f32 * (length - 0.022) - (length - 0.022) * 0.5;
        f.part(a, Surface::Metal).cuboid(
            Vec3::new(0.008, 0.004, width - 0.018),
            0.001,
            Transform::from_translation(p + Vec3::new(x, -0.005, 0.))
                .with_rotation(Quat::from_rotation_z(0.3)),
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build(
    mesh: Option<&mut Assembly>,
    f: &Frame,
    section: f32,
    end_scale: f32,
    bend: f32,
    rise: f32,
    branches: u8,
    seam_pitch: f32,
    generator_version: u32,
) -> Vec<Vec3> {
    let round = f.m.system == CeilingSystem::RoundDuct;
    let mut rng = stream(f.m.seed, 0x4455435453454354);
    let radii = if round {
        Vec2::splat((f.height * 0.29).min(f.width * 0.24) * section)
    } else {
        let aspect = if rng.random_bool(0.40) {
            rng.random_range(0.93..1.07)
        } else {
            rng.random_range(0.45..1.85)
        };
        let width = (f.width * 0.27).min(f.height * 0.25 / aspect);
        Vec2::new(width * aspect, width) * section
    };
    let scale = |t: f32| 1. - (1. - end_scale) * ((t - 0.48) / 0.24).clamp(0., 1.);
    let supports = [0.14, 0.86].map(|t| (route(f, t, bend, rise), radii.x * scale(t)));
    suspended(mesh, f, supports, |a| {
        let rings: Vec<_> = (0..=16)
            .map(|i| {
                let t = i as f32 / 16.;
                (route(f, t, bend, rise), radii * scale(t))
            })
            .collect();
        loft(f.part(a, Surface::Metal), &rings, round);
        let joints = (f.length / seam_pitch).ceil().clamp(1., 24.) as usize;
        for i in 1..joints {
            let t = i as f32 / joints as f32;
            let p = route(f, t, bend, rise);
            let r = radii * scale(t) + Vec2::splat(0.001);
            let outline: Vec<_> = if round {
                (0..=24)
                    .map(|j| {
                        let v = Vec2::from_angle(j as f32 / 24. * std::f32::consts::TAU) * r;
                        p + Vec3::new(0., v.x, v.y)
                    })
                    .collect()
            } else {
                [
                    Vec2::new(1., 1.),
                    Vec2::new(-1., 1.),
                    Vec2::new(-1., -1.),
                    Vec2::new(1., -1.),
                    Vec2::new(1., 1.),
                ]
                .map(|v| p + Vec3::new(0., v.x * r.x, v.y * r.y))
                .to_vec()
            };
            f.tube(a, Surface::Chrome, &outline, 0.0025, 6);
        }
        if round && rng.random_bool(0.75) {
            let turns = (f.length / rng.random_range(0.18..0.40)).clamp(2., 22.);
            let count = (turns * 18.).ceil() as usize;
            let seam: Vec<_> = (0..=count)
                .map(|i| {
                    let t = i as f32 / count as f32;
                    let angle = t * turns * std::f32::consts::TAU;
                    let radius = radii.x * scale(t) + 0.0005;
                    route(f, t, bend, rise)
                        + Vec3::new(0., angle.cos() * radius, angle.sin() * radius)
                })
                .collect();
            f.tube(a, Surface::Metal, &seam, 0.0012, 4);
        } else if !round {
            let seam: Vec<_> = (0..=16)
                .map(|i| {
                    let t = i as f32 / 16.;
                    route(f, t, bend, rise)
                        + Vec3::new(0., -radii.x * scale(t), radii.y * scale(t) * 0.78)
                })
                .collect();
            f.tube(a, Surface::Metal, &seam, 0.0015, 4);
        }
        let stock = f.m.duct_register_stock(generator_version);
        for i in 0..branches {
            let site = stock.map(|s| s.site(f.m, branches, i));
            let t = site.map_or((i as f32 + 1.) / (branches as f32 + 1.), |s| s.station);
            let p = route(f, t, bend, rise);
            let z = site.map_or_else(
                || {
                    if i % 2 == 0 {
                        f.width * 0.39
                    } else {
                        -f.width * 0.39
                    }
                },
                |s| s.lateral_m,
            );
            let side = z.signum();
            let r = (radii.min_element() * scale(t) * 0.42).min(f.height * 0.10);
            let elbow = (p.y - 0.044).min((z - p.z).abs() * 0.40);
            let (points, joint, archived_face) = if elbow > r * 2.5 {
                let mut points = vec![p];
                for j in 0..=6 {
                    let angle = j as f32 / 6. * std::f32::consts::FRAC_PI_2;
                    points.push(Vec3::new(
                        p.x,
                        p.y - elbow * (1. - angle.cos()),
                        z - side * elbow * (1. - angle.sin()),
                    ));
                }
                // Finish at the tangent. A tiny vertical tail folds transported rings.
                let end = *points.last().unwrap();
                (points, end, end - Vec3::Y * 0.026)
            } else {
                // In a shallow plenum the boot seats directly beneath the branch.
                let end_z = if site.is_some() {
                    z + side * r * 0.66
                } else {
                    z
                };
                (
                    vec![p, p.with_z(end_z)],
                    Vec3::new(p.x, p.y - r, z),
                    Vec3::new(p.x, (p.y - r - 0.023).max(0.012), z),
                )
            };
            f.tube(a, Surface::Metal, &points, r, 12);
            if let Some((stock, site)) = stock.zip(site) {
                register::build(a, f, stock, site.size, joint, r, &rings);
            } else {
                archived_grille(
                    a,
                    f,
                    archived_face,
                    0.27_f32.min(f.length * 0.18),
                    f.width * 0.16,
                );
            }
        }
        if branches == 0 {
            let site = stock.map(|s| s.site(f.m, 0, 0));
            let t = site.map_or(0.40, |s| s.station);
            let p = route(f, t, bend, rise);
            let radius = radii.x * scale(t);
            if let Some((stock, site)) = stock.zip(site) {
                register::build(
                    a,
                    f,
                    stock,
                    site.size,
                    p - Vec3::Y * radius,
                    radii.min_element() * scale(t),
                    &rings,
                );
            } else {
                archived_grille(
                    a,
                    f,
                    p - Vec3::Y * (radius + 0.015),
                    0.36_f32.min(f.length * 0.30),
                    radii.y * 1.25,
                );
            }
        }
    })
}
