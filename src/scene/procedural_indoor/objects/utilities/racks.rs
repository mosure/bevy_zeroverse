//! Independent garment-rack structure, stable bases and usable hooks/hangers.
use super::*;

#[derive(Clone, Debug, serde::Serialize)]
pub struct Hook {
    pub angle: f32,
    pub root_height: f32,
    pub reach: f32,
    pub rise: f32,
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct RackProgram {
    pub rail: bool,
    pub radius: f32,
    pub height: f32,
    pub upright: f32,
    pub feet: usize,
    pub foot_phase: f32,
    pub foot_span: Vec2,
    pub base_height: f32,
    pub disk: bool,
    pub hooks: Vec<Hook>,
    pub rail_width: f32,
    pub hanger_x: Vec<f32>,
    pub hanger_width: f32,
    pub finish: Surface,
    pub tips: Surface,
}

pub(crate) fn dimensions(r: &mut impl Rng) -> Vec3 {
    if r.random_bool(0.38) {
        Vec3::new(
            r.random_range(0.82..1.28),
            r.random_range(1.42..1.90),
            r.random_range(0.43..0.65),
        )
    } else {
        let width = r.random_range(0.43..0.73);
        Vec3::new(
            width,
            r.random_range(1.42..1.96),
            width * r.random_range(0.90..1.10),
        )
    }
}

pub fn parameters(o: &IndoorObject) -> RackProgram {
    let mut r = stream(o.seed, 0x434f41545241434b);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x5241434b46494e49);
    let s = o.size;
    let rail = s.x > s.z * 1.35 || r.random_bool(0.26);
    let radius = s.x.min(s.z) * 0.45;
    let upright = (radius * r.random_range(0.035..0.065)).min(s.y * 0.025);
    let height = s.y * r.random_range(0.91..0.97);
    let count = r.random_range(3..=7);
    let phase = r.random_range(0.0..TAU);
    let hooks = if rail {
        Vec::new()
    } else {
        (0..count)
            .map(|i| Hook {
                angle: phase + TAU * (i as f32 + r.random_range(-0.16..0.16)) / count as f32,
                root_height: height * r.random_range(0.57..0.83),
                reach: radius * r.random_range(0.57..0.84),
                rise: height * r.random_range(0.075..0.12),
            })
            .collect()
    };
    let rail_width = s.x * r.random_range(0.71..0.84);
    let hanger_width = (rail_width * r.random_range(0.33..0.58)).min(s.y * 0.27);
    let mut hanger_x: Vec<f32> = Vec::new();
    if rail {
        // Bounded random sequential admission, retaining usable hook separation.
        for _ in 0..r.random_range(0..=5) {
            for _ in 0..8 {
                let x = r.random_range(-0.40..0.40) * rail_width;
                if hanger_x.iter().all(|v| (v - x).abs() > upright * 5.) {
                    hanger_x.push(x);
                    break;
                }
            }
        }
    }
    RackProgram {
        rail,
        radius,
        height,
        upright,
        feet: r.random_range(3..=5),
        foot_phase: r.random_range(0.0..TAU),
        foot_span: s.xz() * r.random_range(0.34..0.43),
        base_height: s.y * r.random_range(0.065..0.12),
        disk: r.random_bool(0.42),
        hooks,
        rail_width,
        hanger_x,
        hanger_width,
        finish: [Surface::Wood, Surface::Metal, Surface::Chrome][f.random_range(0..3)],
        tips: if f.random_bool(0.6) {
            Surface::Rubber
        } else {
            Surface::Wood
        },
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let l = o.kind.class_name();
    let foot = p.upright * 1.15;
    if p.rail {
        // The long dimension hosts the rail; hangers hang in the YZ plane.
        for side in [-1., 1.] {
            let x = side * p.rail_width * 0.5;
            a.part(p.finish, l).rod(
                Vec3::new(x, foot, 0.),
                Vec3::new(x, p.height, 0.),
                p.upright,
            );
            a.part(p.finish, l).rod(
                Vec3::new(x, foot, -p.foot_span.y),
                Vec3::new(x, foot, p.foot_span.y),
                foot * 0.8,
            );
            for end in [-1., 1.] {
                a.part(Surface::Rubber, l).ellipsoid(
                    Vec3::new(foot * 1.5, foot, foot * 1.5),
                    Transform::from_xyz(x, foot, end * p.foot_span.y),
                );
            }
        }
        a.part(p.finish, l).rod(
            Vec3::new(-p.rail_width * 0.5, p.height, 0.),
            Vec3::new(p.rail_width * 0.5, p.height, 0.),
            p.upright,
        );
        a.part(p.finish, l).rod(
            Vec3::new(-p.rail_width * 0.5, p.base_height, 0.),
            Vec3::new(p.rail_width * 0.5, p.base_height, 0.),
            p.upright * 0.65,
        );
        let w = p.hanger_width.min(o.size.z * 0.78);
        // Thin bent wire stays well below the minimum shoulder bend radius.
        // Scaling it like a structural rack tube folds the inside of the bend.
        let wire = (p.upright * 0.22).min(w * 0.0035);
        for &x in &p.hanger_x {
            let hook_radius = p.upright + wire * 1.25;
            // Contact at the top of the horizontal rail. The arc ends with a
            // downward tangent, avoiding a near reversal at the straight stem.
            let center_y = p.height - wire * 0.25;
            let neck = p.height - hook_radius * 3.;
            let mut path: Vec<_> = (0..=18)
                .map(|i| {
                    let t = 0.68 * PI - i as f32 / 18. * 1.18 * PI;
                    Vec3::new(x, center_y + hook_radius * t.cos(), hook_radius * t.sin())
                })
                .collect();
            let corners = [
                Vec3::new(x, neck, -hook_radius),
                Vec3::new(x, neck - w * 0.26, -hook_radius - w * 0.5),
                Vec3::new(x, neck - w * 0.26, -hook_radius + w * 0.5),
            ];
            let mut outline = Vec::with_capacity(34);
            for i in 0..3 {
                let a = corners[i].lerp(corners[(i + 2) % 3], 0.18);
                let b = corners[i];
                let c = corners[i].lerp(corners[(i + 1) % 3], 0.18);
                for j in 0..=10 {
                    let t = j as f32 / 10.;
                    outline.push(a * (1. - t).powi(2) + b * (2. * t * (1. - t)) + c * t * t);
                }
            }
            path.push(outline[5]);
            outline.push(outline[0]);
            a.part(Surface::Metal, l).tube(&path, wire, 8);
            a.part(Surface::Metal, l).tube(&outline, wire, 8);
        }
    } else {
        a.part(p.finish, l)
            .rod(Vec3::Y * foot, Vec3::Y * p.height, p.upright);
        if p.disk {
            a.part(p.finish, l).housing(
                &[
                    (0., p.foot_span * 0.92),
                    (foot * 1.5, p.foot_span),
                    (foot * 2., p.foot_span * 0.85),
                ],
                2.,
                Transform::IDENTITY,
            );
        } else {
            for i in 0..p.feet {
                let t = p.foot_phase + TAU * i as f32 / p.feet as f32;
                let end = Vec3::new(t.cos() * p.foot_span.x, foot, t.sin() * p.foot_span.y);
                a.part(p.finish, l)
                    .rod(Vec3::Y * p.base_height, end, p.upright * 0.8);
                a.part(Surface::Rubber, l).ellipsoid(
                    Vec3::new(foot * 1.4, foot, foot * 1.4),
                    Transform::from_translation(end),
                );
            }
        }
        for h in &p.hooks {
            let d = Vec3::new(h.angle.cos(), 0., h.angle.sin());
            let path = [
                Vec3::Y * h.root_height,
                d * h.reach * 0.68 + Vec3::Y * (h.root_height + h.rise * 0.35),
                d * h.reach + Vec3::Y * (h.root_height + h.rise),
            ];
            a.part(p.finish, l).tube(&path, p.upright * 0.65, 8);
            a.part(p.tips, l).ellipsoid(
                Vec3::splat(p.upright * 1.05),
                Transform::from_translation(path[2]),
            );
        }
    }
}
