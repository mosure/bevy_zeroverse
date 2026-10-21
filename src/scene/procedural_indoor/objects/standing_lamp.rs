//! Floor lamp supports, shades and emitter share one metric program.
use super::*;

/// Round base and shade proportions remain coupled, but not to one catalog size.
pub fn dimensions(seed: u64) -> Vec3 {
    let mut r = stream(seed, 0x4c41_4d50_5349_5a45);
    let diameter = r.random_range(0.30..0.78);
    Vec3::new(diameter, r.random_range(1.08..2.12), diameter)
}
#[derive(Debug, Clone, serde::Serialize)]
pub struct LampProgram {
    pub support: u8,
    pub shade_height: f32,
    pub shade_taper: f32,
    pub shade_radius_fraction: f32,
    pub stem_offset: f32,
    pub stem_depth: f32,
    pub base_fraction: f32,
    pub stem_radius: f32,
    pub bulb_radius: f32,
    pub shade_sides: usize,
    pub shade: luminaires::ShadeProgram,
}
pub fn parameters(o: &IndoorObject) -> LampProgram {
    let mut r = stream(o.seed, 7420);
    LampProgram {
        support: r.random_range(0..4),
        shade_height: r.random_range(0.15_f32..0.44).min(o.size.y * 0.28),
        shade_taper: r.random_range(0.42..1.12),
        shade_radius_fraction: r.random_range(0.26..0.46),
        stem_offset: r.random_range(-0.12..0.12),
        shade_sides: if r.random_bool(0.25) {
            r.random_range(5..=8)
        } else {
            32
        },
        shade: luminaires::parameters(o.seed),
        stem_depth: r.random_range(-0.12..0.12),
        base_fraction: r.random_range(0.25..0.39),
        stem_radius: r.random_range(0.006..0.014),
        bulb_radius: r
            .random_range(0.018_f32..0.040)
            .min(o.size.x.min(o.size.z) * 0.12),
    }
}
pub fn emitter(o: &IndoorObject) -> Vec3 {
    let p = parameters(o);
    o.transform().transform_point(Vec3::new(
        o.size.x * p.stem_offset,
        o.size.y - p.shade_height * 0.62,
        o.size.z * p.stem_depth,
    ))
}
pub(crate) fn emitting_area(o: &IndoorObject) -> f32 {
    let p = parameters(o);
    let r = luminaires::bulb_radii(&p.shade, p.bulb_radius);
    // Knud Thomsen ellipsoid approximation (below 1.1% error for these bulbs).
    let q = (Vec3::new(r.x * r.y, r.x * r.z, r.y * r.z)).powf(1.6075);
    4. * PI * (q.element_sum() / 3.).powf(1. / 1.6075)
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let x = s.x * p.stem_offset;
    let z = s.z * p.stem_depth;
    let r = (s.x.min(s.z) * p.shade_radius_fraction)
        .min((s.x * 0.5 - x.abs()).min(s.z * 0.5 - z.abs()) / p.shade_taper.max(1.));
    let top = s.y - 0.003;
    let bulb = Vec3::new(x, s.y - p.shade_height * 0.62, z);
    let root =
        bulb - Vec3::Y * (luminaires::bulb_radii(&p.shade, p.bulb_radius).y + p.bulb_radius * 0.42);
    if p.support == 0 || p.support == 3 {
        a.part(p.shade.trim, "lamp").lathe(
            &[
                (0., 0.),
                (s.x.min(s.z) * p.base_fraction, 0.),
                (s.x.min(s.z) * p.base_fraction * 0.94, 0.024),
                (0., 0.024),
            ],
            p.shade_sides as u32,
            Transform::IDENTITY,
        );
        if p.support == 3 {
            let points: Vec<_> = (0..=16)
                .map(|i| {
                    let t = i as f32 / 16.;
                    (Vec3::Y * 0.024).lerp(root, t) + Vec3::Z * (s.z * 0.25 * (PI * t).sin())
                })
                .collect();
            a.part(p.shade.trim, "lamp")
                .tube(&points, p.stem_radius, 10);
        } else {
            a.part(p.shade.trim, "lamp")
                .rod(Vec3::Y * 0.024, root, p.stem_radius);
        }
    } else {
        let n = if p.support == 1 { 3 } else { 4 };
        for i in 0..n {
            let t = TAU * (i as f32 + 0.25) / n as f32;
            let foot = Vec3::new(s.x * 0.38 * t.cos(), 0.01, s.z * 0.38 * t.sin());
            a.part(
                if p.support == 1 {
                    Surface::Wood
                } else {
                    Surface::Metal
                },
                "lamp",
            )
            .rod(foot, root, 0.012);
            a.part(Surface::Rubber, "lamp").ellipsoid(
                Vec3::new(0.015, 0.01, 0.015),
                Transform::from_translation(foot),
            );
        }
    }
    let bottom = s.y - p.shade_height;
    let upper = r * if matches!(p.shade.shape, 1 | 3) {
        p.shade_taper.clamp(0.88, 1.)
    } else {
        p.shade_taper
    };
    luminaires::shade(
        a,
        "lamp",
        &p.shade,
        Vec3::new(x, bottom, z),
        r,
        p.shade_height,
        p.shade_taper,
    );
    luminaires::bulb(a, "lamp", &p.shade, bulb, p.bulb_radius);
    for side in [-1., 1.] {
        a.part(Surface::Metal, "lamp").tube(
            &[
                root,
                root + Vec3::X * side * 0.038,
                Vec3::new(x + side * 0.038, top - 0.02, z),
                Vec3::new(x, top, z),
            ],
            0.002,
            6,
        );
        a.part(Surface::Metal, "lamp").rod(
            Vec3::new(x, top, z),
            Vec3::new(x + side * (upper - 0.001), top, z),
            0.002,
        );
    }
}
