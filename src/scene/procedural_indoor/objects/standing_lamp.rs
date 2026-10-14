//! Floor lamp supports, shades and emitter share one metric program.
use super::*;
#[derive(Debug, Clone, serde::Serialize)]
pub struct LampProgram {
    pub support: u8,
    pub shade_height: f32,
    pub shade_taper: f32,
    pub shade_radius_fraction: f32,
    pub stem_offset: f32,
    pub shade_sides: usize,
}
pub fn parameters(o: &IndoorObject) -> LampProgram {
    let mut r = stream(o.seed, 7420);
    LampProgram {
        support: r.random_range(0..3),
        shade_height: r.random_range(0.19_f32..0.35).min(o.size.y * 0.26),
        shade_taper: r.random_range(0.50..1.03),
        shade_radius_fraction: r.random_range(0.31..0.44),
        stem_offset: r.random_range(-0.10..0.10),
        shade_sides: if r.random_bool(0.25) {
            r.random_range(5..=8)
        } else {
            32
        },
    }
}
pub fn emitter(o: &IndoorObject) -> Vec3 {
    let p = parameters(o);
    o.transform().transform_point(Vec3::new(
        o.size.x * p.stem_offset,
        o.size.y - p.shade_height * 0.62,
        0.,
    ))
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let x = s.x * p.stem_offset;
    let r = (s.x.min(s.z) * p.shade_radius_fraction)
        .min((s.x * 0.5 - x.abs()).min(s.z * 0.5) / p.shade_taper.max(1.));
    let top = s.y - p.shade_height * 0.12;
    let root = Vec3::new(x, top, 0.);
    if p.support == 0 {
        a.part(Surface::Metal, "lamp").cylinder(
            s.x.min(s.z) * 0.32,
            0.024,
            Transform::from_xyz(0., 0.012, 0.),
        );
        a.part(Surface::Chrome, "lamp")
            .rod(Vec3::Y * 0.024, root, 0.010);
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
    let upper = r * p.shade_taper;
    let t = 0.003;
    a.part(Surface::FabricAlt, "lamp").lathe(
        &[
            (r, bottom),
            (upper, s.y),
            (upper - t, s.y),
            (r - t, bottom),
            (r, bottom),
        ],
        p.shade_sides as u32,
        Transform::from_xyz(x, 0., 0.),
    );
    let bulb = Vec3::new(x, s.y - p.shade_height * 0.62, 0.);
    a.part(Surface::Light, "lamp")
        .ellipsoid(Vec3::splat(0.027), Transform::from_translation(bulb));
    for side in [-1., 1.] {
        a.part(Surface::Metal, "lamp").rod(
            Vec3::new(x, top, 0.),
            Vec3::new(x + side * upper * 0.85, top, 0.),
            0.002,
        );
    }
}
