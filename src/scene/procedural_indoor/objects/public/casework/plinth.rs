//! Plinth joinery varies below the exact full-area exhibit attachment plane.
use super::*;

#[derive(Clone, Debug, serde::Serialize)]
pub struct PlinthProgram {
    pub structure: u8,
    pub base: Vec2,
    pub top: Vec2,
    pub waist: Vec2,
    pub waist_height: f32,
    pub base_height: f32,
    pub cap_height: f32,
    pub roundness: f32,
    pub blade_fraction: f32,
    pub stretcher_height: f32,
    pub rotate_base: bool,
    pub finish: Surface,
    pub cap_finish: Surface,
}
pub fn parameters(o: &IndoorObject) -> PlinthProgram {
    let mut r = stream(o.seed, 0x504c494e54485052);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x504c494e54484649);
    let finish = [
        Surface::Wood,
        Surface::Metal,
        Surface::Paint,
        Surface::Concrete,
        Surface::Plastic,
    ][f.random_range(0..5)];
    PlinthProgram {
        structure: r.random_range(0..6),
        base: Vec2::new(r.random_range(0.68..0.97), r.random_range(0.68..0.97)),
        top: Vec2::new(r.random_range(0.40..0.91), r.random_range(0.40..0.91)),
        waist: Vec2::new(r.random_range(0.23..0.59), r.random_range(0.23..0.59)),
        waist_height: r.random_range(0.20..0.72),
        base_height: r.random_range(0.018_f32..0.075).min(o.size.y * 0.13),
        cap_height: r.random_range(0.014_f32..0.055).min(o.size.y * 0.15),
        roundness: r.random_range(2.0..14.),
        blade_fraction: r.random_range(0.055..0.17),
        stretcher_height: r.random_range(0.14..0.58),
        rotate_base: r.random_bool(0.5),
        finish,
        cap_finish: if f.random_bool(0.28) {
            [Surface::Wood, Surface::Paint, Surface::Concrete][f.random_range(0..3)]
        } else {
            finish
        },
    }
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let l = o.kind.class_name();
    let top = s.y - p.cap_height;
    let base = s.xz() * p.base * 0.5;
    let crown = s.xz() * p.top * 0.5;
    match p.structure {
        0 | 1 => {
            let waist = s.xz() * p.waist * 0.5;
            let rings = if p.structure == 0 {
                vec![(0., base), (p.base_height, base), (top, crown)]
            } else {
                vec![
                    (0., base),
                    (p.base_height, base),
                    (top * p.waist_height, waist),
                    (top, crown),
                ]
            };
            a.part(p.finish, l)
                .housing(&rings, p.roundness, Transform::IDENTITY);
        }
        2 => {
            let radius = (s.x.min(s.z) * p.blade_fraction * 0.23).min(p.cap_height * 0.8);
            for x in [-1., 1.] {
                for z in [-1., 1.] {
                    let foot = Vec3::new(x * base.x * 0.87, radius, z * base.y * 0.87);
                    let end = Vec3::new(x * crown.x, top, z * crown.y);
                    a.part(p.finish, l).rod(foot, end, radius);
                    a.part(Surface::Rubber, l).cylinder(
                        radius * 1.25,
                        radius,
                        Transform::from_translation(foot.with_y(radius * 0.5)),
                    );
                }
            }
            let y = p.stretcher_height * top;
            let extent = base * 0.87 + (crown - base * 0.87) * ((y - radius) / (top - radius));
            for z in [-1., 1.] {
                a.part(p.finish, l).rod(
                    Vec3::new(-extent.x, y, z * extent.y),
                    Vec3::new(extent.x, y, z * extent.y),
                    radius * 0.65,
                );
            }
        }
        3 | 4 => {
            a.box_part(
                p.finish,
                l,
                Vec3::Y * p.base_height * 0.5,
                Vec3::new(base.x * 2., p.base_height, base.y * 2.),
                0.002,
            );
            let height = top - p.base_height;
            let blade = s.x.min(s.z) * p.blade_fraction;
            let y = p.base_height + height * 0.5;
            if p.structure == 3 {
                for side in [-1., 1.] {
                    let center = if p.rotate_base {
                        Vec3::new(0., y, side * base.y * 0.65)
                    } else {
                        Vec3::new(side * base.x * 0.65, y, 0.)
                    };
                    let size = if p.rotate_base {
                        Vec3::new(base.x * 1.7, height, blade)
                    } else {
                        Vec3::new(blade, height, base.y * 1.7)
                    };
                    a.box_part(p.finish, l, center, size, 0.002);
                }
            } else {
                // One continuous spine plus two non-overlapping arms; no
                // coplanar front faces of two complete crossing slabs.
                a.box_part(
                    p.finish,
                    l,
                    Vec3::Y * y,
                    Vec3::new(blade, height, base.y * 1.8),
                    0.002,
                );
                let arm = (base.x * 1.8 - blade) * 0.5;
                for side in [-1., 1.] {
                    a.box_part(
                        p.finish,
                        l,
                        Vec3::new(side * (blade + arm) * 0.5, y, 0.),
                        Vec3::new(arm, height, blade),
                        0.002,
                    );
                }
            }
        }
        _ => {
            // Recessed toe and three independently proportioned stepped tiers.
            a.box_part(
                p.finish,
                l,
                Vec3::Y * p.base_height * 0.5,
                Vec3::new(base.x * 1.7, p.base_height, base.y * 1.7),
                0.002,
            );
            let h = (top - p.base_height) * p.waist_height;
            a.box_part(
                p.finish,
                l,
                Vec3::Y * (p.base_height + h * 0.5),
                Vec3::new(base.x * 2., h, base.y * 2.),
                0.002,
            );
            let upper = top - p.base_height - h;
            a.box_part(
                p.finish,
                l,
                Vec3::Y * (top - upper * 0.5),
                Vec3::new(crown.x * 2., upper, crown.y * 2.),
                0.002,
            );
        }
    }
    a.box_part(
        p.cap_finish,
        l,
        Vec3::Y * (s.y - p.cap_height * 0.5),
        Vec3::new(s.x, p.cap_height, s.z),
        0.002,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::objects::{audit, build_object};
    #[test]
    fn plinth_construction_varies_with_exact_supported_flat_top() {
        let mut structures = std::collections::BTreeSet::new();
        let mut caps = std::collections::BTreeSet::new();
        for seed in 0..256 {
            let o = audit::fixture(ObjectKind::ExhibitPlinth, seed);
            let p = parameters(&o);
            structures.insert(p.structure);
            caps.insert((p.cap_height * 1000.) as u32);
            let a = build_object(&o);
            audit::check(&o, &a).unwrap();
            let (lo, hi) = a.bounds();
            assert!(lo.y.abs() < 1e-5 && (hi.y - o.size.y).abs() < 1e-5);
        }
        assert_eq!(structures.len(), 6);
        assert!(caps.len() > 32);
    }
}
