//! Pothos-like climbing and cascading growth. Trailing stems cross above the
//! rim before descending; wide nursery pots keep foliage above their rim.
use super::*;

pub(super) fn build(
    a: &mut Assembly,
    o: &IndoorObject,
    g: &Growth,
    base: Vec3,
    spread: f32,
    soil_radius: f32,
) {
    let mut r = stream(o.seed, 6453);
    let height = o.size.y - base.y;
    let climb = o.size.y / o.size.x.min(o.size.z) > 1.75 && r.random_bool(0.60);
    let (pot_radius, pot_height) = pot_dimensions(o, g);
    let pot_outer = pot_radius * PotProfile::sample(o.seed).outer_radius_factor().max(1.10);
    let trailing = pot_outer < spread * 0.52;
    let shoots = g.count(if climb { 8 } else { 11 });
    let stake = base + Vec3::Y * height * 0.86;
    if climb {
        stem(a, &[base, stake], o.size.y * 0.005);
    }
    for j in 0..shoots {
        let dir = radial(g.angle(j));
        let root = if climb {
            base.lerp(stake, 0.20 + 0.80 * g.node(j, shoots - 1))
        } else {
            base + dir * soil_radius * 0.12
        };
        let points = if climb {
            [
                root,
                root + dir * spread * 0.16 + Vec3::Y * height * 0.035,
                root + dir * spread * 0.36,
                root + dir * spread * 0.55 - Vec3::Y * height * 0.065,
                root + dir * spread * 0.66 - Vec3::Y * height * 0.10,
            ]
        } else {
            let rise = height * r.random_range(0.35..0.66);
            let end_y = if trailing {
                -base.y * 0.35
            } else {
                height * 0.10
            };
            [
                root,
                base + dir * spread * 0.18 + Vec3::Y * rise,
                base + dir * spread * 0.44 + Vec3::Y * rise * 0.75,
                base + dir * spread * 0.72 + Vec3::Y * height * 0.12,
                base + dir * spread * 0.83 + Vec3::Y * end_y,
            ]
        };
        stem(a, &points, o.size.y * 0.0022);
        let leaves = r.random_range(7..=10);
        for k in 0..leaves {
            let p = along(&points, 0.12 + 0.88 * k as f32 / (leaves - 1) as f32);
            // Alternate outward-facing leaves. The last leaves on a trailing
            // shoot stay outside the pot rather than cutting back through it.
            let d = radial(g.angle(j) + if k % 2 == 0 { 0.85 } else { -0.85 });
            let length = spread * r.random_range(0.23..0.30);
            let half_width = spread * r.random_range(0.115..0.15);
            let drop = (height * 0.045).min(base.y * 0.15);
            let mut root = p + d * spread * 0.025;
            // Broad bowls and tall cachepots need different rim clearance.
            // Account for curl, twist and droop before making a leaf, and grow
            // a connected upward petiole rather than clipping leaf vertices.
            let width = half_width * g.leaf_aspect;
            if root.xz().length() < pot_outer + width * 1.4 {
                let below = drop + width * (g.droop + 1.13 * g.leaf_curl + 0.34);
                root.y = root.y.max(pot_height + below + o.size.y * 0.002);
            }
            stem(a, &[p, root], o.size.y * 0.0009);
            blade(
                a,
                root,
                d * length - Vec3::Y * drop,
                half_width,
                4,
                r.random(),
                g,
            );
        }
    }
}
