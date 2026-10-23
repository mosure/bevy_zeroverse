//! Connected woody crowns. Secondary shoots fill volume rather than placing
//! isolated leaves on a set of parallel poles. Dimensions use the admitted
//! canopy reserve, never a post-build stretch or a vertex clamp.
use super::*;

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, g: &Growth, base: Vec3, spread: f32) {
    match o.variant % SPECIES {
        0 => rubber(a, o, g, base, spread),
        12 => umbrella(a, o, g, base, spread),
        _ => branching(a, o, g, base, spread),
    }
}

fn rubber(a: &mut Assembly, o: &IndoorObject, g: &Growth, base: Vec3, spread: f32) {
    let mut r = stream(o.seed, 6450);
    let height = o.size.y - base.y;
    for j in 0..g.leader_count {
        let dir = radial(g.angle(j));
        let stature = if j == 0 {
            0.84
        } else {
            r.random_range(0.53..0.80)
        };
        let reach = spread * g.shoot_splay * (1. - stature * 0.6);
        let top = base + dir * reach + Vec3::Y * height * stature;
        let points = [base, base.lerp(top, 0.42) - dir * spread * 0.08, top];
        stem(a, &points, o.size.y * 0.009);
        let n = g.count(r.random_range(10..=13));
        for k in 0..n {
            // Last leaf is attached to the growing tip, not a bare extension.
            let t = g.crown_base + (1. - g.crown_base) * g.node(k, n - 1);
            let p = along(&points, t);
            let d = radial(g.angle(k) + j as f32 * 1.37);
            let root = p + d * spread * 0.065 + Vec3::Y * height * 0.012;
            stem(a, &[p, root], o.size.y * 0.0022);
            let length = spread * r.random_range(0.55..0.74) * (1. - t * 0.16);
            blade(
                a,
                root,
                incline(
                    d,
                    length,
                    r.random_range(-0.65..0.55),
                    root.y,
                    base.y,
                    o.size.y,
                ),
                spread * r.random_range(0.25..0.31) * (1. - t * 0.24),
                0,
                r.random(),
                g,
            );
        }
    }
}

fn branching(a: &mut Assembly, o: &IndoorObject, g: &Growth, base: Vec3, spread: f32) {
    let shrub = o.variant % SPECIES == 13;
    let mut r = stream(o.seed, 6451);
    let height = o.size.y - base.y;
    let top = base
        + Vec3::Y * height * (0.55 + g.crown_roundness * 0.15)
        + radial(g.shoot_phase) * spread * g.crown_skew;
    let trunk = [base, base.lerp(top, 0.40), top];
    if !shrub {
        stem(a, &trunk, o.size.y * 0.012);
    }
    let count = g.count(if shrub { 9 } else { 7 });
    for j in 0..count {
        let t = g.node(j, count - 1);
        let dir = radial(g.angle(j));
        let root = if shrub {
            base
        } else {
            along(&trunk, g.crown_base + (1. - g.crown_base) * t)
        };
        let reach = spread * (0.46 + g.shoot_splay * 0.18) * (1. - t * g.crown_roundness * 0.36);
        let end = if shrub {
            base + dir * reach + Vec3::Y * height * (0.47 + t * 0.38)
        } else {
            base + dir * reach
                + radial(g.shoot_phase) * spread * g.crown_skew * 0.25
                + Vec3::Y * height * (0.55 + t * 0.32)
        };
        let branch = [root, root.lerp(end, 0.52) + Vec3::Y * height * 0.025, end];
        stem(a, &branch, o.size.y * if shrub { 0.005 } else { 0.004 });
        let twigs = r.random_range(3..=4);
        for k in 0..twigs {
            let p = along(&branch, 0.38 + 0.62 * k as f32 / (twigs - 1) as f32);
            let angle = g.angle(j) + if k % 2 == 0 { 0.85 } else { -0.85 };
            let d = radial(angle);
            let tip = p
                + d * spread * r.random_range(0.10..0.17)
                + Vec3::Y * height * r.random_range(-0.012..0.025);
            stem(a, &[p, tip], o.size.y * 0.0018);
            let leaves = r.random_range(3..=5);
            for leaf in 0..leaves {
                let node = p.lerp(tip, 0.15 + 0.85 * leaf as f32 / (leaves - 1) as f32);
                let d = radial(angle + g.angle(leaf));
                let petiole = node + d * spread * 0.025;
                stem(a, &[node, petiole], o.size.y * 0.0010);
                let length = spread * r.random_range(0.33..0.41);
                blade(
                    a,
                    petiole,
                    incline(
                        d,
                        length,
                        r.random_range(-0.85..0.55),
                        petiole.y,
                        base.y,
                        o.size.y,
                    ),
                    length * r.random_range(0.33..0.43),
                    if shrub { 5 } else { 0 },
                    r.random(),
                    g,
                );
            }
        }
    }
}

fn umbrella(a: &mut Assembly, o: &IndoorObject, g: &Growth, base: Vec3, spread: f32) {
    let mut r = stream(o.seed, 6452);
    let height = o.size.y - base.y;
    for j in 0..g.leader_count {
        let dir = radial(g.angle(j));
        let end = base
            + dir * spread * g.shoot_splay * 0.42
            + Vec3::Y
                * height
                * if j == 0 {
                    0.84
                } else {
                    r.random_range(0.55..0.80)
                };
        let points = [base, base.lerp(end, 0.45) - dir * spread * 0.04, end];
        stem(a, &points, o.size.y * 0.006);
        let nodes = r.random_range(3..=5);
        for node in 0..nodes {
            let p = along(&points, 0.29 + 0.71 * g.node(node, nodes - 1));
            let d = radial(g.angle(node) + j as f32 * 1.2);
            let center = p + d * spread * 0.28 + Vec3::Y * height * 0.025;
            stem(a, &[p, center], o.size.y * 0.0020);
            let leaflets = r.random_range(6..=9);
            let pitch = r.random_range(-0.40..0.40);
            for k in 0..leaflets {
                let d = radial(k as f32 * TAU / leaflets as f32 + g.angle(node));
                blade(
                    a,
                    center,
                    incline(
                        d,
                        spread * r.random_range(0.44..0.59),
                        pitch + r.random_range(-0.12..0.12),
                        center.y,
                        base.y,
                        o.size.y,
                    ),
                    spread * r.random_range(0.115..0.14),
                    0,
                    r.random(),
                    g,
                );
            }
        }
    }
}
