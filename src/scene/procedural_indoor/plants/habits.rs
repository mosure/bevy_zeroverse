//! Distinct botanical growth programs: every leaf is attached to a stem or crown.
//! All dimensions are constrained by the declared pot/canopy envelope.
use super::*;

fn succulent_leaf(a: &mut Assembly, base: Vec3, delta: Vec3, width: f32, thick: f32) {
    let length = delta.length();
    a.part(Surface::Succulent, "other_prop").lathe(
        &[
            (0., 0.),
            (width * 0.42, length * 0.10),
            (width, length * 0.38),
            (width * 0.73, length * 0.70),
            (width * 0.28, length * 0.90),
            (0., length),
        ],
        12,
        Transform::from_translation(base)
            .with_rotation(Quat::from_rotation_arc(Vec3::Y, delta.normalize()))
            .with_scale(Vec3::new(1., 1., thick)),
    );
}

pub(super) fn build(
    a: &mut Assembly,
    o: &IndoorObject,
    growth: &Growth,
    base: Vec3,
    spread: f32,
    soil_radius: f32,
) {
    let mut rng = stream(o.seed, 6240);
    let height = o.size.y - base.y;
    match o.variant % SPECIES {
        7 => {
            // Echeveria/agave-like fleshy rosettes with thick, closed leaf volumes.
            let rings = rng.random_range(3..=5);
            for ring in 0..rings {
                let count = growth.count(8 + ring * 2);
                let t = ring as f32 / rings as f32;
                for j in 0..count {
                    let dir = radial(j as f32 * TAU / count as f32 + ring as f32 * 0.43);
                    let length = spread * (0.72 - t * 0.35) * rng.random_range(0.88..1.06);
                    let root = base + Vec3::Y * height * (0.018 + t * 0.15);
                    succulent_leaf(
                        a,
                        root,
                        dir * length + Vec3::Y * height * (0.12 + t * 0.48),
                        spread * rng.random_range(0.065..0.10),
                        rng.random_range(0.28..0.58),
                    );
                }
            }
        }
        8 => {
            // Barrel/columnar cactus clusters: actual rounded ribs, not painted stripes.
            let count = (rng.random_range(2..=5) as f32 * growth.density)
                .round()
                .clamp(2., 5.) as usize;
            for j in 0..count {
                let dir = radial(
                    j as f32 * TAU / (count - 1) as f32
                        + growth.phyllotaxis
                        + rng.random_range(-0.10..0.10),
                );
                let p = base + dir * soil_radius * if j == 0 { 0. } else { 0.82 };
                let ch = height
                    * if j == 0 {
                        rng.random_range(0.63..0.93)
                    } else {
                        rng.random_range(0.30..0.68)
                    };
                let r = soil_radius
                    * if j == 0 {
                        rng.random_range(0.53..0.66)
                    } else {
                        rng.random_range(0.12..0.16)
                    };
                a.part(Surface::Succulent, "other_prop").fluted_lathe(
                    &[
                        (0., 0.),
                        (r * 0.80, 0.),
                        (r, ch * 0.10),
                        (r, ch * 0.65),
                        (r * 0.96, ch * 0.85),
                        (r * 0.72, ch * 0.96),
                        (0., ch),
                    ],
                    40,
                    rng.random_range(0.10..0.23),
                    rng.random_range(6..=12),
                    Transform::from_translation(p),
                );
            }
        }
        9 => {
            // Prickly pear pads form a connected, branching stack of flattened lobes.
            let count = rng.random_range(4..=7);
            let trunk_top = base + Vec3::Y * height * 0.48;
            a.part(Surface::Succulent, "other_prop")
                .rod(base, trunk_top, o.size.y * 0.010);
            for j in 0..count {
                let t = j as f32 / count as f32;
                let dir = radial(growth.angle(j));
                let p =
                    base + Vec3::Y * height * (0.13 + t * 0.57) + dir * spread * (0.06 + t * 0.28);
                let root = base.lerp(trunk_top, (0.15 + t).min(1.));
                a.part(Surface::Succulent, "other_prop")
                    .rod(root, p, o.size.y * 0.004);
                let radii = Vec3::new(
                    spread * rng.random_range(0.20..0.28),
                    height * 0.17,
                    spread * 0.05,
                );
                let g = a.part(Surface::Succulent, "other_prop");
                let first = g.uvs.len();
                g.ellipsoid(
                    radii,
                    Transform::from_translation(p).with_rotation(
                        Quat::from_rotation_y(j as f32 * growth.phyllotaxis)
                            * Quat::from_rotation_z(rng.random_range(-0.26..0.26)),
                    ),
                );
                for uv in &mut g.uvs[first..] {
                    uv[0] *= TAU * (radii.x * radii.z).sqrt();
                    uv[1] *= radii.y * 2.;
                }
            }
        }
        10 => {
            // Bamboo: multiple node-bearing culms and sprays of narrow lateral leaves.
            for j in 0..growth.count(5) {
                let dir = radial(growth.angle(j));
                let start = base + dir * soil_radius * 0.58;
                let top = start
                    + Vec3::Y * height * rng.random_range(0.66..0.95)
                    + dir * spread * growth.crown_skew;
                stem(a, &[start, start.lerp(top, 0.48), top], o.size.y * 0.0055);
                for k in 1..7 {
                    let p = start.lerp(top, k as f32 / 7.);
                    a.part(Surface::LeafLight, "other_prop").cylinder(
                        o.size.y * 0.0064,
                        o.size.y * 0.008,
                        Transform::from_translation(p),
                    );
                    if k < 3 {
                        continue;
                    }
                    let d = radial(j as f32 + k as f32 * growth.phyllotaxis);
                    let end = p + d * spread * 0.31 + Vec3::Y * height * 0.025;
                    stem(a, &[p, end], o.size.y * 0.0018);
                    for leaf in 0..4 {
                        let root = p.lerp(end, 0.25 + leaf as f32 * 0.23);
                        let dd = radial(
                            j as f32 + k as f32 * growth.phyllotaxis + (leaf as f32 - 1.5) * 0.65,
                        );
                        blade(
                            a,
                            root,
                            dd * spread * 0.31 - Vec3::Y * height * 0.015,
                            spread * 0.037,
                            1,
                            rng.random(),
                            growth,
                        );
                    }
                }
            }
        }
        _ => unreachable!("non-succulent/cane habit routed to succulent/cane builder"),
    }
}
