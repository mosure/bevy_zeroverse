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
        6 => {
            // Branching ficus: an asymmetric crown distributed along woody leaders.
            let top = base
                + Vec3::Y * height * 0.88
                + radial(rng.random_range(0.0..TAU)) * spread * growth.crown_skew;
            stem(a, &[base, base.lerp(top, 0.38), top], o.size.y * 0.012);
            let count = growth.count(5);
            for j in 0..count {
                let angle = j as f32 * growth.phyllotaxis;
                let start = base.lerp(top, 0.24 + 0.55 * j as f32 / count as f32);
                let end = start
                    + radial(angle) * spread * growth.branching * 0.48
                    + Vec3::Y * height * rng.random_range(0.10..0.22);
                stem(a, &[start, start.lerp(end, 0.6), end], o.size.y * 0.004);
                for k in 0..growth.count(8) {
                    let t = 0.2 + 0.8 * k as f32 / growth.count(8) as f32;
                    let p = start.lerp(end, t);
                    let d = radial(angle + k as f32 * growth.phyllotaxis);
                    let root = p + d * spread * 0.045;
                    stem(a, &[p, root], o.size.y * 0.0016);
                    blade(
                        a,
                        root,
                        d * spread * rng.random_range(0.20..0.39)
                            + Vec3::Y * height * rng.random_range(-0.03..0.04),
                        spread * 0.075,
                        0,
                        rng.random(),
                        growth,
                    );
                }
            }
        }
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
                let dir = radial(j as f32 * growth.phyllotaxis);
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
                let dir = radial(j as f32 * growth.phyllotaxis);
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
        11 => {
            // Pothos trained on a stake, with side shoots drooping inside its envelope.
            let top = base + Vec3::Y * height * 0.88;
            stem(a, &[base, top], o.size.y * 0.005);
            for shoot in 0..growth.count(7) {
                let dir = radial(shoot as f32 * growth.phyllotaxis);
                let root = base.lerp(top, 0.19 + 0.74 * shoot as f32 / growth.count(7) as f32);
                let reach = spread * rng.random_range(0.46..0.77);
                let points: Vec<_> = (0..7)
                    .map(|k| {
                        let t = k as f32 / 6.;
                        root + dir * reach * t
                            + Vec3::Y * height * (0.10 * (PI * t).sin() - 0.15 * t * t)
                    })
                    .collect();
                stem(a, &points, o.size.y * 0.002);
                for (k, &p) in points.iter().enumerate().take(7).skip(1) {
                    let d = radial(shoot as f32 * growth.phyllotaxis + k as f32 * 2.2);
                    let tip = p + d * spread * 0.19 - Vec3::Y * height * 0.022;
                    blade(a, p, tip - p, spread * 0.089, 4, rng.random(), growth);
                }
            }
        }
        12 => {
            // Schefflera: compound, radial leaflets at each terminal petiole.
            for j in 0..growth.count(7) {
                let dir = radial(j as f32 * growth.phyllotaxis);
                let end =
                    base + Vec3::Y * height * rng.random_range(0.38..0.83) + dir * spread * 0.36;
                stem(a, &[base, base.lerp(end, 0.45), end], o.size.y * 0.004);
                let leaflets = rng.random_range(5..=9);
                for k in 0..leaflets {
                    let d = radial(k as f32 * TAU / leaflets as f32 + j as f32);
                    blade(
                        a,
                        end,
                        d * spread * rng.random_range(0.33..0.50)
                            + Vec3::Y * height * rng.random_range(-0.018..0.05),
                        spread * 0.067,
                        0,
                        rng.random(),
                        growth,
                    );
                }
            }
        }
        _ => {
            // Dense woody/herbaceous shrub: leaves span the whole crown, not a top cap.
            for j in 0..growth.count(8) {
                let dir = radial(j as f32 * growth.phyllotaxis);
                let top = base
                    + Vec3::Y * height * rng.random_range(0.45..0.90)
                    + dir * spread * rng.random_range(0.15..0.41);
                stem(a, &[base, base.lerp(top, 0.44), top], o.size.y * 0.004);
                for k in 0..growth.count(12) {
                    let p = base.lerp(top, 0.22 + k as f32 / growth.count(12) as f32 * 0.75);
                    let d = radial(k as f32 * growth.phyllotaxis + j as f32);
                    let root = p + d * spread * 0.034;
                    stem(a, &[p, root], o.size.y * 0.0015);
                    blade(
                        a,
                        root,
                        d * spread * rng.random_range(0.15..0.32)
                            + Vec3::Y * height * rng.random_range(-0.02..0.04),
                        spread * 0.053,
                        5,
                        rng.random(),
                        growth,
                    );
                }
            }
        }
    }
}
