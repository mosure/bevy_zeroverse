//! Closed elliptical tresses, plaits, fringe and silhouette wisps. No alpha cards.
use super::{groom::Groom, Geometry, HairStyle, IndoorHuman};
use bevy::prelude::*;
use std::f32::consts::{PI, TAU};

pub(super) fn tube(
    g: &mut Geometry,
    rings: u32,
    sides: u32,
    width: f32,
    centre: impl Fn(f32) -> Vec3,
    radius: impl Fn(f32) -> f32,
    uv_offset: f32,
) {
    tress(
        g,
        rings,
        sides,
        Vec2::splat(width),
        centre,
        radius,
        uv_offset,
    );
}

pub(super) fn tress(
    g: &mut Geometry,
    rings: u32,
    sides: u32,
    section: Vec2,
    centre: impl Fn(f32) -> Vec3,
    radius: impl Fn(f32) -> f32,
    uv_offset: f32,
) {
    let base = g.positions.len() as u32;
    let mut distance = 0.;
    let mut previous = centre(0.);
    for row in 0..=rings {
        let t = row as f32 / rings as f32;
        let p = centre(t);
        distance += p.distance(previous);
        previous = p;
        let tangent =
            (centre((t + 0.001).min(1.)) - centre((t - 0.001).max(0.))).normalize_or(Vec3::NEG_Y);
        let x = (Vec3::X - tangent * tangent.x).normalize_or(Vec3::Z);
        let y = tangent.cross(x);
        for side in 0..=sides {
            let a = if side == sides {
                0.
            } else {
                TAU * side as f32 / sides as f32
            };
            let radial = x * (a.cos() * section.x) + y * (a.sin() * section.y);
            let n = (x * (a.cos() / section.x) + y * (a.sin() / section.y)).normalize_or(x);
            g.positions.push((p + radial * radius(t)).to_array());
            g.normals.push(n.to_array());
            g.uvs.push([
                TAU * side as f32 / sides as f32 * section.x + uv_offset,
                distance,
            ]);
        }
    }
    for row in 0..rings {
        for side in 0..sides {
            let a = base + row * (sides + 1) + side;
            let b = a + sides + 1;
            g.indices.extend([a, a + 1, b, b, a + 1, b + 1]);
        }
    }
    for row in [0, rings] {
        let cap = g.positions.len() as u32;
        let ring = base + row * (sides + 1);
        let t = row as f32 / rings as f32;
        g.positions.push(centre(t).to_array());
        g.normals.push(Vec3::Y.to_array());
        g.uvs.push([uv_offset, t * distance]);
        for side in 0..sides {
            if row == 0 {
                g.indices.extend([cap, ring + side + 1, ring + side]);
            } else {
                g.indices.extend([cap, ring + side, ring + side + 1]);
            }
        }
    }
}

/// Closed, thin backing fitted to the same paths as the outer loose locks.
/// It stays behind those locks and shares their metre-based fibre coordinates.
pub(super) fn backing(g: &mut Geometry, paths: &[Vec<Vec3>]) {
    if paths.len() < 2 {
        return;
    }
    let rows = paths[0].len();
    let layer = (rows * paths.len()) as u32;
    let base = g.positions.len() as u32;
    for offset in [-0.0045, -0.0020] {
        for path in paths {
            let mut arc = 0.;
            let mut previous = path[0];
            for &p in path {
                arc += p.distance(previous);
                previous = p;
                g.positions.push((p + Vec3::Z * offset).to_array());
                g.normals.push(Vec3::Z.to_array());
                g.uvs.push([p.x - paths[0][0].x, arc]);
            }
        }
    }
    let mut quad = |a, b, c, d| g.indices.extend([a, b, c, a, c, d]);
    for col in 0..paths.len() - 1 {
        for row in 0..rows - 1 {
            let a = base + (col * rows + row) as u32;
            let b = a + rows as u32;
            quad(a, b, b + 1, a + 1);
            quad(a + layer, a + layer + 1, b + layer + 1, b + layer);
        }
        for row in [0, rows - 1] {
            let a = base + (col * rows + row) as u32;
            let b = a + rows as u32;
            if row == 0 {
                quad(a, a + layer, b + layer, b);
            } else {
                quad(a, b, b + layer, a + layer);
            }
        }
    }
    for col in [0, paths.len() - 1] {
        for row in 0..rows - 1 {
            let a = base + (col * rows + row) as u32;
            if col == 0 {
                quad(a, a + 1, a + layer + 1, a + layer);
            } else {
                quad(a, a + layer, a + layer + 1, a + 1);
            }
        }
    }
}

pub(super) struct Root {
    pub point: Vec3,
    pub normal: Vec3,
}

pub(super) fn roots(
    h: &IndoorHuman,
    groom: &Groom,
    scalp: &[(Vec3, Vec3)],
    scale: f32,
) -> Vec<Root> {
    let style = HairStyle::from_id(h.hairstyle).unwrap();
    let count = if style == HairStyle::TwinBraids { 2 } else { 1 };
    let height = if style == HairStyle::HighPonytail {
        0.76
    } else {
        groom.program.tie_height
    };
    (0..count)
        .filter_map(|i| {
            let side = if count == 2 { i as f32 * 2. - 1. } else { 0. };
            let target = groom.centre
                + Vec3::new(
                    side * groom.size.x * 0.30,
                    groom.size.y * (height - 0.5),
                    groom.size.z * 0.5,
                );
            scalp
                .iter()
                .filter(|&&(p, _)| {
                    p.z > groom.centre.z
                        && groom.sample(p).1 > 0.022
                        && (side == 0. || (p.x - groom.centre.x) * side > 0.)
                })
                .min_by(|(a, _), (b, _)| {
                    a.distance_squared(target)
                        .total_cmp(&b.distance_squared(target))
                })
                .map(|&(p, normal)| Root {
                    // Bury the gathered base slightly in the rendered cap.
                    // Height is a world-metre offset; the rest mesh is scaled.
                    point: p + normal * (groom.sample(p).0 - 0.003) / scale,
                    normal,
                })
        })
        .collect()
}

pub(super) fn tied(h: &IndoorHuman, groom: &Groom, roots: &[Root], g: &mut Geometry) {
    let style = HairStyle::from_id(h.hairstyle).unwrap();
    let program = &groom.program;
    let length = program.drop_m + groom.size.y * 0.38;
    let width = (groom.size.x * 0.14 * groom.volume.sqrt()).clamp(0.018, 0.037);
    for root in roots {
        tube(
            g,
            6,
            12,
            width * 0.85,
            |t| root.point + root.normal * (0.012 * t) - Vec3::Y * (0.010 * t),
            |t| 0.80 + 0.20 * t,
            0.,
        );
    }
    if style == HairStyle::Bun {
        let Some(root) = roots.first() else { return };
        let origin = root.point;
        for k in 0..3 {
            tube(
                g,
                40,
                12,
                width * 0.38,
                |t| {
                    let a = t * TAU * 1.8 + k as f32 * TAU / 3.;
                    let r = width * (0.8 + 0.22 * (PI * t).sin());
                    origin + Vec3::new(r * a.sin(), r * a.cos(), width * 0.6 + t * width * 0.7)
                },
                |t| 1. - 0.55 * t.powi(8),
                k as f32 * 0.017,
            );
        }
        return;
    }
    for root in roots {
        let start = root.point;
        let axis = |t: f32| {
            start
                + Vec3::new(
                    program.sweep * 0.038 * t * t,
                    -length * t
                        + if style == HairStyle::HighPonytail {
                            0.065 * (PI * t).sin()
                        } else {
                            0.
                        },
                    (0.06 + width) * (PI * t * 0.75).sin(),
                )
        };
        let braided = matches!(style, HairStyle::Braid | HairStyle::TwinBraids);
        let strands = if braided { 3 } else { 7 };
        for k in 0..strands {
            let phase = k as f32 * TAU / strands as f32;
            tube(
                g,
                if braided { 72 } else { 40 },
                10,
                width * if braided { 0.48 } else { 0.33 },
                |t| {
                    let a = phase
                        + if braided {
                            t * length / 0.07 * TAU
                        } else {
                            t * groom.curl * TAU * 0.65
                        };
                    let gather = (t * length / 0.045).clamp(0., 1.);
                    let gather = gather * gather * (3. - 2. * gather);
                    let radius = width * (0.18 + 0.46 * gather) * (1. - 0.65 * t.powf(1.4));
                    axis(t) + Vec3::new(a.sin(), 0., a.cos()) * radius
                },
                |t| 0.95 - 0.73 * t.powf(2.4),
                k as f32 * 0.009,
            );
        }
    }
}

fn front_surface(xy: Vec2, triangles: &[[Vec3; 3]]) -> Option<Vec3> {
    let mut z = f32::INFINITY;
    for &[a, b, c] in triangles {
        let ab = (b - a).truncate();
        let ac = (c - a).truncate();
        let delta = xy - a.truncate();
        let det = ab.perp_dot(ac);
        if det.abs() < 1e-10 {
            continue;
        }
        let u = delta.perp_dot(ac) / det;
        let v = ab.perp_dot(delta) / det;
        if u >= -1e-5 && v >= -1e-5 && u + v <= 1.00001 {
            z = z.min(a.z + u * (b.z - a.z) + v * (c.z - a.z));
        }
    }
    z.is_finite().then_some(xy.extend(z))
}

pub(super) fn fringe(h: &IndoorHuman, groom: &Groom, triangles: &[[Vec3; 3]], g: &mut Geometry) {
    let style = HairStyle::from_id(h.hairstyle).unwrap();
    let coverage = if style == HairStyle::Pixie {
        0.36
    } else {
        groom.program.bangs.max(0.2)
    };
    for k in 0..9 {
        let f = (k as f32 - 4.) / 4.;
        let root = groom.centre.truncate()
            + Vec2::new(
                f * groom.size.x * 0.29 + groom.part * 0.012,
                groom.size.y * 0.44,
            );
        let end = groom.centre.truncate()
            + Vec2::new(
                f * groom.size.x * 0.39 + groom.program.sweep * 0.02,
                groom.size.y * (0.35 - coverage * 0.15) + f.abs() * 0.008,
            );
        // Project the combed path onto the actual upper forehead. Bounding-box
        // corners left horizontal free-floating spikes on smaller Anny heads.
        const STEPS: usize = 18;
        let points: Vec<_> = (0..=STEPS)
            .filter_map(|i| {
                let t = i as f32 / STEPS as f32;
                let mut xy = root.lerp(end, t);
                xy.x += groom.program.sweep * 0.006 * (PI * t).sin();
                front_surface(xy, triangles).map(|p| {
                    let n = ((p - groom.centre) / groom.size.powf(2.)).normalize_or(Vec3::NEG_Z);
                    p + n * (groom.sample(p).0 + 0.0028)
                })
            })
            .collect();
        if points.len() != STEPS + 1 {
            continue;
        }
        tress(
            g,
            STEPS as u32,
            10,
            Vec2::new(groom.size.x * 0.055, 0.0018),
            |t| {
                let s = t * STEPS as f32;
                let i = (s as usize).min(STEPS - 1);
                points[i].lerp(points[i + 1], s - i as f32)
            },
            |t| 0.85 - 0.50 * t.powi(3),
            k as f32 * 0.013,
        );
    }
}

/// Face-area normals after posing; the closed topology retains outward winding.
pub(super) fn recompute_normals(g: &mut Geometry, first_vertex: usize, first_index: usize) {
    let mut sums = vec![Vec3::ZERO; g.positions.len() - first_vertex];
    for tri in g.indices[first_index..].as_chunks::<3>().0 {
        let [a, b, c] = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
        let n = (b - a).cross(c - a);
        for &i in tri {
            sums[i as usize - first_vertex] += n;
        }
    }
    let mut welds = std::collections::HashMap::<[u32; 3], Vec3>::new();
    for (p, &sum) in g.positions[first_vertex..].iter().zip(&sums) {
        *welds.entry(p.map(f32::to_bits)).or_default() += sum;
    }
    for (p, n) in g.positions[first_vertex..]
        .iter()
        .zip(&mut g.normals[first_vertex..])
    {
        *n = welds[&p.map(f32::to_bits)].normalize_or(Vec3::Y).to_array();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::humans::{sample_person, HumanPoseKind};

    #[test]
    fn tied_roots_stay_on_individual_scalp_sides_at_extreme_tie_heights() {
        let mut h = sample_person(
            31,
            0,
            Vec3::ZERO,
            0.,
            HumanPoseKind::StandingRelaxed,
            None,
            false,
        );
        for size in [Vec3::new(0.135, 0.285, 0.18), Vec3::new(0.19, 0.34, 0.23)] {
            let centre = Vec3::new(0., 1.65, 0.);
            let scalp: Vec<_> = (1..24)
                .flat_map(|row| (0..48).map(move |col| (row, col)))
                .map(|(row, col)| {
                    let theta = PI * row as f32 / 24.;
                    let phi = TAU * col as f32 / 48.;
                    let direction = Vec3::new(
                        theta.sin() * phi.sin(),
                        theta.cos(),
                        theta.sin() * phi.cos(),
                    );
                    (
                        centre + direction * size * 0.5,
                        (direction / size).normalize(),
                    )
                })
                .collect();
            for style in [
                HairStyle::Bun,
                HairStyle::Ponytail,
                HairStyle::Braid,
                HairStyle::TwinBraids,
                HairStyle::HighPonytail,
            ] {
                h.hairstyle = style as u8;
                for height in [0., 0.18, 0.5, 0.78, 1.] {
                    h.appearance.as_mut().unwrap().hair_program.tie_height = height;
                    let groom = Groom::new(&h, centre - size * 0.5, centre + size * 0.5);
                    for scale in [0.8, 1.25] {
                        let roots = roots(&h, &groom, &scalp, scale);
                        assert_eq!(
                            roots.len(),
                            if style == HairStyle::TwinBraids { 2 } else { 1 }
                        );
                        if roots.len() == 2 {
                            assert!(roots[0].point.x < centre.x && roots[1].point.x > centre.x);
                            assert!(roots[0].point.distance(roots[1].point) > size.x * 0.2);
                        }
                        for root in roots {
                            let distance = scalp
                                .iter()
                                .map(|(p, _)| p.distance(root.point))
                                .fold(f32::INFINITY, f32::min);
                            assert!(distance < 0.018, "detached {style:?} root: {distance} m");
                            assert!(
                                groom.sample(root.point).1 > 0.01,
                                "root below the rendered hairline"
                            );
                        }
                    }
                }
            }
        }
    }
}
