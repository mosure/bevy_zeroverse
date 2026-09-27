//! Potted botanical forms with real leaf silhouettes, petioles and tapered stems.
//! The curved lamina/branch approach follows the review of bevy_ftb; geometry is
//! generated here independently, without its assets, shaders or runtime dependency.
use super::{
    layout::{stream, IndoorObject},
    materials::Surface,
    objects::Assembly,
};
use bevy::prelude::*;
use rand::Rng;
use std::f32::consts::{PI, TAU};

pub const SPECIES: u32 = 6;

/// Independent morphology controls; species constrains growth habit, not a mesh.
#[derive(Debug, Clone, serde::Serialize)]
pub(super) struct Growth {
    pub pot_fraction: f32,
    pub pot_radius: f32,
    pub pot_taper: f32,
    pub density: f32,
    pub phyllotaxis: f32,
}
impl Growth {
    pub fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 271);
        Self {
            pot_fraction: rng.random_range(0.19..0.30),
            pot_radius: rng.random_range(0.36..0.55),
            pot_taper: rng.random_range(0.60..0.94),
            density: rng.random_range(0.50..1.08),
            phyllotaxis: rng.random_range(2.18..2.62),
        }
    }
    fn count(&self, nominal: usize) -> usize {
        (nominal as f32 * self.density).round().max(3.0) as usize
    }
}

fn radial(angle: f32) -> Vec3 {
    Vec3::new(angle.cos(), 0.0, angle.sin())
}

fn stem(a: &mut Assembly, points: &[Vec3], radius: f32) {
    for (i, pair) in points.windows(2).enumerate() {
        let delta = pair[1] - pair[0];
        let r0 = radius * (1.0 - 0.65 * i as f32 / points.len() as f32);
        let r1 = radius * (1.0 - 0.65 * (i + 1) as f32 / points.len() as f32);
        a.part(Surface::Bark, "other_prop").lathe(
            &[
                (0.0, 0.0),
                (r0, 0.0),
                (r1, delta.length()),
                (0.0, delta.length()),
            ],
            8,
            Transform::from_translation(pair[0])
                .with_rotation(Quat::from_rotation_arc(Vec3::Y, delta.normalize())),
        );
    }
}

/// Each blade owns its UV domain: the central vein follows the midrib, even on
/// a twisted leaf. Area-weighted mesh normals follow the actual curved surface.
fn blade(a: &mut Assembly, base: Vec3, delta: Vec3, half_width: f32, shape: u32, seed: u64) {
    let axis = delta.normalize();
    let side = axis.cross(Vec3::Y).try_normalize().unwrap_or(Vec3::X);
    let up = side.cross(axis).normalize();
    let mut rng = stream(seed, 91);
    let bend = rng.random_range(0.18..0.50);
    let twist = rng.random_range(-0.30..0.30);
    let asymmetry = rng.random_range(-0.13..0.13);
    let exponent = if shape == 2 {
        rng.random_range(0.25..0.6)
    } else {
        rng.random_range(0.5..1.3)
    };
    let surface = if shape == 2 {
        Surface::LeafVariegated
    } else if seed.is_multiple_of(5) {
        Surface::LeafLight
    } else {
        Surface::Leaf
    };
    let g = a.part(surface, "other_prop");
    let first = g.positions.len() as u32;
    let mut positions = vec![base];
    let mut uvs = vec![[0.5, 0.0]];
    for row in 1..8 {
        let t = row as f32 / 8.0;
        let envelope = (PI * t).sin().powf(exponent);
        let lobes = if shape == 3 {
            0.48 + 0.52 * (t * PI * 4.0).cos().abs()
        } else {
            1.0
        };
        let mid = base + delta * t + up * half_width * (bend * (PI * t).sin() - 0.25 * t * t);
        for sign in [-1.0, 0.0, 1.0] {
            let w = half_width * envelope * lobes * sign * (1.0 + sign * asymmetry);
            positions.push(mid + side * w + up * (w * twist * t - w.abs() * 0.22));
            uvs.push([0.5 + sign * envelope * lobes * 0.5, t]);
        }
    }
    positions.push(base + delta - up * half_width * 0.25);
    uvs.push([0.5, 1.0]);
    let mut indices = vec![0, 2, 1, 0, 3, 2];
    for row in 0..6 {
        let k = 1 + row * 3;
        indices.extend([
            k,
            k + 1,
            k + 3,
            k + 1,
            k + 4,
            k + 3,
            k + 1,
            k + 2,
            k + 4,
            k + 2,
            k + 5,
            k + 4,
        ]);
    }
    indices.extend([19, 20, 22, 20, 21, 22]);
    let mut normals = vec![Vec3::ZERO; positions.len()];
    for tri in indices.as_chunks::<3>().0 {
        let [i, j, k] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let n = (positions[j] - positions[i]).cross(positions[k] - positions[i]);
        for index in [i, j, k] {
            normals[index] += n;
        }
    }
    g.positions
        .extend(positions.into_iter().map(|p| p.to_array()));
    g.normals
        .extend(normals.into_iter().map(|n| n.normalize().to_array()));
    g.uvs.extend(uvs);
    g.indices.extend(indices.into_iter().map(|i| first + i));
}

pub fn build(a: &mut Assembly, o: &IndoorObject) {
    let mut rng = stream(o.seed, 5);
    let h = o.size.y;
    let growth = Growth::sample(o.seed);
    let spread = o.size.x.min(o.size.z) * 0.44;
    let ph = h * growth.pot_fraction;
    let r = spread * growth.pot_radius;
    let pot = match (o.seed >> 8) % 3 {
        0 => Surface::Terracotta,
        1 => Surface::Concrete,
        _ => Surface::Ceramic,
    };
    let foot = ph * 0.035;
    a.part(pot, "other_prop").lathe(
        &[
            (0.0, foot),
            (r * growth.pot_taper, foot),
            (r * (growth.pot_taper + 0.025), ph * 0.14),
            (r, ph * 0.92),
            (r * 1.035, ph * 0.93),
            (r * 1.035, ph),
            (r * 0.89, ph),
            (r * 0.86, ph * 0.90),
            (r * (growth.pot_taper - 0.09), ph * 0.12),
            (0.0, ph * 0.12),
        ],
        32,
        Transform::IDENTITY,
    );
    a.part(pot, "other_prop").lathe(
        &[(0.0, 0.0), (r * 1.08, 0.0), (r * 1.08, foot), (0.0, foot)],
        32,
        Transform::IDENTITY,
    );
    let soil = ph * 0.86;
    a.part(Surface::Soil, "other_prop").cylinder(
        r * 0.89,
        h * 0.009,
        Transform::from_xyz(0.0, soil, 0.0),
    );
    for _ in 0..18 {
        let p = radial(rng.random_range(0.0..TAU)) * r * rng.random_range(0.12..0.82)
            + Vec3::Y * (soil + h * 0.008);
        a.part(Surface::Soil, "other_prop").cuboid(
            Vec3::splat(h * 0.009),
            h * 0.002,
            Transform::from_translation(p)
                .with_rotation(Quat::from_rotation_y(rng.random_range(0.0..TAU))),
        );
    }
    let base = Vec3::Y * soil;
    match o.variant % SPECIES {
        0 => {
            // Rubber plant: alternate broad leaves on several branching leaders.
            for leader in 0..growth.count(3) {
                let angle = leader as f32 * growth.phyllotaxis + rng.random_range(0.0..0.4);
                let top = base
                    + Vec3::Y * ((h - soil) * rng.random_range(0.64..0.84))
                    + radial(angle) * spread * 0.20;
                stem(
                    a,
                    &[
                        base,
                        base.lerp(top, 0.40) - radial(angle) * spread * 0.05,
                        top,
                    ],
                    h * 0.009,
                );
                for j in 0..growth.count(9) {
                    let p = base.lerp(top, 0.20 + j as f32 / growth.count(9) as f32 * 0.75);
                    let d = radial(angle + j as f32 * growth.phyllotaxis);
                    let petiole = p + d * spread * 0.12 + Vec3::Y * h * 0.025;
                    stem(a, &[p, petiole], h * 0.0025);
                    blade(
                        a,
                        petiole,
                        d * spread * rng.random_range(0.42..0.66)
                            + Vec3::Y * h * rng.random_range(-0.04..0.055),
                        spread * 0.19,
                        0,
                        rng.random(),
                    );
                }
            }
        }
        1 | 4 => {
            // Palm / fern: arching rachises with paired, tapered pinnae.
            let palm = o.variant == 1;
            // Keep the existing <20k-triangle per-plant budget at the lush tail.
            for frond in 0..growth.count(if palm { 14 } else { 16 }).min(16) {
                let dir = radial(frond as f32 * growth.phyllotaxis);
                let reach = spread * rng.random_range(0.76..0.95);
                let rise =
                    (h - soil) * rng.random_range(0.64..0.94) * if palm { 1.0 } else { 0.55 };
                let points: Vec<_> = (0..9)
                    .map(|i| {
                        let t = i as f32 / 8.0;
                        base + dir * reach * t + Vec3::Y * rise * (1.9 * t - 1.12 * t * t)
                    })
                    .collect();
                stem(a, &points, h * 0.0027);
                let side = Vec3::Y.cross(dir);
                for j in 2..16 {
                    let t = j as f32 / 17.0;
                    let p = base + dir * reach * t + Vec3::Y * rise * (1.9 * t - 1.12 * t * t);
                    let length = spread * (if palm { 0.56 } else { 0.48 }) * (PI * t).sin();
                    for sign in [-1.0, 1.0] {
                        blade(
                            a,
                            p,
                            side * sign * length + dir * length * 0.32 - Vec3::Y * length * 0.15,
                            length * if palm { 0.13 } else { 0.22 },
                            1,
                            rng.random(),
                        );
                    }
                }
            }
        }
        2 => {
            // Snake plant: upright, stiff, gently twisted variegated swords.
            for i in 0..growth.count(15) {
                let dir = radial(i as f32 * growth.phyllotaxis);
                let p = base + dir * r * 0.5;
                blade(
                    a,
                    p,
                    Vec3::Y * (h - soil) * rng.random_range(0.48..0.96) + dir * spread * 0.32,
                    spread * rng.random_range(0.075..0.13),
                    2,
                    rng.random(),
                );
            }
        }
        3 => {
            // Split-leaf tropical: independent arched petioles, deeply lobed blades.
            for i in 0..growth.count(11) {
                let dir = radial(i as f32 * growth.phyllotaxis);
                let top = base
                    + Vec3::Y * (h - soil) * rng.random_range(0.40..0.86)
                    + dir * spread * 0.20;
                stem(
                    a,
                    &[base, base.lerp(top, 0.55) - dir * spread * 0.09, top],
                    h * 0.004,
                );
                blade(
                    a,
                    top,
                    dir * spread * 0.69 - Vec3::Y * h * 0.07,
                    spread * 0.31,
                    3,
                    rng.random(),
                );
            }
        }
        _ => {
            // Dracaena: cane trunks and narrow drooping rosettes at distinct heights.
            for cane in 0..3 {
                let dir = radial(cane as f32 * growth.phyllotaxis);
                let start = base + dir * r * 0.35;
                let top = start + Vec3::Y * (h - soil) * (0.43 + cane as f32 * 0.19);
                stem(
                    a,
                    &[start, start.lerp(top, 0.5) + dir * spread * 0.05, top],
                    h * 0.013,
                );
                for j in 0..growth.count(19) {
                    let d = radial(j as f32 * growth.phyllotaxis + cane as f32);
                    blade(
                        a,
                        top,
                        d * spread * rng.random_range(0.45..0.77)
                            + Vec3::Y * h * rng.random_range(-0.18..0.08),
                        spread * 0.045,
                        1,
                        rng.random(),
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::layout::ObjectKind;
    #[test]
    fn all_botanical_forms_have_supported_pots_valid_leaf_frames_and_bounded_canopies() {
        for variant in 0..SPECIES {
            let mut topology_counts = std::collections::BTreeSet::new();
            for seed in 0..12 {
                let o = IndoorObject {
                    id: 0,
                    kind: ObjectKind::Plant,
                    position: Vec3::ZERO,
                    size: Vec3::new(0.9, 1.7, 0.9),
                    yaw: 0.0,
                    variant,
                    seed,
                    solid: true,
                    support: None,
                    neighbor: false,
                    interaction_target: None,
                };
                let mut a = Assembly::default();
                build(&mut a, &o);
                let (lo, hi) = a.bounds();
                assert!(
                    lo.y >= -0.001
                        && hi.y <= o.size.y
                        && lo.x >= -0.45
                        && hi.x <= 0.45
                        && lo.z >= -0.45
                        && hi.z <= 0.45,
                    "species {variant}: {lo:?} {hi:?}"
                );
                let mut triangles = 0;
                for g in a.parts.values() {
                    assert!(g
                        .normals
                        .iter()
                        .all(|n| (Vec3::from_array(*n).length() - 1.0).abs() < 0.001));
                    for t in g.indices.as_chunks::<3>().0 {
                        let p = t
                            .iter()
                            .map(|i| Vec3::from_array(g.positions[*i as usize]))
                            .collect::<Vec<_>>();
                        assert!((p[1] - p[0]).cross(p[2] - p[0]).length_squared() > 1e-18);
                    }
                    triangles += g.indices.len() / 3;
                }
                assert!(
                    (1000..20000).contains(&triangles),
                    "species {variant}: {triangles} triangles"
                );
                topology_counts.insert(triangles);
            }
            assert!(
                topology_counts.len() >= 4,
                "plant growth did not diversify topology: {variant}"
            );
        }
    }
}
