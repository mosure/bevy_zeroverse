//! Potted botanical forms with real leaf silhouettes, petioles and tapered stems.
//! The curved lamina/branch approach follows the review of bevy_ftb; geometry is
//! generated here independently, without its assets, shaders or runtime dependency.
mod habits;
mod pots;
pub use pots::PotProfile;

use super::{
    layout::{stream, IndoorObject},
    materials::Surface,
    objects::Assembly,
};
use bevy::prelude::*;
use rand::Rng;
use std::f32::consts::{PI, TAU};

pub const SPECIES: u32 = 14;

pub fn species_name(variant: u32) -> &'static str {
    [
        "rubber_plant",
        "palm",
        "snake_plant",
        "split_leaf",
        "fern",
        "dracaena",
        "branching_ficus",
        "succulent_rosette",
        "ribbed_cactus",
        "paddle_cactus",
        "bamboo",
        "trailing_vine",
        "umbrella_plant",
        "leafy_shrub",
    ][(variant % SPECIES) as usize]
}

/// Existing floor/tabletop plant proposals also need a compatible growth habit:
/// a two-metre narrow reserve must not turn into a vertically stretched rosette.
pub(super) fn sample_species(size: Vec3, rng: &mut impl Rng) -> u32 {
    let ratio = size.y / size.x.min(size.z);
    let ranges = [
        (1.1, 6.0),
        (1.2, 5.5),
        (1.3, 8.0),
        (0.7, 3.0),
        (0.65, 1.9),
        (1.6, 7.0),
        (1.3, 6.0),
        (0.30, 1.7),
        (0.75, 6.0),
        (0.70, 4.0),
        (1.8, 8.0),
        (0.70, 4.0),
        (1.2, 5.0),
        (0.70, 3.0),
    ];
    let mut choices = [0u32; SPECIES as usize];
    let mut count = 0;
    for (i, (min, max)) in ranges.into_iter().enumerate() {
        if (min..=max).contains(&ratio) && (i != 7 || size.y <= 0.70 || ratio <= 1.1) {
            choices[count] = i as u32;
            count += 1;
        }
    }
    if count == 0 {
        if ratio < 1. {
            7
        } else {
            2
        }
    } else {
        choices[rng.random_range(0..count)]
    }
}

/// Independent morphology controls; species constrains growth habit, not a mesh.
#[derive(Debug, Clone, serde::Serialize)]
pub(super) struct Growth {
    pub pot_fraction: f32,
    pub pot_radius: f32,
    pub pot_taper: f32,
    pub density: f32,
    pub phyllotaxis: f32,
    pub leaf_aspect: f32,
    pub crown_skew: f32,
    pub droop: f32,
    pub branching: f32,
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
            leaf_aspect: rng.random_range(0.72..1.28),
            crown_skew: rng.random_range(-0.16..0.16),
            droop: rng.random_range(0.08..0.42),
            branching: rng.random_range(0.38..0.92),
        }
    }
    fn count(&self, nominal: usize) -> usize {
        (nominal as f32 * self.density).round().max(3.0) as usize
    }
}

fn radial(angle: f32) -> Vec3 {
    Vec3::new(angle.cos(), 0.0, angle.sin())
}

/// Root volume scales with stature as well as canopy width. The saucer and
/// sculpted rim remain inside the existing placement reserve. Tall decorative
/// planters retain a separate continuous aspect range, rather than stretching
/// every tall plant's nursery pot vertically.
pub(super) fn pot_dimensions(o: &IndoorObject, growth: &Growth) -> (f32, f32) {
    // Cactus and succulent crowns do not have the spreading leaf canopy of a
    // ficus. Give their root volumes broad nursery pots rather than pencil pots.
    let form = match o.variant % SPECIES {
        7 => 1.35,
        8 | 9 => 1.55,
        _ => 1.0,
    };
    let mut rng = stream(o.seed, 6230);
    let width = o.size.x.min(o.size.z);
    let radius = (width * 0.40 * growth.pot_radius * form)
        .max(o.size.y * rng.random_range(0.055..0.085))
        .min(width * 0.40);
    let max_aspect = if rng.random_bool(0.22) {
        rng.random_range(1.15..1.65)
    } else {
        rng.random_range(0.68..1.15)
    };
    let height = (o.size.y * growth.pot_fraction).min(2. * radius * max_aspect);
    (radius, height)
}

pub(super) fn pot_radius(o: &IndoorObject, growth: &Growth) -> f32 {
    pot_dimensions(o, growth).0
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
fn blade(
    a: &mut Assembly,
    base: Vec3,
    delta: Vec3,
    half_width: f32,
    shape: u32,
    seed: u64,
    morphology: &Growth,
) {
    let axis = delta.normalize();
    let side = axis.cross(Vec3::Y).try_normalize().unwrap_or(Vec3::X);
    let up = side.cross(axis).normalize();
    let mut rng = stream(seed, 91);
    let half_width = half_width * morphology.leaf_aspect;
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
        let envelope = (PI * t).sin().powf(exponent)
            * if shape == 4 {
                (1.30 - t).clamp(0.35, 1.0)
            } else {
                1.0
            };
        let lobes = if shape == 3 {
            0.48 + 0.52 * (t * PI * 4.0).cos().abs()
        } else if shape == 5 {
            0.88 + 0.12 * (t * PI * 12.0).cos().abs()
        } else {
            1.0
        };
        let mid =
            base + delta * t + up * half_width * (bend * (PI * t).sin() - morphology.droop * t * t);
        for sign in [-1.0, 0.0, 1.0] {
            let w = half_width * envelope * lobes * sign * (1.0 + sign * asymmetry);
            positions.push(mid + side * w + up * (w * twist * t - w.abs() * 0.22));
            uvs.push([0.5 + sign * envelope * lobes * 0.5, t]);
        }
    }
    positions.push(base + delta - up * half_width * morphology.droop);
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
    // The foliage envelope includes maximum leaf-width modulation and droop.
    let spread = o.size.x.min(o.size.z) * 0.40;
    let (r, ph) = pot_dimensions(o, &growth);
    let pot = match (o.seed >> 8) % 5 {
        0 => Surface::Terracotta,
        1 => Surface::Concrete,
        2 => Surface::Plastic,
        3 => Surface::Metal,
        _ => Surface::Ceramic,
    };
    let pot_program = PotProfile::sample(o.seed);
    let (soil, soil_radius) = pots::build(a, pot, r, ph, growth.pot_taper, &pot_program);
    for _ in 0..18 {
        let p = radial(rng.random_range(0.0..TAU))
            * (soil_radius - h * 0.01).max(0.)
            * rng.random_range(0.12..0.92)
            + Vec3::Y * (soil + h * 0.008);
        a.part(Surface::Soil, "other_prop").cuboid(
            Vec3::splat(h * 0.009),
            h * 0.002,
            Transform::from_translation(p)
                .with_rotation(Quat::from_rotation_y(rng.random_range(0.0..TAU))),
        );
    }
    let base = Vec3::Y * soil;
    // Broad nursery pots need compact crowns: a horizontal leaf's rise must
    // fit the headroom too. Stem and pot dimensions retain their metric scale.
    let spread = spread.min((h - soil) * 0.80);
    if o.variant % SPECIES >= 6 {
        habits::build(a, o, &growth, base, spread, soil_radius);
        return;
    }
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
                        &growth,
                    );
                }
            }
        }
        1 | 4 => {
            // Palm / fern: arching rachises with paired, tapered pinnae.
            let palm = o.variant % SPECIES == 1;
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
                            &growth,
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
                    &growth,
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
                    &growth,
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
                        &growth,
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
    fn habit_selection_does_not_stretch_compact_plants_into_tall_floor_envelopes() {
        let mut tall = std::collections::BTreeSet::new();
        let mut compact = std::collections::BTreeSet::new();
        for seed in 0..256 {
            let mut rng = stream(seed, 6280);
            tall.insert(sample_species(Vec3::new(0.75, 1.80, 0.75), &mut rng));
            compact.insert(sample_species(Vec3::new(0.26, 0.22, 0.26), &mut rng));
        }
        assert!(!tall.contains(&7) && !tall.contains(&4));
        assert!(tall.len() >= 8 && compact.contains(&7) && compact.contains(&4));
        assert!(!compact.contains(&2) && !compact.contains(&5));
    }
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
                    cable_path: None,
                    cable_target: None,
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

    #[test]
    fn compact_and_wide_botanical_canopies_keep_actual_vertices_in_reserved_bounds() {
        for dimensions in [
            Vec3::new(0.18, 0.23, 0.16),
            Vec3::new(0.95, 2.12, 1.064),
            Vec3::new(0.45, 2.30, 0.45),
            Vec3::new(0.75, 0.40, 0.70),
        ] {
            for variant in 0..SPECIES {
                for seed in 0..12 {
                    let o = IndoorObject {
                        id: 0,
                        kind: ObjectKind::Plant,
                        position: Vec3::ZERO,
                        size: dimensions,
                        yaw: 0.,
                        variant,
                        seed,
                        solid: true,
                        support: None,
                        neighbor: false,
                        interaction_target: None,
                        cable_path: None,
                        cable_target: None,
                    };
                    let mut a = Assembly::default();
                    let growth = Growth::sample(o.seed);
                    let (radius, height) = pot_dimensions(&o, &growth);
                    assert_eq!((radius, height), pot_dimensions(&o, &growth));
                    assert!(height / (2. * radius) <= 1.65);
                    assert!(radius * 1.10 <= dimensions.x.min(dimensions.z) * 0.5);
                    if dimensions.y > 2.2 && dimensions.x < 0.5 {
                        assert!(radius * 2. >= 0.25, "insufficient tall-plant root volume");
                    }
                    build(&mut a, &o);
                    let (lo, hi) = a.bounds();
                    assert!(
                        lo.y >= -0.0001
                            && hi.y <= dimensions.y + 0.0001
                            && lo.x >= -dimensions.x * 0.5
                            && hi.x <= dimensions.x * 0.5
                            && lo.z >= -dimensions.z * 0.5
                            && hi.z <= dimensions.z * 0.5,
                        "{} seed {seed} size {dimensions:?}: {lo:?} {hi:?}",
                        species_name(variant)
                    );
                    for g in a.parts.values() {
                        assert!(g.normals.iter().all(|n| n.iter().all(|v| v.is_finite())
                            && (Vec3::from_array(*n).length() - 1.).abs() < 0.001));
                    }
                }
            }
        }
    }
}
