//! Potted botanical forms with real leaf silhouettes, petioles and tapered stems.
//! The curved lamina/branch approach follows the review of bevy_ftb; geometry is
//! generated here independently, without its assets, shaders or runtime dependency.
mod broadleaf;
mod leaf;
use leaf::{blade, incline};
mod habits;
mod palm;
mod pots;
mod sampling;
mod vines;
mod woody;
pub use pots::PotProfile;
pub(super) use sampling::{sample_size, sample_species};

use super::{
    layout::{stream, IndoorObject},
    materials::Surface,
    objects::Assembly,
};
use bevy::prelude::*;
use rand::Rng;
use std::f32::consts::{PI, TAU};

pub const SPECIES: u32 = 17;

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
        "bird_of_paradise",
        "fan_palm",
        "zz_plant",
    ][(variant % SPECIES) as usize]
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
    pub maturity: f32,
    pub crown_spread: f32,
    pub shoot_phase: f32,
    pub internode_exponent: f32,
    pub angular_irregularity: f32,
    pub leader_count: usize,
    pub leaf_curl: f32,
    pub leaf_lobes: f32,
    pub leaf_tip_exponent: f32,
    /// Height of the first woody branch, as a fraction of the live stem.
    pub crown_base: f32,
    /// Continuous rounded versus conical crown profile.
    pub crown_roundness: f32,
    /// Lower shoots splay farther; upper growth stays more upright.
    pub shoot_splay: f32,
}
impl Growth {
    pub fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 271);
        let mut age = stream(seed, 6283);
        let mut crown = stream(seed, 6440);
        let maturity: f32 = age.random_range(0.25..1.);
        let mut growth = Self {
            maturity,
            pot_fraction: rng.random_range(0.19..0.30),
            pot_radius: rng.random_range(0.36..0.55),
            pot_taper: rng.random_range(0.60..0.94),
            density: rng.random_range(0.78..1.08),
            phyllotaxis: rng.random_range(2.18..2.62),
            leaf_aspect: rng.random_range(0.88..1.28),
            crown_skew: rng.random_range(-0.16..0.16),
            droop: rng.random_range(0.08..0.42),
            branching: rng.random_range(0.38..0.92),
            crown_spread: crown.random_range(0.90..1.0),
            shoot_phase: crown.random_range(0.0..TAU),
            internode_exponent: crown.random_range(0.65..1.65),
            angular_irregularity: crown.random_range(0.04..0.38),
            leader_count: crown.random_range(2..=4),
            leaf_curl: crown.random_range(0.12..0.38),
            leaf_lobes: crown.random_range(3.0..6.0),
            leaf_tip_exponent: crown.random_range(0.60..1.25),
            crown_base: crown.random_range(0.18..0.34),
            crown_roundness: crown.random_range(0.35..0.85),
            shoot_splay: crown.random_range(0.36..0.72),
        };
        // Age drives connected leader/leaf occupancy jointly. Mature plants
        // should not independently draw both sparse foliage and weak branching.
        growth.density = (growth.density * 0.30 + (0.80 + maturity * 0.28) * 0.70).min(1.08);
        growth.branching = growth.branching * 0.45 + (0.45 + maturity * 0.40) * 0.55;
        growth
    }
    fn count(&self, nominal: usize) -> usize {
        (nominal as f32 * self.density).round().max(3.0) as usize
    }
    fn angle(&self, index: usize) -> f32 {
        self.shoot_phase
            + index as f32 * self.phyllotaxis
            + self.angular_irregularity * (index as f32 * 2.37).sin()
    }
    fn node(&self, index: usize, count: usize) -> f32 {
        let t = index as f32 / count as f32;
        t * 0.35 + t.powf(self.internode_exponent) * 0.65
    }
}

fn radial(angle: f32) -> Vec3 {
    Vec3::new(angle.cos(), 0.0, angle.sin())
}

/// Attach petioles to the emitted piecewise-linear stem, not an idealized curve.
fn along(points: &[Vec3], t: f32) -> Vec3 {
    let u = t.clamp(0., 1.) * (points.len() - 1) as f32;
    let i = (u.floor() as usize).min(points.len() - 2);
    points[i].lerp(points[i + 1], u - i as f32)
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

pub fn build(a: &mut Assembly, o: &IndoorObject) {
    let mut rng = stream(o.seed, 5);
    let h = o.size.y;
    let growth = Growth::sample(o.seed);
    // The foliage envelope includes maximum leaf-width modulation and droop.
    let spread = o.size.x.min(o.size.z) * 0.40;
    let (r, ph) = pot_dimensions(o, &growth);
    let pot = match stream(o.seed, 6441).random_range(0..5) {
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
    let spread = spread.min((h - soil) * 0.80) * growth.crown_spread;
    if matches!(o.variant % SPECIES, 0 | 6 | 12 | 13) {
        woody::build(a, o, &growth, base, spread);
        return;
    }
    if o.variant % SPECIES == 11 {
        vines::build(a, o, &growth, base, spread, soil_radius);
        return;
    }
    if o.variant % SPECIES >= 14 {
        broadleaf::build(a, o, &growth, base, spread, soil_radius);
        return;
    }
    if o.variant % SPECIES >= 6 {
        habits::build(a, o, &growth, base, spread, soil_radius);
        return;
    }
    match o.variant % SPECIES {
        1 | 4 => {
            // Palm / fern: arching rachises with paired, tapered pinnae.
            let palm = o.variant % SPECIES == 1;
            // Keep the existing <20k-triangle per-plant budget at the lush tail.
            for frond in 0..growth.count(if palm { 14 } else { 16 }).min(16) {
                let dir = radial(growth.angle(frond));
                let outward = rng.random_range(0.0..1.0);
                let reach = spread * (0.65 + outward * 0.31);
                let rise = (h - soil)
                    * if palm {
                        rng.random_range(0.55..0.92)
                    } else {
                        0.70 - outward * 0.25
                    };
                // Older outer fronds arch and descend; new central growth
                // stands higher. Avoid a flat bowl of identical radial ribs.
                let sag = if palm {
                    rng.random_range(1.05..1.35)
                } else {
                    1.18 + outward * 0.50
                };
                let points: Vec<_> = (0..9)
                    .map(|i| {
                        let t = i as f32 / 8.0;
                        base + dir * reach * t + Vec3::Y * rise * (1.9 * t - sag * t * t)
                    })
                    .collect();
                stem(a, &points, h * 0.0027);
                let side = Vec3::Y.cross(dir);
                let pinnae = rng.random_range(12..=15);
                for j in 2..pinnae {
                    let t = j as f32 / (pinnae + 1) as f32 + rng.random_range(-0.008..0.008);
                    let p = along(&points, t);
                    let length = spread * (if palm { 0.56 } else { 0.48 }) * (PI * t).sin();
                    for sign in [-1.0, 1.0] {
                        blade(
                            a,
                            p,
                            side * sign * length + dir * length * 0.32 - Vec3::Y * length * 0.15,
                            length
                                * if palm {
                                    0.13
                                } else {
                                    rng.random_range(0.22..0.27)
                                },
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
                let dir = radial(growth.angle(i));
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
                let dir = radial(growth.angle(i));
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
            for cane in 0..growth.leader_count {
                let dir = radial(growth.angle(cane));
                let start = base + dir * r * 0.35;
                let top = start
                    + Vec3::Y
                        * (h - soil)
                        * (0.43 + growth.node(cane, growth.leader_count - 1) * 0.38);
                stem(
                    a,
                    &[
                        start,
                        start.lerp(top, 0.5) + dir * spread * 0.05,
                        top,
                        top + Vec3::Y * h * 0.013,
                    ],
                    h * 0.013,
                );
                // Age-graded whorls form a crown, rather than a single sparse
                // star of blades at one point. Older outer leaves hang lower;
                // the compact centre retains upright new growth.
                for j in 0..growth.count(34) {
                    let tier = (j % 3) as f32;
                    let d = radial(j as f32 * growth.phyllotaxis + cane as f32);
                    blade(
                        a,
                        top + Vec3::Y * h * (tier - 1.) * 0.012,
                        d * spread * rng.random_range(0.52..0.79) * (1. - tier * 0.16)
                            + Vec3::Y * h * (rng.random_range(-0.13..-0.045) + tier * 0.075),
                        spread * rng.random_range(0.046..0.065),
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
mod tests;
