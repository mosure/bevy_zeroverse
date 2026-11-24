//! Fleshy spiral crowns. Younger leaves emerge upright near the growing tip;
//! older leaves splay out and present their broad face towards the light.
use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;

#[derive(Clone, Copy, serde::Serialize)]
pub(crate) struct RosetteHabit {
    leaf_count: usize,
    divergence_radians: f32,
    outer_rise: f32,
    inner_rise: f32,
    half_width_fraction: f32,
    thickness_to_half_width: f32,
    shoulder: f32,
    cup: f32,
    tip_curl: f32,
    crown_rise: f32,
}

impl RosetteHabit {
    pub(crate) fn sample(seed: u64, growth: &Growth) -> Self {
        let mut r = stream(seed, 0x524f_5345_5454_4501);
        let handedness = if r.random_bool(0.5) { 1. } else { -1. };
        Self {
            leaf_count: growth.count(r.random_range(18..=34) + (growth.maturity * 9.) as usize),
            divergence_radians: handedness * (2.4 + (growth.phyllotaxis - 2.4) * 0.48),
            outer_rise: r.random_range(0.10..0.32),
            inner_rise: r.random_range(0.46..0.78),
            half_width_fraction: r.random_range(0.13..0.29),
            thickness_to_half_width: r.random_range(0.18..0.48),
            shoulder: r.random(),
            cup: r.random_range(0.03..0.36),
            tip_curl: r.random_range(-0.22..0.80),
            crown_rise: r.random_range(0.07..0.17),
        }
    }
}

struct Blade {
    root: Vec3,
    direction: Vec3,
    reach: f32,
    rise: f32,
    width: f32,
    bend: f32,
    texture_origin: Vec2,
}

const SIDES: usize = 8;
// Axial station, half-width of a lanceolate blade and half-width of a spoon-shaped blade.
const STATIONS: [(f32, f32, f32); 5] = [
    (0.12, 0.60, 0.30),
    (0.32, 1.00, 0.65),
    (0.54, 0.88, 0.94),
    (0.74, 0.57, 1.00),
    (0.89, 0.27, 0.70),
];
const DIAGONAL: f32 = std::f32::consts::FRAC_1_SQRT_2;
const SECTION: [(f32, f32); SIDES] = [
    (-1., 0.),
    (-DIAGONAL, DIAGONAL),
    (0., 1.),
    (DIAGONAL, DIAGONAL),
    (1., 0.),
    (DIAGONAL, -DIAGONAL),
    (0., -1.),
    (-DIAGONAL, -DIAGONAL),
];

impl Blade {
    fn write(&self, mesh: &mut Geometry, habit: &RosetteHabit) {
        let side = self.direction.cross(Vec3::Y);
        let tip = self.root + self.direction * self.reach + Vec3::Y * self.rise;
        let tangent = |t: f32| {
            (self.direction * self.reach + Vec3::Y * (self.rise + self.bend * (2. * t - 1.)))
                .normalize()
        };
        let mut rings = [[Vec3::ZERO; SIDES]; STATIONS.len()];
        let mut along = [0.; STATIONS.len()];
        let mut last = self.root;
        let mut distance = 0.;
        for (row, &(t, lance, spoon)) in STATIONS.iter().enumerate() {
            let center = self.root
                + self.direction * (self.reach * t)
                + Vec3::Y * (self.rise * t + self.bend * t * (t - 1.));
            distance += center.distance(last);
            along[row] = distance;
            last = center;
            let upper = side.cross(tangent(t));
            let width = self.width * lance.lerp(spoon, habit.shoulder);
            for (j, &(across, depth)) in SECTION.iter().enumerate() {
                let offset = across * across * habit.cup
                    + depth * habit.thickness_to_half_width * (t * 2.7).min(1.);
                rings[row][j] = center + width * (side * across + upper * offset);
            }
        }

        let start = mesh.positions.len() as u32;
        let mut vertex = |p: Vec3, n: Vec3, uv: Vec2| {
            mesh.positions.push(p.to_array());
            mesh.normals.push(n.to_array());
            mesh.uvs.push((self.texture_origin + uv).to_array());
        };
        vertex(self.root, -tangent(0.), Vec2::ZERO);
        for (row, ring) in rings.iter().enumerate() {
            let mut around = 0.;
            for j in 0..=SIDES {
                let col = j % SIDES;
                let left = ring[(col + SIDES - 1) % SIDES];
                let right = ring[(col + 1) % SIDES];
                let before = if row == 0 {
                    self.root
                } else {
                    rings[row - 1][col]
                };
                let after = rings.get(row + 1).map_or(tip, |next| next[col]);
                // Parameter derivatives of the actual cupped, tapered skin.
                let normal = (right - left).cross(after - before).normalize();
                if j > 0 {
                    around += ring[col].distance(left);
                }
                vertex(ring[col], normal, Vec2::new(around, along[row]));
            }
        }
        vertex(tip, tangent(1.), Vec2::Y * (distance + tip.distance(last)));
        let end = start + 1 + (STATIONS.len() * (SIDES + 1)) as u32;
        for j in 0..SIDES as u32 {
            let first = start + 1 + j;
            mesh.indices.extend([start, first + 1, first]);
            for row in 0..STATIONS.len() as u32 - 1 {
                let a = first + row * (SIDES as u32 + 1);
                let b = a + SIDES as u32 + 1;
                mesh.indices.extend([a, a + 1, b + 1, a, b + 1, b]);
            }
            let last = end - (SIDES as u32 + 1) + j;
            mesh.indices.extend([end, last, last + 1]);
        }
    }
}

#[derive(Clone, Copy)]
struct Crown {
    offset: Vec3,
    spread: f32,
    phase: f32,
    leaves: usize,
}

/// Roots stay inside the actual soil disk, including square and fluted pots.
/// A wide mouth can support a separate basal offset with its own growing tip.
fn crowns(
    seed: u64,
    growth: &Growth,
    habit: &RosetteHabit,
    spread: f32,
    soil_radius: f32,
) -> (Crown, Option<Crown>) {
    let mut r = stream(seed, 0x524f_5345_5454_4503);
    let axis = radial(r.random_range(0.0..TAU));
    let main_offset = (soil_radius * 0.62).min(spread * 0.40);
    let pup_offset = (soil_radius * 0.82).min(spread * 0.84);
    let separation = main_offset + pup_offset;
    // Keep a mature parent at least 60% of solitary crown scale. Narrow
    // openings take a single plant instead of shrinking a pair to fit.
    if separation > spread * 1.08 && r.random_bool(0.42) {
        // Each blade, including the curved margins and its offset node, lies
        // within 1.20 times the spread. Allocate disjoint horizontal disks.
        let available = separation * 0.98 / 1.20;
        let share = r.random_range(0.69..0.76);
        let leaves = (habit.leaf_count as f32 * r.random_range(0.24..0.34)).round() as usize;
        (
            Crown {
                offset: -axis * main_offset,
                spread: (available * share).min((1.24 * spread - main_offset) / 1.20),
                phase: growth.shoot_phase,
                leaves: habit.leaf_count - leaves,
            },
            Some(Crown {
                offset: axis * pup_offset,
                spread: available * (1. - share),
                phase: r.random_range(0.0..TAU),
                leaves,
            }),
        )
    } else {
        let offset = soil_radius.min(spread) * r.random_range(0.02..0.18);
        (
            Crown {
                offset: axis * offset,
                spread: spread - offset * 0.80,
                phase: growth.shoot_phase,
                leaves: habit.leaf_count,
            },
            None,
        )
    }
}

pub(super) fn build(
    a: &mut Assembly,
    o: &IndoorObject,
    growth: &Growth,
    soil: Vec3,
    spread: f32,
    soil_radius: f32,
    rim: f32,
) {
    let habit = RosetteHabit::sample(o.seed, growth);
    let mut r = stream(o.seed, 0x524f_5345_5454_4502);
    let (main, pup) = crowns(o.seed, growth, &habit, spread, soil_radius);
    let mesh = a.part(Surface::Succulent, "other_prop");
    for crown in std::iter::once(main).chain(pup) {
        let spread = crown.spread;
        let height = (o.size.y - rim).min(1.45 * spread) * 0.90;
        let crown_height = height * habit.crown_rise;
        let radius = (spread * 0.13).min(soil_radius * 0.35);
        // A thick, cupped lower leaf can dip below its node. Give its underside
        // room over the lip using the actual section profiles and curvature.
        let min_rise = height * habit.outer_rise * 0.90;
        let underside = STATIONS
            .iter()
            .map(|&(t, lance, spoon)| {
                let thickness = spread
                    * habit.half_width_fraction
                    * 1.10
                    * lance.lerp(spoon, habit.shoulder)
                    * habit.thickness_to_half_width
                    * (2.7 * t).min(1.);
                thickness - min_rise * t * (1. - (habit.tip_curl + 0.10).max(0.) * 0.80 * (1. - t))
            })
            .fold(0.0_f32, f32::max);
        let base = soil + crown.offset - Vec3::Y * (crown_height * 0.18);
        let collar = rim + underside;
        let apex = base.with_y(collar + crown_height);
        mesh.tapered_tube(&[base, apex], radius, radius * 0.40, 8);
        for j in 0..crown.leaves {
            let age = j as f32 / (crown.leaves - 1) as f32;
            let angle =
                crown.phase + j as f32 * habit.divergence_radians + r.random_range(-0.08..0.08);
            let direction = radial(angle);
            let y = collar + crown_height * (0.04 + 0.91 * age);
            let node = (y - base.y) / (apex.y - base.y);
            let root = base.with_y(y) + direction * (radius * (1. - 0.60 * node) * 0.55);
            let reach = spread * (0.95 - 0.83 * age * age) * r.random_range(0.89..1.05);
            let rise =
                height * habit.outer_rise.lerp(habit.inner_rise, age) * r.random_range(0.90..1.04);
            let width =
                spread * habit.half_width_fraction * (1. - 0.68 * age) * r.random_range(0.88..1.10);
            Blade {
                root,
                direction,
                reach,
                rise,
                width,
                bend: (0.45 * reach).min(0.80 * rise)
                    * (habit.tip_curl + r.random_range(-0.10..0.10)),
                texture_origin: Vec2::new(r.random_range(0.0..0.25), r.random_range(0.0..0.25)),
            }
            .write(mesh, &habit);
        }
    }
}

#[cfg(test)]
mod tests;
