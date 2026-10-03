//! Replayable spatial program: recursively split usable space, retain real portals,
//! then populate each leaf with dimensioned functional groups. All units are metres.
pub mod activity;
pub mod furnishing;
use super::{
    layout::{stream, IndoorLayout, IndoorManifest},
    materials::Surface,
    objects::Assembly,
};
use bevy::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Zone {
    #[serde(default)]
    pub composition: Option<activity::ActivityMix>,
    #[serde(default)]
    pub furnishing: Option<furnishing::FurnishingField>,
    pub min: Vec2,
    pub max: Vec2,
    pub activity: IndoorLayout,
    pub orientation: f32,
    pub aisle: f32,
    pub desk_width: f32,
    pub desk_depth: f32,
    pub seat_pitch: f32,
    pub occupancy: f32,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Partition {
    /// Axis normal to the wall: 0 = X, 1 = Z.
    pub axis: usize,
    pub coordinate: f32,
    pub start: f32,
    pub end: f32,
    pub door_center: f32,
    pub door_width: f32,
    pub door_height: f32,
    /// Elliptic arch rise; zero retains a rectangular portal.
    #[serde(default)]
    pub arch_rise: f32,
    pub thickness: f32,
    pub sill: f32,
    pub glazing_fraction: f32,
    #[serde(default = "default_mullion_pitch")]
    pub mullion_pitch: f32,
    #[serde(default)]
    pub transom_fraction: Option<f32>,
}
fn default_mullion_pitch() -> f32 {
    1.35
}
impl Partition {
    pub fn opening_height(&self, along: f32) -> f32 {
        let x = ((along - self.door_center) / (self.door_width * 0.5)).clamp(-1., 1.);
        self.door_height - self.arch_rise + self.arch_rise * (1. - x * x).max(0.).sqrt()
    }
    pub fn portal_obstacles(&self, height: f32) -> Vec<(Vec3, Vec3)> {
        let n = if self.arch_rise > 0. { 32 } else { 1 };
        (0..n)
            .map(|i| {
                let lo = self.door_center - self.door_width * 0.5
                    + self.door_width * i as f32 / n as f32;
                let hi = lo + self.door_width / n as f32;
                let floor = self.opening_height(lo).min(self.opening_height(hi)) - 0.025;
                let a = self.position(lo, floor);
                let b = self.position(hi, height);
                let pad =
                    if self.axis == 0 { Vec3::X } else { Vec3::Z } * (self.thickness * 0.5 + 0.02);
                (a - pad, b + pad)
            })
            .collect()
    }
    pub fn position(&self, along: f32, height: f32) -> Vec3 {
        if self.axis == 0 {
            Vec3::new(self.coordinate, height, along)
        } else {
            Vec3::new(along, height, self.coordinate)
        }
    }
    fn size(&self, length: f32, height: f32, depth: f32) -> Vec3 {
        if self.axis == 0 {
            Vec3::new(depth, height, length)
        } else {
            Vec3::new(length, height, depth)
        }
    }
    pub fn spans(&self) -> [(f32, f32); 2] {
        [
            (self.start, self.door_center - self.door_width * 0.5),
            (self.door_center + self.door_width * 0.5, self.end),
        ]
    }
    pub fn approach(&self) -> (Vec3, Vec3) {
        let p = self.position(self.door_center, 0.0);
        let half = self.size(self.door_width + 0.30, 0.0, 2.0) * 0.5;
        (p - half, p + half)
    }
    pub fn obstacles(&self, height: f32) -> Vec<(Vec3, Vec3)> {
        self.spans()
            .into_iter()
            .map(|(lo, hi)| {
                let p = self.position((lo + hi) * 0.5, height * 0.5);
                let half = self.size(hi - lo, height, self.thickness + 0.045) * 0.5;
                (p - half, p + half)
            })
            .collect()
    }
    pub fn build(&self, height: f32, a: &mut Assembly) {
        // A narrow opaque gasket closes the construction reveal at each wall
        // junction. Adjacent caps meet with opposite normals and no shared area.
        for along in [self.start + 0.003, self.end - 0.003] {
            a.box_part(
                Surface::Rubber,
                "wall",
                self.position(along, height * 0.5),
                self.size(0.006, height, self.thickness),
                0.0,
            );
        }
        for (index, (mut lo, mut hi)) in self.spans().into_iter().enumerate() {
            // Window posts must terminate beside the door jamb, not occupy the
            // same volume. A 4.5 mm construction reveal separates their side faces.
            if index == 0 {
                hi -= 0.022;
                lo += 0.006;
            } else {
                lo += 0.022;
                hi -= 0.006;
            }
            let length = hi - lo;
            let center = (lo + hi) * 0.5;
            let glazed = (height - self.sill - 0.16) * self.glazing_fraction;
            let top = self.sill + glazed;
            // Metal rails replace this strip of plaster. Identical thickness
            // plus overlapping extents made both surfaces compete in depth.
            let rail_half = if glazed > 0.01 { 0.0175 } else { 0.0 };
            for (bottom, upper) in [(0.0, self.sill - rail_half), (top + rail_half, height)] {
                if upper - bottom > 0.002 {
                    a.box_part(
                        Surface::Paint,
                        "wall",
                        self.position(center, (bottom + upper) * 0.5),
                        self.size(length, upper - bottom, self.thickness),
                        0.003,
                    );
                }
            }
            if glazed > 0.01 {
                a.box_part(
                    Surface::GlassInterior,
                    "window",
                    self.position(center, self.sill + glazed * 0.5),
                    self.size(
                        (length - 0.074).max(0.001),
                        (glazed - 0.039).max(0.001),
                        0.010,
                    ),
                    0.001,
                );
                let bays = (length / self.mullion_pitch).ceil().max(1.0) as u32;
                for i in 0..=bays {
                    a.box_part(
                        Surface::Metal,
                        "window",
                        self.position(
                            if i == 0 {
                                lo + 0.0175
                            } else if i == bays {
                                hi - 0.0175
                            } else {
                                lo + length * i as f32 / bays as f32
                            },
                            self.sill + glazed * 0.5,
                        ),
                        self.size(0.035, (glazed - 0.035).max(0.005), self.thickness),
                        0.002,
                    );
                }
                for y in [self.sill, top] {
                    a.box_part(
                        Surface::Metal,
                        "window",
                        self.position(center, y),
                        self.size(length, 0.035, self.thickness),
                        0.002,
                    );
                }
                if let Some(fraction) = self.transom_fraction {
                    for bay in 0..bays {
                        let lo = lo + length * bay as f32 / bays as f32 + 0.038;
                        let hi = lo + length / bays as f32 - 0.076;
                        if hi > lo {
                            a.box_part(
                                Surface::Metal,
                                "window",
                                self.position((lo + hi) * 0.5, self.sill + glazed * fraction),
                                self.size(hi - lo, 0.027, self.thickness),
                                0.002,
                            );
                        }
                    }
                }
            }
            a.box_part(
                Surface::WoodEdge,
                "wall",
                self.position(center, 0.05),
                self.size(length, 0.10, self.thickness + 0.035),
                0.002,
            );
        }
        if self.arch_rise > 0. {
            super::envelope::construction::arched_portal(self, height, a);
            return;
        }
        a.box_part(
            Surface::Paint,
            "wall",
            self.position(self.door_center, (height + self.door_height) * 0.5),
            self.size(self.door_width, height - self.door_height, self.thickness),
            0.003,
        );
        for along in [
            self.door_center - self.door_width * 0.5,
            self.door_center + self.door_width * 0.5,
        ] {
            a.box_part(
                Surface::WoodEdge,
                "door",
                self.position(along, (self.door_height - 0.0175) * 0.5),
                self.size(0.035, self.door_height - 0.0175, self.thickness + 0.02),
                0.002,
            );
        }
        a.box_part(
            Surface::WoodEdge,
            "door",
            self.position(self.door_center, self.door_height),
            self.size(self.door_width + 0.035, 0.035, self.thickness + 0.02),
            0.002,
        );
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndoorProgram {
    #[serde(default)]
    pub domain: Option<super::domain::SceneDomain>,
    pub zones: Vec<Zone>,
    pub partitions: Vec<Partition>,
    pub light_spacing: Vec2,
    pub light_phase: Vec2,
    pub light_drop: f32,
    pub fixture_size: Vec2,
    pub materials: Vec<super::materials::program::MaterialRecipe>,
    #[serde(default)]
    pub finishes: Option<super::architecture::details::FinishParameters>,
}
impl IndoorProgram {
    pub fn sample(seed: u64, size: Vec3, activity: IndoorLayout, density: f32) -> Self {
        let mut rng = stream(seed, 70);
        let domain = super::domain::SceneDomain::sample(seed);
        let half = Vec2::new(size.x, size.z) * 0.5;
        let mut leaves = vec![(-half, half)];
        let mut partitions: Vec<Partition> = Vec::new();
        let target_area = rng.random_range(14.0_f32.ln()..140.0_f32.ln()).exp();
        // Tall loft volumes reserve the long, open run needed for a real stair;
        // subdividing them first can make every mezzanine proposal impossible.
        let attempts = if size.y > 5.8 {
            0
        } else {
            ((size.x * size.z / target_area).round() as usize).clamp(1, 12) - 1
        };
        for _ in 0..attempts {
            let index = leaves
                .iter()
                .enumerate()
                .max_by(|(_, (a, b)), (_, (c, d))| {
                    ((b.x - a.x) * (b.y - a.y)).total_cmp(&((d.x - c.x) * (d.y - c.y)))
                })
                .unwrap()
                .0;
            let (lo, hi) = leaves[index];
            let extent = hi - lo;
            let axis = if extent.x > extent.y * 1.25 {
                0
            } else if extent.y > extent.x * 1.25 {
                1
            } else {
                rng.random_range(0..2)
            };
            if extent[axis] < 6.6 {
                continue;
            }
            let coordinate = rng.random_range(lo[axis] + 3.25..hi[axis] - 3.25);
            let along = 1 - axis;
            // A later branch may not terminate inside an existing portal or its
            // approach. This preserves connectivity through recursive subdivision.
            if partitions.iter().any(|p| {
                p.axis != axis
                    && (coordinate - p.door_center).abs() < p.door_width * 0.5 + 0.65
                    && (p.coordinate - lo[along])
                        .abs()
                        .min((p.coordinate - hi[along]).abs())
                        < 0.02
            }) {
                continue;
            }
            let width = rng.random_range(0.95..1.55);
            let door =
                rng.random_range(lo[along] + width * 0.5 + 0.30..hi[along] - width * 0.5 - 0.30);
            partitions.push(Partition {
                axis,
                coordinate,
                start: lo[along],
                end: hi[along],
                door_center: door,
                door_width: width,
                door_height: rng.random_range(2.08..2.35),
                arch_rise: 0.0,
                thickness: rng.random_range(0.10..0.18),
                sill: rng.random_range(0.10..1.35),
                glazing_fraction: if rng.random_bool(0.35) {
                    0.0
                } else {
                    rng.random_range(0.25..1.0)
                },
                mullion_pitch: rng.random_range(0.65..2.1),
                transom_fraction: rng.random_bool(0.45).then(|| rng.random_range(0.35..0.80)),
            });
            let mut mid_hi = hi;
            mid_hi[axis] = coordinate;
            let mut mid_lo = lo;
            mid_lo[axis] = coordinate;
            leaves[index] = (lo, mid_hi);
            leaves.push((mid_lo, hi));
        }
        let zones = leaves
            .into_iter()
            .enumerate()
            .map(|(i, (min, max))| {
                let activity = if i == 0 || rng.random_bool(0.68) {
                    activity
                } else {
                    IndoorLayout::PROFILES[rng.random_range(0..IndoorLayout::PROFILES.len())]
                };
                Zone {
                    composition: Some(activity::ActivityMix::sample(seed, i, activity)),
                    furnishing: Some(furnishing::FurnishingField::sample(seed, i, activity)),
                    min,
                    max,
                    activity,
                    orientation: rng.random_range(0..4) as f32 * std::f32::consts::FRAC_PI_2
                        + rng.random_range(-1.0..1.0) * (0.18 + 0.62 * domain.disorder),
                    aisle: rng.random_range(0.78..1.9),
                    desk_width: rng.random_range(0.95..2.25),
                    desk_depth: rng.random_range(0.56..1.05),
                    seat_pitch: rng.random_range(0.78..1.45),
                    occupancy: (0.40 + density * 0.60) * rng.random_range(0.72..1.0),
                }
            })
            .collect();
        Self {
            zones,
            partitions,
            light_spacing: Vec2::new(rng.random_range(2.1..5.3), rng.random_range(2.1..5.3)),
            light_phase: Vec2::new(rng.random_range(-0.15..0.15), rng.random_range(-0.15..0.15)),
            light_drop: domain.ceiling_relief + rng.random_range(0.12..0.40),
            fixture_size: Vec2::new(rng.random_range(0.3..1.6), rng.random_range(0.10..0.65)),
            materials: super::materials::program::sample(seed),
            finishes: Some(super::architecture::details::FinishParameters::sample(seed)),
            domain: Some(domain),
        }
    }
    pub fn portal_clear(&self, lo: Vec3, hi: Vec3) -> bool {
        !self.partitions.iter().any(|p| {
            let (a, b) = p.approach();
            lo.x < b.x && hi.x > a.x && lo.z < b.z && hi.z > a.z
        })
    }
}

pub fn primary_zone(scene: &IndoorManifest) -> Option<(Vec3, Vec2)> {
    scene
        .program
        .as_ref()?
        .zones
        .iter()
        .max_by(|a, b| {
            let x = a.max - a.min;
            let y = b.max - b.min;
            (x.x * x.y).total_cmp(&(y.x * y.y))
        })
        .map(|z| {
            let c = (z.min + z.max) * 0.5;
            (Vec3::new(c.x, 0.0, c.y), z.max - z.min)
        })
}

impl IndoorProgram {
    pub fn validate(&self, scene: &IndoorManifest) -> Result<(), String> {
        if let Some(niche) = self.finishes.as_ref().and_then(|f| f.niche.as_ref()) {
            niche.validate()?;
        }
        if let Some(domain) = &self.domain {
            domain.validate()?;
        }
        if self.zones.is_empty()
            || self.zones.len() > 16
            || if scene.envelope.is_some() {
                self.partitions.len() >= self.zones.len()
            } else {
                self.partitions.len() + 1 != self.zones.len()
            }
        {
            return Err("invalid spatial program topology".into());
        }
        let half = Vec2::new(scene.room_size.x, scene.room_size.z) * 0.5;
        let mut area = 0.0;
        for (i, zone) in self.zones.iter().enumerate() {
            if let Some(mix) = &zone.composition {
                mix.validate()?;
            }
            if let Some(field) = &zone.furnishing {
                field.validate()?;
            }
            let size = zone.max - zone.min;
            if !zone.min.is_finite()
                || !zone.max.is_finite()
                || size.min_element() < 3.0
                || zone.min.cmplt(-half - Vec2::splat(0.001)).any()
                || zone.max.cmpgt(half + Vec2::splat(0.001)).any()
                || !zone.orientation.is_finite()
                || !(0.0..=1.0).contains(&zone.occupancy)
            {
                return Err("invalid spatial program zone".into());
            }
            area += size.x * size.y;
            for other in &self.zones[i + 1..] {
                let overlap = zone.max.min(other.max) - zone.min.max(other.min);
                if overlap.min_element() > 0.001 {
                    return Err("spatial program zones overlap".into());
                }
            }
        }
        if (area - scene.room_size.x * scene.room_size.z).abs() > 0.01 {
            return Err("spatial program does not cover envelope".into());
        }
        for p in &self.partitions {
            if p.axis > 1
                || !(0.55..=2.4).contains(&p.mullion_pitch)
                || p.transom_fraction
                    .is_some_and(|f| !(0.2..=0.85).contains(&f))
                || ![
                    p.coordinate,
                    p.start,
                    p.end,
                    p.door_center,
                    p.door_width,
                    p.door_height,
                    p.arch_rise,
                    p.thickness,
                    p.sill,
                    p.glazing_fraction,
                ]
                .into_iter()
                .all(f32::is_finite)
                || p.start >= p.end
                || !(0.9..=1.7).contains(&p.door_width)
                || p.door_center - p.door_width * 0.5 < p.start + 0.25
                || p.door_center + p.door_width * 0.5 > p.end - 0.25
                || p.door_height < 2.0
                || p.door_height > scene.room_size.y - 0.1
                || p.arch_rise < 0.0
                || p.arch_rise > 0.8
                || p.door_height - p.arch_rise < 2.0
                || !(0.0..=1.0).contains(&p.glazing_fraction)
            {
                return Err("invalid partition or portal".into());
            }
        }
        if self.materials.len() != super::materials::program::SURFACES.len()
            || !self.light_spacing.is_finite()
            || self.light_spacing.min_element() < 1.0
            || !self.fixture_size.is_finite()
            || self.fixture_size.min_element() < 0.08
            || !self.light_drop.is_finite()
        {
            return Err("invalid material or lighting program".into());
        }
        for (i, r) in self.materials.iter().enumerate() {
            r.validate()?;
            if r.surface as usize != i {
                return Err("material roles are not in canonical order".into());
            }
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn program_replays_covers_space_and_has_clear_portals() {
        let mut signatures = std::collections::HashSet::new();
        let mut max_zones = 0;
        for seed in 0..512 {
            let scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.7, 0, 0.0)
                    .unwrap();
            let p = scene.program.as_ref().unwrap();
            p.validate(&scene).unwrap();
            max_zones = max_zones.max(p.zones.len());
            signatures
                .insert(serde_json::to_string(&(p.zones.clone(), p.partitions.clone())).unwrap());
            for portal in &p.partitions {
                let midpoint = portal.position(portal.door_center, 1.0);
                for (lo, hi) in super::super::floorplan::obstacles(&scene) {
                    assert!(
                        !(midpoint.cmpgt(lo).all() && midpoint.cmplt(hi).all()),
                        "portal blocked seed={seed}"
                    );
                }
            }
        }
        assert_eq!(signatures.len(), 512);
        assert!(max_zones >= 4);
    }
}
