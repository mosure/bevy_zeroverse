//! The island, its growing surface and its crowns share one fitted footprint.
use super::*;
use crate::scene::procedural_indoor::{
    geometry::{Geometry, ProfileJoints, RevolvedOutline},
    layout::ObjectKind,
    objects::append_at,
};

mod collection;
pub(crate) use collection::Collection;
mod edge;
mod ground;
mod legacy;
#[cfg(test)]
mod tests;
use edge::Edging;
use ground::Ground;

const VERSION: u32 = 311;
const SIDES: usize = 32;

#[derive(serde::Serialize)]
struct Crown {
    species: u32,
    seed: u64,
    root_m: Vec3,
    yaw_radians: f32,
    width_m: f32,
    height_m: f32,
}

#[derive(serde::Serialize)]
pub(crate) struct Garden {
    collection: Collection,
    edging: Edging,
    ground: Ground,
    crowns: Vec<Crown>,
}

impl Garden {
    pub(crate) fn sample(o: &IndoorObject) -> Self {
        let collection = Collection::from_bed(o.variant);
        let edging = Edging::sample(o);
        let ground = Ground::sample(o, &edging, collection);
        let mut placement = stream(o.seed, u64::from_le_bytes(*b"YardSite"));
        let aspect = o.size.x.max(o.size.z) / o.size.x.min(o.size.z);
        let count = match placement.random_range(0.0..1.0) {
            p if p < 0.16 => 1,
            p if p < (0.47 + (aspect - 1.) * 0.12).min(0.79) => 3,
            _ => 2,
        };
        let phase = placement.random_range(-PI..PI);
        let mut crowns = Vec::with_capacity(count);
        for i in 0..count {
            // Each group can face any side of the bed. Stagger both azimuth
            // and radius, while keeping separate root balls even in a narrow trough.
            let separation = match (count, i) {
                (_, 0) => 0.,
                (2, _) => PI + placement.random_range(-0.55..0.55),
                (_, 1) => placement.random_range(1.63..2.42),
                _ => -placement.random_range(1.63..2.42),
            };
            let radius = if i == 0 {
                placement.random_range(0.0..0.61)
            } else {
                placement.random_range(0.42..0.73)
            };
            let at = Vec2::from_angle(phase + separation) * radius * ground.radii_m;
            let grade = ground.at(at);
            let mut growth = stream(o.seed, u64::from_le_bytes(*b"YardGrow") + i as u64);
            let species = if i == 0 {
                o.variant % SPECIES
            } else {
                let companions = collection.companions();
                companions[growth.random_range(0..companions.len())]
            };
            let headroom = o.size.y - grade.height_m;
            let fraction = if i == 0 {
                1.
            } else {
                growth.random_range(0.20..0.64)
            };
            // Fit one species-shaped envelope into the available space. A
            // rosette planted in a tall reserve must still grow as a rosette.
            let habit = sample_size(species, false, &mut growth);
            let aspect = habit.y / habit.xz().min_element();
            let width_m = (headroom * fraction / aspect)
                .min(grade.clearance_m * growth.random_range(1.55..1.95));
            crowns.push(Crown {
                species,
                seed: growth.random(),
                root_m: Vec3::new(at.x, grade.height_m, at.y),
                yaw_radians: growth.random_range(-PI..PI),
                width_m,
                height_m: width_m * aspect,
            });
        }
        Self {
            collection,
            edging,
            ground,
            crowns,
        }
    }

    fn build(self, a: &mut Assembly, o: &IndoorObject, version: u32) {
        self.edging.mesh(a, o.size.xz() * 0.5, &self.ground);
        self.ground.mesh(a);
        let mut child = o.clone();
        child.kind = ObjectKind::Plant;
        for crown in self.crowns {
            child.variant = crown.species;
            child.seed = crown.seed;
            child.size = Vec3::new(crown.width_m, crown.height_m, crown.width_m);
            let mut plant = Assembly::default();
            build_plant(&mut plant, &child, version, Roots::Bed);
            let bottom = 0.008 - crown.root_m.y;
            for g in plant.parts.values_mut() {
                if g.positions.iter().any(|v| v[1] < bottom) {
                    g.clip_half_space(|p| f64::from(p.y) - f64::from(bottom));
                }
            }
            let tf = Transform::from_translation(crown.root_m)
                .with_rotation(Quat::from_rotation_y(crown.yaw_radians));
            append_at(a, plant, tf);
        }
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    if version < VERSION {
        legacy::build(a, o, version);
    } else {
        Garden::sample(o).build(a, o, version);
    }
}

fn at_height(point: Vec2, height: f32) -> Vec3 {
    Vec3::new(point.x, height, point.y)
}
