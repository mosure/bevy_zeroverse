//! A graduated set at the back of an ensemble faces one connected playing area.
//! Sample an instrument line once. A failed bay leaves no drums or reservations.
use super::*;
use crate::scene::procedural_indoor::objects::public::music::timpani::{self, Timpani};

struct Kit {
    sizes: Vec<Vec3>,
    line: u64,
    splay: f32,
    gap: f32,
}
impl Kit {
    fn sample(r: &mut ChaCha8Rng) -> Self {
        let count = match r.random_range(0..10) {
            0 => 1,
            1..=6 => 2,
            _ => 3,
        };
        let start = r.random_range(0..=timpani::SIZES.len() - count);
        let height = r.random_range(0.825..0.865);
        let mut sizes: Vec<_> = timpani::SIZES[start..start + count]
            .iter()
            .map(|&inches| timpani::dimensions(inches, height + r.random_range(-0.006..0.006), r))
            .collect();
        if r.random_bool(0.5) {
            sizes.reverse();
        }
        Self {
            sizes,
            line: r.random(),
            splay: r.random_range(0.16..0.52),
            gap: r.random_range(0.060..0.16),
        }
    }
}

#[derive(Clone, Copy)]
struct Bay {
    across: bool,
    inner_edge: f32,
    outwards: f32,
    back: f32,
    rearward: f32,
    near_edge: f32,
    standing_depth: f32,
}
impl Bay {
    /// Across the bank from the aisle, then from the back wall toward the leader.
    fn world(self, across: f32, forward: f32) -> Vec2 {
        oriented(
            Vec2::new(
                self.inner_edge + self.outwards * across,
                self.back - self.rearward * forward,
            ),
            self.across,
        )
    }
    fn fit(
        self,
        scene: &mut IndoorManifest,
        index: usize,
        sizes: &[Vec3],
        kit: &Kit,
        r: &mut ChaCha8Rng,
        structure: &PlacementStructure,
    ) -> bool {
        let first = scene.objects.len();
        let mut edge = 0.07;
        for (i, &size) in sizes.iter().enumerate() {
            let fraction = if sizes.len() > 1 {
                2. * i as f32 / (sizes.len() - 1) as f32 - 1.
            } else {
                0.
            };
            let angle = -fraction * kit.splay * self.outwards;
            let (sin, cos) = angle.sin_cos();
            let forward = oriented(Vec2::new(sin, self.rearward * cos), self.across);
            let half_across = 0.5 * (size.x * cos + size.z * sin.abs());
            let half_along = 0.5 * (size.z * cos + size.x * sin.abs());
            let center = self.world(edge + half_across, self.near_edge + half_along + 0.045);
            edge += 2. * half_across + kit.gap;
            let proposal = FloorProposal {
                kind: ObjectKind::Timpani,
                position: Vec3::new(center.x, 0., center.y),
                size,
                yaw: forward.x.atan2(forward.y),
            };
            if scene
                .public_fit_at_offsets(index, proposal, r, structure, &[Vec2::ZERO], |o| {
                    o.finish_seed = Some(kit.line);
                    true
                })
                .is_none()
            {
                scene.objects.truncate(first);
                return false;
            }
        }
        let a = self.world(0., self.near_edge);
        let b = self.world(
            (edge - kit.gap + 0.08).max(0.84),
            self.near_edge - self.standing_depth,
        );
        let access = Reservation {
            role: ReservationRole::EquipmentAccess,
            min: a.min(b),
            max: a.max(b),
        };
        let center = (a + b) * 0.5;
        let extent = (a - b).abs();
        let probe = scene.floor_candidate(
            ObjectKind::Timpani,
            Vec3::new(center.x, 0., center.y),
            Vec3::new(extent.x, 1.92, extent.y),
            0.,
            r,
        );
        let zone = &scene.functional_programs()[index];
        if access.min.cmplt(zone.min).any()
            || access.max.cmpgt(zone.max).any()
            || !scene.placement_clear_with_structure(&probe, 0.02, false, structure)
            || scene
                .objects
                .iter()
                .filter(|o| o.solid && !o.neighbor)
                .any(|o| {
                    let (lo, hi) = o.bounds();
                    access.overlaps(lo, hi)
                })
            || !scene.objects[first..]
                .iter()
                .all(|o| pedal_reaches(scene, o, &access))
        {
            scene.objects.truncate(first);
            return false;
        }
        scene.assembly_diagnostics.programs[index]
            .reservations
            .push(access);
        true
    }
}

fn pedal_reaches(scene: &IndoorManifest, o: &IndoorObject, access: &Reservation) -> bool {
    // Stand behind the pedal, with room for both feet. This is in the actual
    // instrument frame, so it also detects a reversed archived installation.
    let feet = o
        .transform()
        .transform_point(Vec3::Z * (o.size.z * 0.5 + 0.36))
        .xz();
    feet.cmpge(access.min + 0.05).all()
        && feet.cmple(access.max - 0.05).all()
        && (scene.floor_height(feet) - o.position.y).abs() < 0.035
}

impl IndoorManifest {
    pub(in super::super) fn furnish_percussion(&mut self, index: usize) {
        if self.generator_version < timpani::VERSION {
            return;
        }
        let zone = &self.functional_programs()[index];
        let mut r = stream(self.seed, u64::from_le_bytes(*b"PercSite") ^ index as u64);
        if !r.random_bool(0.58) {
            return;
        }
        let aisle = &zone.reservations[0];
        let across = aisle.min.x == zone.min.x && aisle.max.x == zone.max.x;
        let min = oriented(zone.min, across);
        let max = oriented(zone.max, across);
        let inner = [oriented(aisle.min, across).x, oriented(aisle.max, across).x];
        let rearward = if oriented(zone.focus.xz(), across).y < (min.y + max.y) * 0.5 {
            1.
        } else {
            -1.
        };
        let back = if rearward > 0. { max.y } else { min.y };
        let kit = Kit::sample(&mut r);
        let structure = PlacementStructure::new();
        let first = usize::from(r.random_bool(0.5));
        // Existing seats and reading desks have priority. Narrow banks may fit
        // fewer drums. Both sides get a chance at the complete set first.
        for count in (1..=kit.sizes.len()).rev() {
            for i in [first, 1 - first] {
                for offset in [0., 0.61] {
                    let depth = r.random_range(0.82..1.06);
                    let bay = Bay {
                        across,
                        inner_edge: inner[i],
                        outwards: if i == 0 { -1. } else { 1. },
                        back,
                        rearward,
                        near_edge: depth + offset + r.random_range(0.13..0.52),
                        standing_depth: depth,
                    };
                    if bay.fit(self, index, &kit.sizes[..count], &kit, &mut r, &structure) {
                        return;
                    }
                }
            }
        }
    }
}

pub(in super::super::super) fn validate(
    scene: &IndoorManifest,
    zone: &ProgramAssembly,
) -> Result<(), &'static str> {
    let mut areas = zone
        .reservations
        .iter()
        .filter(|a| a.role == ReservationRole::EquipmentAccess);
    let access = areas.next();
    let mut sizes = [0u8; 3];
    let mut count = 0;
    let mut finish = None;
    for &id in &zone.members {
        let Some(o) = scene.objects.get(id) else {
            return Err("dangling ensemble member");
        };
        if o.kind != ObjectKind::Timpani {
            continue;
        }
        if !access.is_some_and(|a| pedal_reaches(scene, o, a)) {
            return Err("timpani pedal has no standing space");
        }
        if count == sizes.len()
            || o.finish_seed.is_none()
            || finish.is_some_and(|f| Some(f) != o.finish_seed)
        {
            return Err("inconsistent graduated timpani set");
        }
        sizes[count] = Timpani::nominal_size(o.size.x);
        count += 1;
        finish = o.finish_seed;
    }
    let sizes = &mut sizes[..count];
    sizes.sort_unstable();
    if sizes.windows(2).any(|pair| pair[1] - pair[0] != 3) {
        return Err("inconsistent graduated timpani set");
    }
    if let Some(access) = access {
        let aisle = zone
            .reservations
            .iter()
            .find(|a| a.role == ReservationRole::Circulation)
            .ok_or("percussion section lacks a connected aisle")?;
        if sizes.is_empty()
            || areas.next().is_some()
            || access.min.cmplt(zone.min).any()
            || access.max.cmpgt(zone.max).any()
            || (access.max - access.min).min_element() < 0.80
            || access.min.cmpgt(aisle.max + 0.001).any()
            || access.max.cmplt(aisle.min - 0.001).any()
        {
            return Err("percussion playing area is disconnected or undersized");
        }
    }
    Ok(())
}
