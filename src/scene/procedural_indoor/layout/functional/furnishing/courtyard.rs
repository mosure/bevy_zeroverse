//! An entrance walk goes through a real bay of the colonnade. Plant groups
//! and outdoor seats fit around that walk and the individual column bases.
use super::*;
use crate::scene::procedural_indoor::{
    envelope::courtyard::CourtyardRoof, objects::services::bench, plants,
};

pub(in crate::scene::procedural_indoor::layout::functional) fn routes(
    s: &IndoorManifest,
    roof: &CourtyardRoof,
) -> Vec<Reservation> {
    let half = s.room_size.xz() * 0.5;
    let mid = (roof.max + roof.min) * 0.5;
    let mut posts: Vec<_> = s
        .envelope
        .as_ref()
        .unwrap()
        .pillars
        .iter()
        .filter(|p| p.center.y > roof.max.y)
        .map(|p| p.center.x)
        .collect();
    posts.sort_unstable_by(f32::total_cmp);
    let gate = posts
        .windows(2)
        .map(|p| (p[0] + p[1]) * 0.5)
        .min_by(|x, y| (x - s.door_x).abs().total_cmp(&(y - s.door_x).abs()))
        .unwrap_or(mid.x);
    let w = (s.entry_width() * 0.44).clamp(0.50, 0.68);
    let front = half.y - 0.90;
    let path = |a: Vec2, b: Vec2| Reservation {
        role: ReservationRole::Circulation,
        min: a.min(b) - Vec2::splat(w),
        max: a.max(b) + Vec2::splat(w),
    };
    let entry = Vec2::new(s.door_x, half.y - 0.09 - w);
    let turn = Vec2::new(s.door_x, front);
    let bay = Vec2::new(gate, front);
    vec![
        path(entry, turn),
        path(turn, bay),
        path(bay, Vec2::new(gate, mid.y)),
    ]
}

impl IndoorManifest {
    pub(super) fn furnish_courtyard(&mut self) {
        let roof = self.envelope.as_ref().unwrap().courtyard.unwrap();
        let mut rng = stream(self.seed, u64::from_le_bytes(*b"CourtFit"));
        let half = self.room_size.xz() * 0.5;
        let opening = roof.max - roof.min;
        let center = (roof.min + roof.max) * 0.5;
        let stations = (self.density * 3.2 + rng.random_range(1.3..3.4)).round() as usize;
        self.assembly_diagnostics.programs.push(ProgramAssembly {
            zone: 0,
            activity: IndoorLayout::Courtyard,
            role: SpaceRole::Waiting,
            capacity_unit: CapacityUnit::SeatingStations,
            parameters: ProgramParameters::sample(self.seed, 0, self.density),
            min: -half,
            max: half,
            members: vec![],
            required: vec![ObjectKind::Plant],
            requested_capacity: stations,
            accepted_capacity: 0,
            rejected_candidates: 0,
            reservations: routes(self, &roof),
            services: vec![],
            focus: Vec3::new(center.x, 1.1, center.y),
        });
        // Site-compatible collections, with independent plant size, crown and
        // container manufacture. Leave negative space between distinct groups.
        let collection: &[u32] = match rng.random_range(0..3) {
            0 => &[6, 13, 10, 4],
            1 => &[15, 1, 14, 13, 3],
            _ => &[8, 9, 7, 13, 15],
        };
        let groups = (rng.random_range(1.4..2.7) + self.density * 1.8).round() as usize;
        let first = self.objects.len();
        let mut planted = Vec::new();
        let mut bed_rng = stream(self.seed, u64::from_le_bytes(*b"YardLand"));
        if bed_rng.random_bool(0.78) {
            let size = Vec3::new(
                bed_rng.random_range(2.15..3.80),
                bed_rng.random_range(1.30..2.95),
                bed_rng.random_range(1.08..1.85),
            );
            let species = collection[bed_rng.random_range(0..collection.len())];
            for _ in 0..8 {
                let at = center
                    + opening
                        * Vec2::new(
                            bed_rng.random_range(-0.32..0.32),
                            bed_rng.random_range(-0.38..0.20),
                        );
                let proposal = FloorProposal {
                    kind: ObjectKind::PlantingBed,
                    position: Vec3::new(at.x, 0., at.y),
                    size,
                    yaw: bed_rng.random_range(-PI..PI),
                };
                if let Some(id) = self.public_fit_with(0, proposal, &mut bed_rng, |o| {
                    o.variant = species;
                    let (lo, hi) = o.bounds();
                    lo.xz().cmpge(roof.min).all() && hi.xz().cmple(roof.max).all()
                }) {
                    planted.push(self.objects[id].position.xz());
                    self.assembly_diagnostics.programs[0].required[0] = ObjectKind::PlantingBed;
                    break;
                }
            }
        }
        // Trade a group of nursery containers for the island's planted crowns.
        for _ in planted.len()..groups {
            let group = center
                + opening * Vec2::new(rng.random_range(-0.37..0.37), rng.random_range(-0.40..0.40));
            let count = rng.random_range(1..=3);
            let mut anchor = None;
            for i in 0..count {
                let species = collection[rng.random_range(0..collection.len())];
                let scale = if i == 0 {
                    rng.random_range(1.15..1.82)
                } else {
                    rng.random_range(0.78..1.23)
                };
                let mut size = plants::sample_size(species, false, &mut rng) * scale;
                size.y = size.y.min(self.room_size.y - 0.40);
                for trial in 0..6 {
                    let at = match anchor {
                        Some(origin) => {
                            origin
                                + Vec2::from_angle(rng.random_range(-PI..PI))
                                    * rng.random_range(0.72..2.0)
                        }
                        None if trial == 0 => group,
                        None => {
                            center
                                + opening
                                    * Vec2::new(
                                        rng.random_range(-0.46..0.46),
                                        rng.random_range(-0.46..0.46),
                                    )
                        }
                    };
                    let yaw = rng.random_range(-PI..PI);
                    let proposal = FloorProposal {
                        kind: ObjectKind::Plant,
                        position: Vec3::new(at.x, 0., at.y),
                        size,
                        yaw,
                    };
                    if let Some(id) = self.public_fit_with(0, proposal, &mut rng, |o| {
                        o.variant = species;
                        true
                    }) {
                        let pos = self.objects[id].position.xz();
                        planted.push(pos);
                        anchor.get_or_insert(pos);
                        break;
                    }
                }
            }
        }
        if planted.is_empty() {
            // A narrow entrance reservation can reject the only sampled group
            // at minimum density. Fit a compact shrub towards a garden corner.
            for sign in [
                Vec2::ONE,
                -Vec2::ONE,
                Vec2::new(-1., 1.),
                Vec2::new(1., -1.),
            ] {
                let at = center + opening * sign * 0.36;
                let proposal = FloorProposal {
                    kind: ObjectKind::Plant,
                    position: Vec3::new(at.x, 0., at.y),
                    size: plants::sample_size(13, false, &mut rng),
                    yaw: rng.random_range(-PI..PI),
                };
                if let Some(id) = self.public_fit_with(0, proposal, &mut rng, |o| {
                    o.variant = 13;
                    true
                }) {
                    planted.push(self.objects[id].position.xz());
                    break;
                }
            }
        }
        let eased = |r: &mut ChaCha8Rng| r.random_range(0.12..0.88);
        for _ in 0..stations {
            let size = Vec3::new(
                rng.random_range(1.08..2.58),
                rng.random_range(0.62..1.02),
                rng.random_range(0.48..0.72),
            );
            for _ in 0..8 {
                let side = rng.random_range(0..4);
                let axis = side / 2;
                let other = 1 - axis;
                let mut pos = center;
                let low = side % 2 == 0;
                let edge = if low { roof.min[axis] } else { roof.max[axis] };
                let wall = if low { -half[axis] } else { half[axis] };
                pos[axis] = edge + (wall - edge) * rng.random_range(0.40..0.72);
                pos[other] = roof.min[other] + opening[other] * eased(&mut rng);
                let focus = planted
                    .iter()
                    .min_by(|a, b| a.distance_squared(pos).total_cmp(&b.distance_squared(pos)))
                    .copied()
                    .unwrap_or(center);
                let toward = focus - pos;
                let yaw = toward.x.atan2(toward.y) + rng.random_range(-0.32..0.32);
                let proposal = FloorProposal {
                    kind: ObjectKind::Bench,
                    position: Vec3::new(pos.x, 0., pos.y),
                    size,
                    yaw,
                };
                if self
                    .public_fit_with(0, proposal, &mut rng, |o| {
                        o.variant = bench::OUTDOOR;
                        bench::fit_height(o);
                        true
                    })
                    .is_some()
                {
                    break;
                }
            }
        }
        let p = &mut self.assembly_diagnostics.programs[0];
        p.members = (first..self.objects.len()).collect();
        p.accepted_capacity = p.realized_capacity(&self.objects);
    }
}
