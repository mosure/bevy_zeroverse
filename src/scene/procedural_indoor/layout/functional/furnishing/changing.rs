//! Dry changing rooms: install accessible locker fronts first, then fit benches
//! to the banks they serve without occupying the working space between them.
use super::*;
use crate::scene::procedural_indoor::objects::services::bench;

impl IndoorManifest {
    pub(super) fn changing_program(&mut self, index: usize, rng: &mut ChaCha8Rng) {
        let first = self.objects.len();
        let program = &mut self.assembly_diagnostics.programs[index];
        let area = (program.max - program.min).element_product();
        let target = (area * (0.044 + self.density * 0.047) * rng.random_range(0.84..1.20))
            .round()
            .clamp(3., 10.) as usize;
        program
            .required
            .extend([ObjectKind::Locker, ObjectKind::Bench]);
        program.requested_capacity = target;
        let bench_probability = rng.random_range(0.34..0.76);
        let structure = PlacementStructure::new();
        let include_island = rng.random_bool(0.59);
        let mut banks = 0;
        let mut benches = 0;
        for attempt in 0..target * 2 + 8 {
            if banks >= target {
                break;
            }
            let mut size = geometry::dimensions(ObjectKind::Locker, rng.random());
            size.x = (size.x * rng.random_range(1.16..1.68)).clamp(1.06, 2.60);
            size.z = rng.random_range(0.48..0.68);
            let site = u64::from_le_bytes(*b"LockSite").wrapping_add(attempt as u64);
            let Some(bank) = self.fit_locker_bank(index, size, site, &structure, rng) else {
                continue;
            };
            if benches == 0 || rng.random_bool(bench_probability) {
                if self.fit_changing_bench(index, &bank, rng, &structure) {
                    if benches == 0 {
                        let o = &self.objects[bank.object];
                        self.assembly_diagnostics.programs[index].focus = o
                            .transform()
                            .transform_point(Vec3::new(0., o.size.y * 0.60, o.size.z * 0.5));
                    }
                    benches += 1;
                } else if benches == 0 {
                    // The first station needs both parts. Give a different wall its
                    // own proposal, releasing exactly the rejected bank and its routes.
                    self.objects.truncate(bank.object);
                    self.assembly_diagnostics.programs[index]
                        .reservations
                        .truncate(bank.door_approach);
                    continue;
                }
            }
            banks += 1;
            if banks == 1 && include_island && self.fit_locker_island(index, rng, &structure) {
                banks += 2;
            }
        }
        // An occasional folded towel or personal bag occupies one end, leaving
        // the rest of the seat clear. It has its own object and support annotation.
        let mut belongings = stream(self.seed, u64::from_le_bytes(*b"LockLinn") ^ index as u64);
        for id in first..self.objects.len() {
            if self.objects[id].kind != ObjectKind::Bench {
                continue;
            }
            let seat = &self.objects[id];
            let chance = belongings.random::<f32>();
            let (kind, size) = if chance < 0.32 {
                (
                    ObjectKind::KitchenTowel,
                    Vec3::new(
                        belongings.random_range(0.21..0.32),
                        belongings.random_range(0.024..0.065),
                        seat.size.z * belongings.random_range(0.48..0.73),
                    ),
                )
            } else if chance < 0.47 && seat.size.x >= 1.30 {
                (
                    ObjectKind::Bag,
                    Vec3::new(
                        belongings.random_range(0.30..0.42),
                        belongings.random_range(0.23..0.37),
                        seat.size.z * belongings.random_range(0.46..0.64),
                    ),
                )
            } else {
                continue;
            };
            let seat = seat.clone();
            let side = if belongings.random_bool(0.5) { -1. } else { 1. };
            let offset = Vec3::X * seat.size.x * side * belongings.random_range(0.20..0.31);
            self.program_prop(
                &seat,
                kind,
                offset,
                size,
                belongings.random_range(-0.24..0.24),
                &mut belongings,
            );
        }
    }

    /// A factory pair shares one footprint datum and two outward-facing working
    /// fronts. Test both cabinets and both aisles before committing either one.
    fn fit_locker_island(
        &mut self,
        index: usize,
        rng: &mut ChaCha8Rng,
        structure: &PlacementStructure,
    ) -> bool {
        let program = &self.assembly_diagnostics.programs[index];
        let (lo, hi) = (program.min, program.max);
        let spine = &program.reservations[0];
        let cross = usize::from(spine.max.x - spine.min.x > spine.max.y - spine.min.y);
        let long = 1 - cross;
        let wings = [(lo[cross], spine.min[cross]), (spine.max[cross], hi[cross])];
        let size = Vec3::new(
            rng.random_range(1.30..2.42),
            rng.random_range(1.50..2.02),
            rng.random_range(0.50..0.63),
        );
        let yaw = if cross == 0 { FRAC_PI_2 } else { 0. };
        let front = Vec3::new(yaw.sin(), 0., yaw.cos());
        let separation = size.z + rng.random_range(0.048..0.072);
        let mut pair = [
            self.floor_candidate(ObjectKind::Locker, Vec3::ZERO, size, yaw, rng),
            self.floor_candidate(ObjectKind::Locker, Vec3::ZERO, size, yaw + PI, rng),
        ];
        let first = self.objects.len();
        for (i, o) in pair.iter_mut().enumerate() {
            o.id = first + i;
            o.variant = geometry::education::ISLAND_BANK;
            o.finish_seed = Some(self.seed.wrapping_add(index as u64));
            o.interaction_target = Some(first + (1 - i));
        }
        let phase = rng.random_range(0..2);
        for attempt in 0..10 {
            let (left, right) = wings[(phase + attempt) % 2];
            if right - left < separation + size.z + 0.45 {
                continue;
            }
            let mut center = Vec2::ZERO;
            center[cross] = left + (right - left) * rng.random_range(0.36..0.66);
            center[long] = lo[long] + (hi[long] - lo[long]) * rng.random_range(0.22..0.78);
            for (i, o) in pair.iter_mut().enumerate() {
                let side = if i == 0 { 0.5 } else { -0.5 };
                let at = center + front.xz() * (separation * side);
                o.position = Vec3::new(at.x, self.floor_height(at), at.y);
            }
            let depth = rng.random_range(0.66..0.91);
            let access = pair
                .each_ref()
                .map(|o| lockers::access_strip(o, depth, true));
            if pair.iter().zip(&access).all(|(o, a)| {
                let (b, t) = o.bounds();
                b.xz().cmpge(lo).all()
                    && t.xz().cmple(hi).all()
                    && self.placement_clear_with_structure(o, 0.035, false, structure)
                    && self.locker_approach_clear(o, a, (lo, hi), structure)
            }) {
                self.objects.extend(pair);
                self.assembly_diagnostics.programs[index]
                    .reservations
                    .extend(access);
                return true;
            }
        }
        false
    }

    fn fit_changing_bench(
        &mut self,
        index: usize,
        installation: &lockers::FittedLocker,
        rng: &mut ChaCha8Rng,
        structure: &PlacementStructure,
    ) -> bool {
        let bank = self.objects[installation.object].clone();
        let program = &self.assembly_diagnostics.programs[index];
        let region = program.member_region(self, &bank);
        let size = Vec3::new(
            (bank.size.x * rng.random_range(0.68..1.24)).clamp(0.95, 2.50),
            rng.random_range(0.405..0.485),
            rng.random_range(0.34..0.52),
        );
        // Sample the furniture once; bounded alternatives change only its placement.
        let product_seed = rng.random();
        let angle: f32 = rng.random_range(-0.095..0.095);
        let half_depth = 0.5 * (size.x * angle.sin().abs() + size.z * angle.cos());
        let tf = bank.transform();
        for _ in 0..6 {
            let gap = rng.random_range(0.78..1.20);
            let route = lockers::access_strip(&bank, gap, true);
            if !self.locker_approach_clear(&bank, &route, region, structure) {
                continue;
            }
            let position = tf.transform_point(Vec3::new(
                bank.size.x * rng.random_range(-0.26..0.26),
                0.,
                bank.size.z * 0.5 + 0.07 + gap + half_depth,
            ));
            let proposal = FloorProposal {
                kind: ObjectKind::Bench,
                position,
                size,
                yaw: bank.yaw + angle,
            };
            if self
                .public_fit_at_offsets(index, proposal, rng, structure, &[Vec2::ZERO], |o| {
                    o.variant = bench::CHANGING;
                    o.seed = product_seed;
                    o.interaction_target = Some(bank.id);
                    let (lo, hi) = o.bounds();
                    !route.overlaps(lo, hi)
                })
                .is_some()
            {
                let p = &mut self.assembly_diagnostics.programs[index];
                // Replace this bank's original, shorter door approach. A free-
                // standing bank's rear circulation is a distinct reservation.
                p.reservations[installation.door_approach] = route;
                return true;
            }
        }
        false
    }
}

/// Inspect the realized placement rather than re-running its random proposals.
pub(in crate::scene::procedural_indoor::layout::functional) fn validate(
    scene: &IndoorManifest,
    p: &ProgramAssembly,
) -> Result<(), &'static str> {
    if p.role != SpaceRole::Changing
        || p.capacity_unit != CapacityUnit::LockerBanks
        || p.accepted_capacity > p.requested_capacity
    {
        return Err("invalid changing-room locker capacity");
    }
    for o in p.members.iter().filter_map(|&id| scene.objects.get(id)) {
        if o.kind == ObjectKind::Locker && o.variant == geometry::education::ISLAND_BANK {
            let mate = o
                .interaction_target
                .and_then(|id| scene.objects.get(id))
                .filter(|mate| {
                    mate.kind == ObjectKind::Locker
                        && mate.variant == o.variant
                        && mate.interaction_target == Some(o.id)
                        && p.members.contains(&mate.id)
                })
                .ok_or("unpaired island locker")?;
            let relative = o.transform().rotation.inverse() * (mate.position - o.position);
            let back_gap = -relative.z - 0.5 * (mate.size.z + o.size.z);
            if relative.x.abs() > 0.002
                || relative.y.abs() > 0.002
                || (mate.yaw - o.yaw).cos() > -0.9999
                || !(0.046..0.074).contains(&back_gap)
            {
                return Err("island locker backs are misaligned");
            }
        }
        if o.kind != ObjectKind::Bench {
            continue;
        }
        let bank = o
            .interaction_target
            .and_then(|id| scene.objects.get(id))
            .filter(|bank| bank.kind == ObjectKind::Locker && p.members.contains(&bank.id))
            .ok_or("changing bench lacks its locker bank")?;
        let local = bank.transform().rotation.inverse() * (o.position - bank.position);
        let (sin, cos) = (o.yaw - bank.yaw).sin_cos();
        let gap = local.z - 0.5 * (bank.size.z + o.size.x * sin.abs() + o.size.z * cos.abs());
        let inside = bank
            .transform()
            .transform_point(Vec3::Z * (bank.size.z * 0.5 + gap - 0.10))
            .xz();
        if o.variant != bench::CHANGING
            || cos < 0.99
            || local.x.abs() > bank.size.x * 0.27
            || !(0.84..1.28).contains(&gap)
            || (o.position.y - bank.position.y).abs() > 0.015
            || !p.reservations.iter().any(|r| {
                r.role == ReservationRole::DoorApproach
                    && inside.cmpge(r.min).all()
                    && inside.cmple(r.max).all()
            })
        {
            return Err("changing bench misses the clear working front of its locker bank");
        }
    }
    Ok(())
}
