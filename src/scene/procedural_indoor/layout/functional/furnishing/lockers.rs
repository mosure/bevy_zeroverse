//! Install a classroom locker bank along a solid wall or a two-sided aisle.
//! Sample the product once, then test alternative poses and their access strips.
use super::*;
use crate::scene::procedural_indoor::{
    architecture::{facade, interior_glazing, trim::TrimProgram},
    envelope::polygon,
};
use rand::seq::SliceRandom;

impl IndoorManifest {
    pub(super) fn fit_classroom_lockers(&mut self, index: usize, r: &mut ChaCha8Rng) {
        let size = geometry::dimensions(ObjectKind::Locker, r.random());
        let mut bank = self.floor_candidate(ObjectKind::Locker, Vec3::ZERO, size, 0., r);
        bank.variant = geometry::education::WALL_BANK;
        bank.finish_seed = Some(self.seed.wrapping_add(index as u64));
        let mut placement = stream(self.seed, 0x4c4f_434b_5741_4c4c ^ index as u64);
        let Some(envelope) = &self.envelope else {
            return;
        };
        let mut walls: Vec<_> = envelope
            .walls
            .iter()
            .map(|w| (w.edge, w.facade.as_ref()))
            .collect();
        // The shared entrance partition has its own construction program and
        // intentionally has no entry in Envelope::walls.
        walls.extend(
            (0..envelope.footprint.len())
                .filter(|&edge| envelope.shared_edge(edge, self.room_size))
                .map(|edge| (edge, None)),
        );
        walls.shuffle(&mut placement);
        let region = self.assembly_diagnostics.programs[index].member_region(self, &bank);
        let baseboard = TrimProgram::for_scene(self)
            .and_then(|p| p.base.as_ref())
            .map_or(0., |p| p.depth_m);
        let shared = std::cell::OnceCell::new();
        for (edge, exterior) in walls {
            let n = envelope.footprint.len();
            let span = envelope.footprint[edge].distance(envelope.footprint[(edge + 1) % n]);
            let tf = envelope.wall_transform(edge);
            let yaw = tf.rotation.to_euler(EulerRot::YXZ).0;
            let extent = (span - size.x) * 0.5 - 0.14;
            if extent <= 0. {
                continue;
            }
            let at_entry = envelope.shared_edge(edge, self.room_size);
            let facade = if at_entry {
                Some(shared.get_or_init(|| {
                    interior_glazing::shared_facade(self, interior_glazing::partition_height(self))
                }))
            } else {
                exterior
            };
            // The shared entrance glazing is specified about the centre of the
            // whole room, even when the structural edge is only part of that wall.
            let origin = if at_entry { -tf.translation.x } else { 0. };
            let mut spans = wall_spans(extent, size, facade, origin);
            spans.shuffle(&mut placement);
            for (lo, hi) in spans {
                let phase = placement.random_range(0_f32..1.);
                for sample in 0..3 {
                    let u = (phase + sample as f32 / 3.).fract();
                    let x = lo + (hi - lo) * u;
                    let gap = baseboard.max(0.044) + placement.random_range(0.013..0.065);
                    let center = tf.transform_point(Vec3::new(x, 0., gap + size.z * 0.5));
                    bank.yaw = yaw;
                    bank.position = center.with_y(self.floor_height(center.xz()));
                    let (a, b) = bank.bounds();
                    if !a.xz().cmpge(region.0).all()
                        || !b.xz().cmple(region.1).all()
                        || !self.placement_clear(&bank, 0.035)
                        || facade::overlaps_opening(self, a, b, 0.045)
                        || (at_entry
                            && facade.is_some_and(|f| {
                                f.openings.iter().any(|w| {
                                    let reach = size.x * 0.5 + f.frame_width + 0.08;
                                    let local = x + origin;
                                    local + reach > w.min.x
                                        && local - reach < w.max.x
                                        && b.y > w.min.y - 0.08
                                        && a.y < w.max.y + 0.08
                                })
                            }))
                    {
                        continue;
                    }
                    // The door swing is already inside the physical bounds.
                    // Keep usable standing space directly in front of the locks.
                    let access = access_strip(&bank, placement.random_range(0.48..0.66), true);
                    if !self.locker_approach_clear(&bank, &access, region) {
                        continue;
                    }
                    self.objects.push(bank);
                    self.assembly_diagnostics.programs[index]
                        .reservations
                        .push(access);
                    return;
                }
            }
        }
        // A fully glazed or occupied perimeter can instead support a freestanding
        // bank. Only admit it when people can pass behind it and use the front.
        bank.variant = geometry::education::FREESTANDING_BANK;
        let program = &self.assembly_diagnostics.programs[index];
        let (min, max) = (program.min, program.max);
        let mut sides = [Vec3::Z, Vec3::X, Vec3::NEG_Z, Vec3::NEG_X];
        sides.shuffle(&mut placement);
        for inward in sides {
            let side = Vec3::Y.cross(inward);
            let half = (max - min) * 0.5;
            let lateral = side.xz().abs().dot(half) - size.x * 0.5 - 0.09;
            if lateral <= 0. {
                continue;
            }
            let phase = placement.random_range(0_f32..1.);
            for i in 0..3 {
                bank.yaw = inward.x.atan2(inward.z) + placement.random_range(-0.055..0.055);
                let along = 2. * (phase + i as f32 / 3.).fract() - 1.;
                let inset = placement.random_range(0.12..0.46);
                let behind = inward.xz().abs().dot(half) - size.z * 0.5 - inset;
                let c = (min + max) * 0.5 + side.xz() * along * lateral - inward.xz() * behind;
                bank.position = Vec3::new(c.x, self.floor_height(c), c.y);
                let (lo, hi) = bank.bounds();
                if !lo.xz().cmpge(min).all()
                    || !hi.xz().cmple(max).all()
                    || !self.placement_clear(&bank, 0.06)
                {
                    continue;
                }
                let front = access_strip(&bank, placement.random_range(0.50..0.66), true);
                let back = access_strip(&bank, placement.random_range(0.56..0.78), false);
                if self.locker_approach_clear(&bank, &front, region)
                    && self.locker_approach_clear(&bank, &back, region)
                {
                    self.objects.push(bank);
                    self.assembly_diagnostics.programs[index]
                        .reservations
                        .extend([front, back]);
                    return;
                }
            }
        }
    }
    fn locker_approach_clear(
        &self,
        bank: &IndoorObject,
        access: &Reservation,
        (min, max): (Vec2, Vec2),
    ) -> bool {
        let (lo, hi) = bank.bounds();
        let a = Vec3::new(access.min.x, bank.position.y, access.min.y);
        let b = Vec3::new(access.max.x, bank.position.y + 1.60, access.max.y);
        access.min.cmpge(min).all()
            && access.max.cmple(max).all()
            && !access.overlaps(lo, hi)
            && self.floor_support_clear(a, b)
            && self
                .envelope
                .as_ref()
                .is_some_and(|e| polygon::box_inside(&e.footprint, access.min, access.max, 0.035))
            && !self.entry_approach_overlaps(a, b)
            && self.program.as_ref().is_none_or(|p| p.portal_clear(a, b))
            && !self
                .objects
                .iter()
                .filter(|o| o.solid && o.id != bank.id)
                .any(|o| {
                    let (lo, hi) = o.bounds();
                    access.overlaps(lo, hi)
                })
            && !self
                .columns()
                .iter()
                .any(|(lo, hi)| access.overlaps(*lo, *hi))
    }
}

/// Allowable bank centers on one wall. Windows above the locker remain usable.
fn wall_spans(
    extent: f32,
    size: Vec3,
    facade: Option<&facade::Facade>,
    origin: f32,
) -> Vec<(f32, f32)> {
    let mut spans = vec![(-extent, extent)];
    if let Some(f) = facade {
        for opening in &f.openings {
            if opening.min.y > size.y + 0.08 {
                continue;
            }
            let reach = size.x * 0.5 + f.frame_width + 0.08;
            let cut = (
                opening.min.x - origin - reach,
                opening.max.x - origin + reach,
            );
            let mut next = Vec::with_capacity(spans.len() + 1);
            for (a, b) in spans {
                if a < cut.0 {
                    next.push((a, b.min(cut.0)));
                }
                if b > cut.1 {
                    next.push((a.max(cut.1), b));
                }
            }
            spans = next;
            if spans.is_empty() {
                break;
            }
        }
    }
    spans
}

fn access_strip(o: &IndoorObject, depth: f32, front: bool) -> Reservation {
    let forward = if front { Vec3::Z } else { Vec3::NEG_Z };
    let f = o.transform();
    let center = f
        .transform_point(forward * (o.size.z * 0.5 + 0.025 + depth * 0.5))
        .xz();
    let half = crate::scene::procedural_indoor::placement::rotated_half(
        Vec2::new(o.size.x * 0.91, depth),
        o.yaw,
    );
    Reservation {
        role: if front {
            ReservationRole::DoorApproach
        } else {
            ReservationRole::Circulation
        },
        min: center - half,
        max: center + half,
    }
}

#[cfg(test)]
mod tests;
